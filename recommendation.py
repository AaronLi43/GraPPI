# recommendation.py
import json
import asyncio
import networkx as nx
import nest_asyncio
from typing import List, Dict, Tuple, Any, Optional
import sys
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
import numpy as np
from .llm_utils import (
    print_flush,
    extract_string_value,
    clean_json_output,
    retry_with_exponential_backoff
)
from .prompts import edge_relevance_prompt, path_explanation_prompt

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class ProteinRecommender:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # Initialize chains
        self.edge_relevance_chain = (
            {"query": RunnablePassthrough(), 
             "start_protein": RunnablePassthrough(),
             "end_protein": RunnablePassthrough(),
             "start_annotation": RunnablePassthrough(),
             "end_annotation": RunnablePassthrough()}
            | edge_relevance_prompt
            | self.llm
            | StrOutputParser()
        )

        self.path_explanation_chain = (
            {"query": RunnablePassthrough(),
             "path": RunnablePassthrough(),
             "path_details": RunnablePassthrough()}
            | path_explanation_prompt
            | self.llm
            | StrOutputParser()
        )

    async def score_edge(self, G: nx.Graph, start: str, end: str, query: str) -> str:
        start_annotation = G.nodes[start]['annotation']
        end_annotation = G.nodes[end]['annotation']

        try:
            print_flush(f"Scoring edge {start} -> {end}")
            result = await retry_with_exponential_backoff(
                self.edge_relevance_chain.ainvoke,
                {
                    "query": query,
                    "start_protein": start,
                    "end_protein": end,
                    "start_annotation": start_annotation,
                    "end_annotation": end_annotation
                }
            )

            result = extract_string_value(result)
            print_flush(f"Debug - LLM output for edge {start} -> {end}:")
            print_flush(f"Content: {result}")
            explanation = result

        except Exception as e:
            print_flush(f"Error in scoring edge {start} -> {end}: {str(e)}")
            explanation = f"Error in scoring: {str(e)}"

        return explanation

    async def score_path(self, G: nx.Graph, path: List[str], query: str) -> Tuple[float, str, str]:
        path_details = ""

        edge_tasks = []
        for i in range(len(path) - 1):
            start, end = path[i], path[i+1]
            edge_tasks.append(self.score_edge(G, start, end, query))

        edge_results = await asyncio.gather(*edge_tasks)

        for i, edge_explanation in enumerate(edge_results):
            start, end = path[i], path[i+1]
            path_details += f"{start} -> {end}:\n{edge_explanation}\n\n"

        try:
            print_flush("Generating path explanation")
            path_explanation_result = await retry_with_exponential_backoff(
                self.path_explanation_chain.ainvoke,
                {
                    "query": query,
                    "path": " -> ".join(path),
                    "path_details": path_details
                }
            )

            path_explanation_result = extract_string_value(path_explanation_result)
            parsed_result = clean_json_output(path_explanation_result)

            path_explanation = parsed_result['explanation']
            path_score = float(parsed_result.get('relevance_score', 0))

            if not 0 <= path_score <= 100:
                print_flush(f"Warning: Relevance score out of range (0-100) for path explanation: {path_score}")
                path_score = max(0, min(100, path_score))

        except Exception as e:
            print_flush(f"Error in generating path explanation: {str(e)}")
            path_explanation = f"Error in generating explanation: {str(e)}"
            path_score = 0

        return path_score, path_explanation, path_details

    @staticmethod
    def parse_edge_explanations(path_details: str) -> Dict[str, str]:
        edge_explanations = {}
        current_edge = None
        current_explanation = []

        for line in path_details.split('\n'):
            if ' -> ' in line and line.endswith(':'):
                if current_edge and current_explanation:
                    edge_explanations[current_edge] = '\n'.join(current_explanation).strip()
                current_edge = line.strip()[:-1]  # Remove the colon
                current_explanation = []
            elif line.strip() and current_edge is not None:
                current_explanation.append(line)

        if current_edge and current_explanation:
            edge_explanations[current_edge] = '\n'.join(current_explanation).strip()

        return edge_explanations

    async def recommend_paths(self, G: nx.Graph, query: str, start_protein: str, max_length: int = 4, top_k: int = 20) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        paths = []
        for node in G.nodes():
            paths.extend(nx.all_simple_paths(G, start_protein, node, cutoff=max_length))
        print_flush(f"Found {len(paths)} paths.")

        scored_paths = []
        for path in paths:
            if len(path) < 2:
                continue
            path_score, path_explanation, path_details = await self.score_path(G, path, query)
            scored_paths.append((path, path_score, path_explanation, path_details))

        scored_paths.sort(key=lambda x: x[1], reverse=True)
        top_paths = scored_paths[:top_k]

        recommended_proteins = []
        recommended_relationships = []
        recommended_path_details = []

        for path, path_score, path_explanation, path_details in top_paths:
            path_proteins = []
            path_relationships = []

            # Parse edge explanations
            edge_explanations = self.parse_edge_explanations(path_details)

            # Process proteins
            for protein in path:
                if protein not in [p['name'] for p in recommended_proteins]:
                    protein_info = {
                        'name': protein,
                        'annotation': G.nodes[protein]['annotation'],
                        'depth': G.nodes[protein]['depth'] if 'depth' in G.nodes[protein] else 0
                    }
                    recommended_proteins.append(protein_info)
                    path_proteins.append(protein_info)

            # Process relationships
            for i in range(len(path) - 1):
                start, end = path[i], path[i+1]
                edge_data = G[start][end]
                edge_key = f"{start} -> {end}"

                explanation = edge_explanations.get(edge_key, "No explanation available")

                relationship_info = {
                    'start': start,
                    'end': end,
                    'type': edge_data['type'],
                    'properties': {
                        'score': edge_data['properties'].get('score', 0),
                        'annotation': edge_data['properties'].get('annotation', 'No annotation available')
                    },
                    'explanation': explanation
                }
                recommended_relationships.append(relationship_info)
                path_relationships.append(relationship_info)

            # Add path details
            recommended_path_details.append({
                'path': path,
                'score': path_score,
                'explanation': path_explanation,
                'details': path_details,
                'proteins': path_proteins,
                'relationships': path_relationships
            })

        return recommended_proteins, recommended_relationships, recommended_path_details