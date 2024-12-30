# analysis.py
from typing import List, Dict, Tuple, Optional
import graph_utils
import models
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
import networkx as nx
import numpy as np
from .recommendation import ProteinRecommender

# Don't initialize these globally - they should be passed to ProteinAnalyzer
# GraphCache = graph_utils.GraphCache()  # Remove this
# Neo4jConnection = models.Neo4jConnection()  # Remove this

class ProteinAnalyzer:
    def __init__(self, neo4j_conn: 'models.Neo4jConnection', 
                 llm: ChatOpenAI, 
                 graph_cache: Optional['graph_utils.GraphCache'] = None):
        self.neo4j_conn = neo4j_conn
        self.llm = llm
        self.graph_cache = graph_cache or graph_utils.GraphCache()
        self.recommender = ProteinRecommender(llm)

    def get_recommended_proteins(self, protein_name: str, k: int = 10, n_fold: int = 1) -> List[Dict]:
        precalculated = self.graph_cache.get_precalculated_nodes(protein_name, k, n_fold)

        if precalculated:
            sorted_proteins, sorted_distances = precalculated
        else:
            interacting_proteins = self.neo4j_conn.fetch_interacting_proteins(protein_name)
            if not interacting_proteins:
                print(f"No interacting proteins found for {protein_name}")
                return []

            start_node_details, query_embedding, _ = self.neo4j_conn.fetch_start_node_details(protein_name)
            sorted_proteins, sorted_distances = graph_utils.embedding_search_on_filtered(query_embedding, interacting_proteins, k, n_fold)
            self.graph_cache.set_precalculated_nodes(protein_name, k, n_fold, sorted_proteins, sorted_distances)

        recommended_proteins, _ = graph_utils.get_proteins_for_fold(sorted_proteins, sorted_distances, k, n_fold)
        return recommended_proteins

    def expand_network(self, start_protein: str, k: int = 10, depth: int = 2, n_fold: int = 1) -> Tuple[List[Dict], List[Dict]]:
        # Fetch details for the start protein
        start_protein_details = self.neo4j_conn.fetch_protein_details([start_protein])
        all_proteins = {start_protein: {**start_protein_details[start_protein], 'depth': 0}}
        all_relationships = []
        proteins_to_process = [(start_protein, 0)]

        while proteins_to_process:
            current_protein, current_depth = proteins_to_process.pop(0)

            if current_depth >= depth:
                continue

            # Determine the number of proteins to recommend based on the current depth
            num_recommendations = k if current_depth == 0 else 4
            current_n_fold = n_fold if current_depth == 0 else 1

            recommended_proteins = self.get_recommended_proteins(current_protein, num_recommendations, current_n_fold)

            if not recommended_proteins:
                print(f"No recommended proteins found for {current_protein} at depth {current_depth}")
                continue

            start_node = {'name': current_protein}
            relationships = self.neo4j_conn.fetch_relationships_between_nodes(start_node, recommended_proteins)
            all_relationships.extend(relationships)

            # Fetch details for recommended proteins
            recommended_protein_names = [protein['name'] for protein in recommended_proteins]
            protein_details = self.neo4j_conn.fetch_protein_details(recommended_protein_names)

            for protein_name, details in protein_details.items():
                if protein_name not in all_proteins or all_proteins[protein_name]['depth'] > current_depth + 1:
                    all_proteins[protein_name] = {
                        **details,
                        'depth': current_depth + 1
                    }
                    proteins_to_process.append((protein_name, current_depth + 1))

        return list(all_proteins.values()), all_relationships

    def get_network(self, start_protein: str, k: int = 10, depth: int = 2, 
                   nodes_to_remove: Optional[List[str]] = None, 
                   relationship_types_to_remove: Optional[List[str]] = None, 
                   n_fold: int = 1) -> Tuple[List[Dict], List[Dict]]:
        nodes_to_remove = nodes_to_remove or []
        relationship_types_to_remove = relationship_types_to_remove or []

        cached_result = self.graph_cache.get(start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold)
        if cached_result:
            return cached_result

        # Expand network
        proteins, relationships = self.expand_network(start_protein, k, depth, n_fold)

        # Cache the result
        self.graph_cache.set(start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold, proteins, relationships)

        return proteins, relationships

    def prune_network(self, proteins: List[Dict], relationships: List[Dict], 
                     nodes_to_remove: Optional[List[str]] = None, 
                     relationship_types_to_remove: Optional[List[str]] = None) -> Tuple[List[Dict], List[Dict]]:
        if nodes_to_remove:
            proteins = [p for p in proteins if p['name'] not in nodes_to_remove]
            relationships = [r for r in relationships if r['start'] not in nodes_to_remove and r['end'] not in nodes_to_remove]

        if relationship_types_to_remove:
            relationships = [r for r in relationships if r['type'] not in relationship_types_to_remove]

        return proteins, relationships
    
    async def run_recommendation(self, G: nx.Graph, query: str, start_protein: str, max_length: int = 4, top_k: int = 20):
        return await self.recommender.recommend_paths(G, query, start_protein, max_length, top_k)