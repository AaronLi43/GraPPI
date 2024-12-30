# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
import json
import networkx as nx
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
import nest_asyncio
from typing import Optional, List
from config import (
    DB_URI,
    DB_USER,
    DB_PASSWORD,
    DEFAULT_K,
    DEFAULT_DEPTH
)
from models import Neo4jConnection
from analysis import ProteinAnalyzer
from graph_utils import GraphCache, build_graph

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

app = FastAPI()

class AnalysisRequest(BaseModel):
    start_protein: str
    num_graphs: int
    query: str

class GraphManager:
    def __init__(self):
        self.neo4j_conn = Neo4jConnection(DB_URI, DB_USER, DB_PASSWORD)
        self.llm = ChatOpenAI(model="gpt-4o")
        self.graph_cache = GraphCache()
        self.analyzer = ProteinAnalyzer(self.neo4j_conn, self.llm, self.graph_cache)

    def close(self):
        self.neo4j_conn.close()

    async def process_single_graph(self, start_protein: str, k: int, depth: int, n_fold: int, query: str):
        try:
            # Get network data
            protein_data, relationship_data = self.analyzer.get_network(
                start_protein,
                k=k,
                depth=depth,
                n_fold=n_fold
            )

            # Build graph
            G = build_graph(protein_data, relationship_data)

            # Run recommendation
            proteins, relationships, path_details = await self.analyzer.run_recommendation(
                G,
                query,
                start_protein
            )

            # Prepare results
            results = {
                'Recommended Proteins': proteins,
                'Relationships': relationships,
                'Path Details': path_details
            }

            # Save to file
            graph_name = f"{start_protein}_d{depth}_k{k}_f{n_fold}"
            filename = f'results_{graph_name}.json'
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)

            return {
                'fold': n_fold,
                'filename': filename,
                'graph_name': graph_name
            }

        except Exception as e:
            print(f"Error processing graph {n_fold}: {str(e)}")
            raise e

@app.post("/analyze")
async def analyze_protein_networks(request: AnalysisRequest):
    graph_manager = None
    try:
        graph_manager = GraphManager()
        tasks = []
        
        # Create tasks for each graph
        for n_fold in range(1, request.num_graphs + 1):
            task = graph_manager.process_single_graph(
                request.start_protein,
                DEFAULT_K,
                DEFAULT_DEPTH,
                n_fold,
                request.query
            )
            tasks.append(task)
        
        # Run all tasks concurrently and collect results
        results = await asyncio.gather(*tasks)
        
        return {
            "message": "Analysis complete",
            "num_graphs_processed": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if graph_manager:
            graph_manager.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)