# graph_utils.py
import hnswlib
from typing import List, Dict, Tuple, Optional
import networkx as nx
import numpy as np
import hashlib  # Missing import for GraphCache
from config import DIMENSION  # Be explicit about what we're importing

def build_graph(proteins: List[Dict], relationships: List[Dict]) -> nx.Graph:
    """
    Build a NetworkX graph from protein and relationship data.
    
    Args:
        proteins: List of protein dictionaries containing protein information
        relationships: List of relationship dictionaries containing interaction information
        
    Returns:
        NetworkX graph representing the protein interaction network
    """
    G = nx.Graph()
    for protein in proteins:
        # Make a copy of protein dict to avoid modifying original
        protein_data = protein.copy()
        name = protein_data.pop('name')  # Remove name from attributes
        G.add_node(name, **protein_data)

    for relationship in relationships:
        start, end = relationship['start'], relationship['end']
        # Make a copy of relationship dict to avoid modifying original
        rel_data = relationship.copy()
        rel_data.pop('start')
        rel_data.pop('end')
        G.add_edge(start, end, **rel_data)
    return G

def embedding_search_on_filtered(query_embedding: np.ndarray, 
                               interacting_proteins: List[Dict], 
                               k: int, 
                               n_fold: int) -> Tuple[List[Dict], List[float]]:
    """
    Perform embedding search on filtered proteins.
    
    Args:
        query_embedding: Numpy array of the query protein embedding
        interacting_proteins: List of proteins to search through
        k: Number of proteins to retrieve per fold
        n_fold: Number of folds
        
    Returns:
        Tuple of (sorted proteins, sorted distances)
    """
    if not interacting_proteins:
        return [], []
        
    embeddings = [protein["embedding"] for protein in interacting_proteins]
    total_k = min(k * n_fold, len(interacting_proteins))

    # Initialize HNSW index
    try:
        index = hnswlib.Index(space='l2', dim=DIMENSION)
        index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        index.add_items(embeddings, list(range(len(embeddings))))
    except Exception as e:
        print(f"Error initializing HNSW index: {str(e)}")
        return [], []

    try:
        labels, distances = index.knn_query(query_embedding, k=total_k)
        labels = labels.flatten()
        distances = distances.flatten()
    except Exception as e:
        print(f"Error performing kNN search: {str(e)}")
        return [], []

    protein_distance_pairs = [(interacting_proteins[label], distance) 
                            for label, distance in zip(labels, distances)]
    protein_distance_pairs.sort(key=lambda x: x[1])

    sorted_proteins = [pair[0] for pair in protein_distance_pairs]
    sorted_distances = [pair[1] for pair in protein_distance_pairs]

    return sorted_proteins, sorted_distances

def get_proteins_for_fold(sorted_proteins: List[Dict], 
                         sorted_distances: List[float], 
                         k: int, 
                         n_fold: int) -> Tuple[List[Dict], List[float]]:
    """
    Get proteins for a specific fold from sorted results.
    
    Args:
        sorted_proteins: List of sorted protein dictionaries
        sorted_distances: List of sorted distances
        k: Number of proteins per fold
        n_fold: Which fold to retrieve
        
    Returns:
        Tuple of (proteins for fold, distances for fold)
    """
    if not sorted_proteins or not sorted_distances:
        return [], []
        
    start_index = (n_fold - 1) * k
    end_index = min(n_fold * k, len(sorted_proteins))
    
    if start_index >= len(sorted_proteins):
        return [], []
        
    return sorted_proteins[start_index:end_index], sorted_distances[start_index:end_index]

class GraphCache:
    """Cache for storing and retrieving graph computation results."""
    
    def __init__(self):
        self.cache = {}
        self.precalculated_nodes = {}

    def get_cache_key(self, 
                     start_protein: str, 
                     k: int, 
                     depth: int, 
                     nodes_to_remove: Optional[List[str]], 
                     relationship_types_to_remove: Optional[List[str]], 
                     n_fold: int) -> str:
        """Generate a unique cache key."""
        nodes_to_remove = nodes_to_remove or []
        relationship_types_to_remove = relationship_types_to_remove or []
        key_string = f"{start_protein}_{k}_{depth}_{sorted(nodes_to_remove)}_{sorted(relationship_types_to_remove)}_{n_fold}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, 
           start_protein: str, 
           k: int, 
           depth: int, 
           nodes_to_remove: Optional[List[str]], 
           relationship_types_to_remove: Optional[List[str]], 
           n_fold: int) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """Retrieve cached result if it exists."""
        key = self.get_cache_key(start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold)
        return self.cache.get(key)

    def set(self, 
           start_protein: str, 
           k: int, 
           depth: int, 
           nodes_to_remove: Optional[List[str]], 
           relationship_types_to_remove: Optional[List[str]], 
           n_fold: int, 
           proteins: List[Dict], 
           relationships: List[Dict]):
        """Cache the results."""
        key = self.get_cache_key(start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold)
        self.cache[key] = (proteins, relationships)

    def get_precalculated_nodes(self, 
                              protein_name: str, 
                              k: int, 
                              n_fold: int) -> Optional[Tuple[List[Dict], List[float]]]:
        """Retrieve precalculated nodes if they exist."""
        key = f"{protein_name}_{k}_{n_fold}"
        return self.precalculated_nodes.get(key)

    def set_precalculated_nodes(self, 
                              protein_name: str, 
                              k: int, 
                              n_fold: int, 
                              sorted_proteins: List[Dict], 
                              sorted_distances: List[float]):
        """Cache precalculated nodes."""
        key = f"{protein_name}_{k}_{n_fold}"
        self.precalculated_nodes[key] = (sorted_proteins, sorted_distances)

    def clear_cache(self):
        """Clear all cached data."""
        self.cache = {}
        self.precalculated_nodes = {}
        print("Cache and precalculated nodes cleared.")

    def get_cache_size(self) -> Tuple[int, int]:
        """Get the current size of both caches."""
        return len(self.cache), len(self.precalculated_nodes)