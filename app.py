from flask import Flask, request, jsonify
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import json
import nest_asyncio
from langchain_openai import ChatOpenAI
import hashlib
import networkx as nx
import matplotlib.pyplot as plt
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,AutoConfig
import torch
from huggingface_hub import login
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
import time
import random
import re
from typing import Callable, Any, List, Dict, Tuple
from accelerate import Accelerator

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Neo4j connection
DB_URI = os.getenv("DB_URI")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")
driver = GraphDatabase.driver(DB_URI, auth=(DB_USER, DB_PASSWORD))
# huggingface API
login(token=HUGGINGFACE_KEY,add_to_git_credential=True)
# OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Set up LangChain components
llm = ChatOpenAI(model="gpt-4o")

edge_relevance_prompt = PromptTemplate(
    input_variables=["query", "start_protein", "end_protein", "start_annotation", "end_annotation"],
    template="""As a protein interaction expert, evaluate the relevance of the interaction between {start_protein} and {end_protein}
    to the query: "{query}".

        You should follow those steps for analysis:
    <Steps>
    1. Use the name of proteins along the path to figure out the directional graph. Consider the direction of the interaction and always and only consider the impact of the direct last protein.
    2. The previous protein should be able to directly interact with the current protein according to the information provided. If not, the interaction is not relevant.
    3. Explan potential biological processes or mechanisms before recommending it. If you can not find any explict reasons and just say they are relevant. The path will be deleted.
    </Steps>

    Protein information:
    {start_protein}: {start_annotation}
    {end_protein}: {end_annotation}

    Explain why this interaction is relevant or not relevant to the query using less than 35 words.
    Try to explan the edge informatively using simple and short sentences structure to improve the readability.



    """

)

path_explanation_prompt = PromptTemplate(
    input_variables=["query", "path", "path_details"],
    template="""As a protein-protein interaction expert, choose the following protein interaction paths to satisfy the query: "{query}"
    You should follow those steps for analysis:
    <Steps>
    1. Use the name of proteins along the path to figure out the directional graph. Protein A -> Protein B -> Protein C means that protein c can influence protein b and protein b can influence protein a.
    2. The path should satisfy the requirements in the query. The more likely the requirements get satisfied, the higher the relevance score is.
    3. The previous protein should be able to directly interact with the current protein according to the information provided. If not, the interaction is not relevant.
    4. Explan potential biological processes or mechanisms before recommending it. If you can not find any explict reasons and just say they are relevant. The path will be deleted.
    5. Provide a relevance score from 0 to 100, where 0 is not relevant at all and 100 is highly relevant This score will be used to rank the paths according to their relevance of the explanation to the query.
    </Steps>
    Some examples of paths:
    <Path>### Edge 3: MARK2 -> STK11

- **Explanation**: The interaction is relevant because STK11 phosphorylates MARK4, influencing its activity.</Path>,<Decision>we do not want it since the direction of influence is wrong</Decision>
    <Path>Edge 1: MARK4 -> MARK3
    Explanation: The interaction between MARK4 and MARK3 is relevant.</Path>,<Decision>we do not want it because it does tell the explict reasons</Decision>
    <Path>Edge 2: MARK3 -> MARK2
    Explanation: This interaction is relevant because both MARK3 and MARK2 are serine/threonine kinases involved in microtubule dynamics.</Path>,<Decision>we do not want it because MARK4 and MARK3 do not have direct interaction :Find proteins that inhibit or activate MARK4</Decision>

    Path: {path}

    Details of each step:
    {path_details}


    Discuss any potential biological processes or mechanisms that this path might represent and explain why you recommnend this path, using less than 80 words.

    Your explanation should be informative and accessible, as if you're explaining it to a fellow researcher.
                Your response MUST be in the following JSON format:
    {{
        "explanation": "Your explanation here",
        "relevance_score": "0-100"
    }}

    """


)
# Define edge_relevance_prompt and path_explanation_prompt here
# (Omitted for brevity, use the same prompts as before)

edge_relevance_chain = (
    {"query": RunnablePassthrough(), "start_protein": RunnablePassthrough(), "end_protein": RunnablePassthrough(),
     "start_annotation": RunnablePassthrough(), "end_annotation": RunnablePassthrough()}
    | edge_relevance_prompt
    | llm
    | StrOutputParser()
)

path_explanation_chain = (
    {"query": RunnablePassthrough(), "path": RunnablePassthrough(), "path_details": RunnablePassthrough()}
    | path_explanation_prompt
    | llm
    | StrOutputParser()
)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class GraphCache:
    def __init__(self):
        self.cache = {}
        self.precalculated_nodes = {}

    def get_cache_key(self, start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold):
        key_string = f"{start_protein}_{k}_{depth}_{sorted(nodes_to_remove)}_{sorted(relationship_types_to_remove)}_{n_fold}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold):
        key = self.get_cache_key(start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold)
        return self.cache.get(key)

    def set(self, start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold, proteins, relationships):
        key = self.get_cache_key(start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold)
        self.cache[key] = (proteins, relationships)

    def get_precalculated_nodes(self, protein_name, k, n_fold):
        key = f"{protein_name}_{k}_{n_fold}"
        return self.precalculated_nodes.get(key)

    def set_precalculated_nodes(self, protein_name, k, n_fold, sorted_proteins, sorted_distances):
        key = f"{protein_name}_{k}_{n_fold}"
        self.precalculated_nodes[key] = (sorted_proteins, sorted_distances)

    def clear_cache(self):
        self.cache = {}
        self.precalculated_nodes = {}
        print("Cache and precalculated nodes cleared.")

    def get_cache_size(self):
        return len(self.cache), len(self.precalculated_nodes)

class GraphManager:
    def __init__(self):
        self.graphs = {}
        self.current_graph = None

    def add_graph(self, name, proteins, relationships, start_protein, n_fold):
        self.graphs[name] = {
            'proteins': proteins,
            'relationships': relationships,
            'start_protein': start_protein,
            'n_fold': n_fold
        }
        self.current_graph = name
        print(f"Graph '{name}' added and set as current graph.")

    def get_graph(self, name):
        return self.graphs.get(name)

    def set_current_graph(self, name):
        if name in self.graphs:
            self.current_graph = name
            print(f"Current graph set to '{name}'.")
        else:
            print(f"Graph '{name}' not found.")
            print("Available graphs:")
            for graph_name in self.graphs.keys():
                print(f"  - '{graph_name}'")

    def list_graphs(self):
        return list(self.graphs.keys())

graph_cache = GraphCache()
graph_manager = GraphManager()

# Helper functions (implement these)

# Current function library
def clean_embedding(embedding_list):
    # Remove the '[' and ']' from the first and last elements respectively
    embedding_list[0] = embedding_list[0].replace('[', '')
    embedding_list[-1] = embedding_list[-1].replace(']', '')
    return np.array([float(x) for x in embedding_list])


def fetch_start_node_details(protein_name):
    query = """
MATCH (p:Protein)
WHERE p.name < $name
WITH COUNT(p) AS stepsToSkip
MATCH (specificProtein:Protein {name: $name})
RETURN specificProtein AS protein, stepsToSkip

    """
    with driver.session() as session:
        result = session.run(query, name=protein_name)
        for record in result:
            protein_details = record['protein']
            id = record['stepsToSkip']

            cleaned_embedding = clean_embedding(protein_details['embedding'])
    return protein_details, cleaned_embedding, id

def fetch_interacting_proteins(driver, protein_name):
    query = """
    MATCH (p:Protein {name: $name})
    MATCH (interactor:Protein)-[r:INTERACTS_FULL|INTERACTS_PHY]->(p)
    RETURN interactor.id AS id, interactor.name AS name, interactor.embedding AS embedding,
           interactor.annotation AS annotation, TYPE(r) AS relationshipType, r AS properties
    """
    with driver.session() as session:
        result = session.run(query, name=protein_name)
        return [{
                 "name": record["name"],
                 "embedding": clean_embedding(record["embedding"]),
                 "score": record["properties"].get('score', 'No score available'),
                 "relationshipType": record["relationshipType"]}
                for record in result]



def embedding_search_on_filtered(query_embedding, interacting_proteins, k, n_fold):
    embeddings = [protein["embedding"] for protein in interacting_proteins]

    # Calculate the total number of proteins to retrieve
    total_k = k * n_fold

    # Ensure we don't try to retrieve more proteins than available
    total_k = min(total_k, len(interacting_proteins))

    index = hnswlib.Index(space='l2', dim=DIMENSION)
    index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
    index.add_items(embeddings, list(range(len(embeddings))))

    # Retrieve total_k nearest neighbors
    labels, distances = index.knn_query(query_embedding, k=total_k)

    # Reshape labels and distances to 1D arrays
    labels = labels.flatten()
    distances = distances.flatten()

    # Create a list of (protein, distance) tuples
    protein_distance_pairs = [(interacting_proteins[label], distance) for label, distance in zip(labels, distances)]

    # Sort the pairs by distance
    protein_distance_pairs.sort(key=lambda x: x[1])

    # Separate the sorted proteins and distances
    sorted_proteins = [pair[0] for pair in protein_distance_pairs]
    sorted_distances = [pair[1] for pair in protein_distance_pairs]

    return sorted_proteins, sorted_distances

def get_proteins_for_fold(sorted_proteins, sorted_distances, k, n_fold):
    start_index = (n_fold - 1) * k
    end_index = min(n_fold * k, len(sorted_proteins))
    return sorted_proteins[start_index:end_index], sorted_distances[start_index:end_index]

def fetch_relationships_between_nodes(driver, start_node, recommended_nodes):
    recommended_node_names = [node['name'] for node in recommended_nodes]
    print("Recommended Node Names:", recommended_node_names)
    print("Start Node ID:", start_node['name'])
    query = """
    MATCH (start:Protein {name: $startName})
    MATCH (recommended:Protein) WHERE recommended.name IN $recommendedNodeNames
    OPTIONAL MATCH (start)-[r:INTERACTS_FULL | INTERACTS_PHY]->(recommended)
    RETURN start, recommended, TYPE(r) AS relationshipType, r AS properties
    """
    with driver.session() as session:
        result = session.run(query, startName=start_node['name'], recommendedNodeNames=recommended_node_names)
        relationships = []
        for record in result:
            start_node_name = record['start']['name']
            recommended_node_name = record['recommended']['name']
            recommended_node_annotation = record['recommended'].get('annotation', 'No annotation available')

            if record['properties']:  # If there are properties, it means there is a relationship
                relationships.append({
                    'start': start_node_name,
                    'end': recommended_node_name,
                    'type': record['relationshipType'],
                    'properties': {
                        'score': record['properties'].get('score', 'No score available'),
                        'annotation': recommended_node_annotation

                    }
                })
            else:
                print(f"No relationships found between {start_node_name} and {recommended_node_name}.")
    return relationships





## Data Loading and Graph Construction

def load_protein_data(protein_data: str) -> Dict[str, Dict]:
    proteins = {}
    for line in protein_data.split('\n'):
        if line.strip():
            protein = json.loads(line)
            proteins[protein['name']] = protein
    return proteins

def load_relationship_data(relationship_data: str) -> List[Dict]:
    relationships = []
    for line in relationship_data.split('\n'):
        if line.strip():
            relationship = json.loads(line)
            relationships.append(relationship)
    return relationships

def build_graph(proteins: List[Dict], relationships: List[Dict]) -> nx.Graph:
    G = nx.Graph()
    for protein in proteins:
        G.add_node(protein['name'], **protein)

    for relationship in relationships:
        start, end = relationship['start'], relationship['end']
        G.add_edge(start, end, **relationship)
    return G





## Path Finding and Scoring
# Function to ensure prints are immediately visible
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def extract_string_value(value: Any) -> str:
    if hasattr(value, 'text'):
        return value.text
    elif hasattr(value, 'content'):
        return str(value.content)
    elif isinstance(value, str):
        return value
    else:
        return str(value)


def clean_json_output(raw_output: str) -> dict:
    # First, try to parse the input as is
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    # If that fails, try to clean it as JSON
    cleaned_output = re.sub(r'^```json|```$', '', raw_output).strip()
    try:
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        pass

    # If JSON parsing fails, try to evaluate it as a Python literal
    try:
        return ast.literal_eval(cleaned_output)
    except (ValueError, SyntaxError):
        pass

    # If all parsing attempts fail, raise an error
    raise ValueError(f"Unable to parse output as JSON or dict: {raw_output}")



async def retry_with_exponential_backoff(
    func: Callable,
    *args,
    max_retries: int = 50,
    base_delay: float = 1,
    rate_limit_errors: tuple = ('rate_limit_exceeded', 'too_many_requests')
):
    retries = 0
    while True:
        try:
            return await func(*args)
        except Exception as e:
            error_message = str(e).lower()
            is_rate_limit_error = any(err in error_message for err in rate_limit_errors)

            if not is_rate_limit_error or retries >= max_retries:
                print(f"Error occurred: {e}")
                raise

            # delay = base_delay * (2 ** retries) + random.uniform(0, 1)
            delay = base_delay
            print(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
            retries += 1

def find_paths(G: nx.Graph, start: str, max_length: int = 4) -> List[List[str]]:
    paths = []
    for node in G.nodes():
        paths.extend(nx.all_simple_paths(G, start, node, cutoff=max_length))
    return paths


async def extract_json_from_string(s):
    try:
        # Find the start and end of the JSON object
        start = s.find('{')
        end = s.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = s[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return None

async def score_edge(G: nx.Graph, start: str, end: str, query: str) -> str:
    start_annotation = G.nodes[start]['annotation']
    end_annotation = G.nodes[end]['annotation']

    try:
        print_flush(f"Scoring edge {start} -> {end}")
        result = await retry_with_exponential_backoff(
            edge_relevance_chain.ainvoke,
            {
                "query": query,
                "start_protein": start,
                "end_protein": end,
                "start_annotation": start_annotation,
                "end_annotation": end_annotation
            }
        )

        # Extract string value from result
        result = extract_string_value(result)

        print_flush(f"Debug - LLM output for edge {start} -> {end}:")
        print_flush(f"Content: {result}")

        # No need to parse JSON here, just return the explanation
        explanation = result

    except Exception as e:
        print_flush(f"Error in scoring edge {start} -> {end}: {str(e)}")
        explanation = f"Error in scoring: {str(e)}"

    return explanation

async def score_path(G: nx.Graph, path: List[str], query: str) -> Tuple[float, str, str]:
    path_details = ""

    edge_tasks = []
    for i in range(len(path) - 1):
        start, end = path[i], path[i+1]
        edge_tasks.append(score_edge(G, start, end, query))

    edge_results = await asyncio.gather(*edge_tasks)

    for i, edge_explanation in enumerate(edge_results):
        start, end = path[i], path[i+1]
        path_details += f"{start} -> {end}:\n{edge_explanation}\n\n"

    try:
        print_flush("Generating path explanation")
        path_explanation_result = await retry_with_exponential_backoff(
            path_explanation_chain.ainvoke,
            {
                "query": query,
                "path": " -> ".join(path),
                "path_details": path_details
            }
        )

        # Extract string value from result
        path_explanation_result = extract_string_value(path_explanation_result)

        parsed_result = clean_json_output(path_explanation_result)

        path_explanation = parsed_result['explanation']
        path_score = float(parsed_result.get('relevance_score', 0))

        if not 0 <= path_score <= 100:
            print_flush(f"Warning: Relevance score out of range (0-100) for path explanation: {path_score}")
            path_score = max(0, min(100, path_score))  # Clamp the score between 0 and 100

    except Exception as e:
        print_flush(f"Error in generating path explanation: {str(e)}")
        path_explanation = f"Error in generating explanation: {str(e)}"
        path_score = 0

    return path_score, path_explanation, path_details
# New Path Generation Function Using Raw Annotations

async def score_path_with_annotations(path: list, protein_annotations: list, query: str) -> tuple:
    # Build node annotations string
    node_annotations_str = ""
    for node, annotation in zip(path, protein_annotations):
        node_annotations_str += f"{node}: {annotation}\n"

    try:
        print(f"Generating path explanation using node annotations for path {' -> '.join(path)}")
        path_explanation_result = await retry_with_exponential_backoff(
            path_explanation_chain_with_annotations.ainvoke,
            {
                "query": query,
                "path": " -> ".join(path),
                "node_annotations": node_annotations_str
            }
        )

        # Extract string value from result
        path_explanation_result = extract_string_value(path_explanation_result)

        parsed_result = clean_json_output(path_explanation_result)

        path_explanation = parsed_result['explanation']
        path_score = float(parsed_result.get('relevance_score', 0))

        if not 0 <= path_score <= 100:
            print(f"Warning: Relevance score out of range (0-100) for path explanation: {path_score}")
            path_score = max(0, min(100, path_score))  # Clamp the score between 0 and 100

    except Exception as e:
        print(f"Error in generating path explanation: {str(e)}")
        path_explanation = f"Error in generating explanation: {str(e)}"
        path_score = 0

    return path_score, path_explanation


async def async_recommend_paths(G: nx.Graph, query: str, start_protein: str, max_length: int = 4, top_k: int = 10) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    paths = find_paths(G, start_protein, max_length)
    print_flush(f"Found {len(paths)} paths.")

    scored_paths = []
    for path in paths:
        if len(path) < 2:
            continue
        path_score, path_explanation, path_details = await score_path(G, path, query)
        scored_paths.append((path, path_score, path_explanation, path_details))

    scored_paths.sort(key=lambda x: x[1], reverse=True)
    top_paths = scored_paths[:top_k]

    recommended_proteins = []
    recommended_relationships = []
    recommended_path_details = []

    for path, path_score, path_explanation, path_details in top_paths:
        path_proteins = []
        path_relationships = []

        # Parse path_details to extract edge explanations
        edge_explanations = {}
        print_flush(f"Path details: {path_details}")
        print_flush(f"Type of path_details: {type(path_details)}")
        for detail in path_details.split('\n'):

            # if ' -> ' in detail and ':' in detail:
            #     edge_key, explanation = detail.split(':', 1)
            #     edge_explanations[edge_key.strip()] = explanation.strip()

            #     print_flush('EDGE KEY:', edge_key)
            #     print_flush('EXPLANATION:', edge_explanations)
              if ' -> ' in detail and detail.endswith(':'):
                edge_key = detail.strip()[:-1]  # Remove the colon at the end
                explanation_start = path_details.index(detail) + len(detail)
                explanation_end = path_details.find('\n\n', explanation_start)
                if explanation_end == -1:  # If it's the last explanation
                    explanation_end = len(path_details)
                explanation_text = path_details[explanation_start:explanation_end].strip()
                edge_explanations[edge_key] = (explanation_text)
                print_flush('EDGE KEY:', edge_key)
                print_flush('EXPLANATION:', edge_explanations)

        for protein in path:
            if protein not in [p['name'] for p in recommended_proteins]:
                protein_info = {
                    'name': protein,
                    'annotation': G.nodes[protein]['annotation'],
                    'depth': G.nodes[protein]['depth']
                }
                recommended_proteins.append(protein_info)
                path_proteins.append(protein_info)

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
                    'score': edge_data['properties']['score'],
                    'annotation': edge_data['properties']['annotation']
                },
                'explanation': explanation
            }
            recommended_relationships.append(relationship_info)
            path_relationships.append(relationship_info)

        recommended_path_details.append({
            'path': path,
            'score': path_score,
            'explanation': path_explanation,
            'details': path_details,
            'proteins': path_proteins,
            'relationships': path_relationships
        })

    return recommended_proteins, recommended_relationships, recommended_path_details
# Apply nest_asyncio to allow nested event loops in Jupyter
nest_asyncio.apply()

def run_recommendation(G: nx.Graph, query: str, start_protein: str, max_length: int = 4, top_k: int = 10) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    async def run_async():
        task = asyncio.create_task(async_recommend_paths(G, query, start_protein, max_length, top_k))
        try:
            return await task
        except asyncio.CancelledError:
            print("Task was cancelled due to keyboard interrupt.")
            return None, None, None

    loop = asyncio.get_event_loop()
    main_task = asyncio.ensure_future(run_async())

    try:
        return loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cancelling the task...")
        main_task.cancel()
        loop.run_until_complete(main_task)  # Allow task to be cancelled
        return None, None, None







@app.route('/analyze_graph', methods=['POST'])
def analyze_graph():
    data = request.json
    start_protein = data.get('start_protein')
    k = data.get('k', 10)
    depth = data.get('depth', 2)
    n_fold = data.get('n_fold', 1)
    nodes_to_remove = data.get('nodes_to_remove', None)
    relationship_types_to_remove = data.get('relationship_types_to_remove', None)

    try:
        # Check if the graph is already in cache
        cached_result = graph_cache.get(start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold)
        if cached_result:
            proteins, relationships = cached_result
        else:
            # Implement or import the run_graph_analysis function
            graph_name, proteins, relationships = run_graph_analysis(driver, start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold)
            
            # Cache the result
            graph_cache.set(start_protein, k, depth, nodes_to_remove, relationship_types_to_remove, n_fold, proteins, relationships)
        
        # Add the graph to GraphManager
        graph_name = f"{start_protein}_d{depth}_k{k}_f{n_fold}"
        graph_manager.add_graph(graph_name, proteins, relationships, start_protein, n_fold)
        
        return jsonify({
            'graph_name': graph_name,
            'proteins': proteins,
            'relationships': relationships
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recommend_paths', methods=['POST'])
def recommend_paths():
    data = request.json
    query = data.get('query')
    start_protein = data.get('start_protein')
    max_length = data.get('max_length', 4)
    top_k = data.get('top_k', 10)

    try:
        # Get the graph from GraphManager
        graph_name = f"{start_protein}_d{max_length}_k{top_k}_f1"  # Adjust this if you use different naming conventions
        graph_data = graph_manager.get_graph(graph_name)
        
        if not graph_data:
            return jsonify({'error': 'Graph not found. Please analyze the graph first.'}), 404
        
        proteins = graph_data['proteins']
        relationships = graph_data['relationships']
        
        G = build_graph(proteins, relationships)
        results = run_recommendation(G, query, start_protein, max_length, top_k)
        
        if results[0] is not None:
            recommended_proteins, recommended_relationships, recommended_path_details = results
            return jsonify({
                'recommended_proteins': recommended_proteins,
                'recommended_relationships': recommended_relationships,
                'recommended_path_details': recommended_path_details
            }), 200
        else:
            return jsonify({'error': 'No results obtained due to interruption.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/list_graphs', methods=['GET'])
def list_graphs():
    try:
        graphs = graph_manager.list_graphs()
        return jsonify({'graphs': graphs}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)