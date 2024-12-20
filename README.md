# GraPPI

This Flask-based API provides functionality for analyzing protein interactions and recommending protein paths based on user queries. (since constantly running the large-scale database would be extremely expensive (~3$/hour), once you want to reproduce the results or just explore PPI, please feel free to make your appointment in the issue. We will resume the database in the appointed time period.)

## Features

- Graph analysis of protein interactions
- Path recommendations based on user queries
- Graph caching and management for improved performance

## Installation

1. Clone the repository:
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Set up your environment variables in a `.env` file (see .env.example for required variables).

## Usage

1. Start the Flask server:
```
python app.py
```

2. Use the following endpoints:
- `POST /analyze_graph`: Analyze a protein interaction graph
- `POST /recommend_paths`: Get path recommendations based on a query
- `GET /list_graphs`: List all stored graphs

## API Endpoints

1. Analyze Graph
- Endpoint: `/analyze_graph`
- Description: Analyzes protein interactions and creates a graph based on the provided parameters.
- Request Body: 
```
{
    "start_protein": "MAPT",// Required: Starting protein name
    "k": 10,                // Optional: Number of neighbors to explore (default: 10)
    "depth": 2,            // Optional: Depth of exploration (default: 2)
    "n_fold": 1,           // Optional: Fold parameter for analysis (default: 1)
    "nodes_to_remove": ["ABC", "XYZ"],// Optional: List of proteins to exclude (default: null)
    "relationship_types_to_remove": [] // Optional: Types of relationships to exclude (default: null)
}
```
- Response:
```
{
    "graph_name": "MAPT_d2_k10_f1",
    "proteins": [
        {
            "name": "MAPT",
            "annotation": "protein annotation",
            "depth": 0
        },
        // ... more proteins
    ],
    "relationships": [
        {
            "start": "MAPT",
            "end": "ProteinB",
            "type": "INTERACTS_FULL",
            "properties": {
                "score": 0.85,
                "annotation": "interaction annotation"
            }
        },
        // ... more relationships
    ]
}
```

2. Recommend Paths
- Endpoint: `POST /recommend_paths`
- Description: Recommends protein interaction paths based on a user query.
- Request Body: 
```
{
    "query": "Find protein paths that the end protein phosphorylates MAPT", // Required: Query describing the desired interaction
    "start_protein": "MAPT",                                               // Required: Starting protein name
    "max_length": 4,                                                       // Optional: Maximum path length (default: 4)
    "top_k": 10                                                           // Optional: Number of top paths to return (default: 10)
}
```
- Response:
```
{
    "recommended_proteins": [
        {
            "name": "ProteinA",
            "annotation": "protein annotation",
            "depth": 1
        },
        // ... more proteins
    ],
    "recommended_relationships": [
        {
            "start": "MAPT",
            "end": "ProteinA",
            "type": "INTERACTS_FULL",
            "properties": {
                "score": 0.75,
                "annotation": "interaction annotation"
            },
            "explanation": "Explanation of the interaction relevance"
        },
        // ... more relationships
    ],
    "recommended_path_details": [
        {
            "path": ["MAPT", "ProteinA", "ProteinB"],
            "score": 85.5,
            "explanation": "Detailed explanation of why this path is relevant",
            "details": "Step-by-step analysis of the path",
            "proteins": [...],
            "relationships": [...]
        },
        // ... more paths
    ]
}
```

3. List Graphs
- Endpoint: `GET /list_graphs`
- Description: Lists all stored graphs in the system.
- Request Body: 
```
{
    "graphs": [
        "MAPT_d2_k10_f1",
        "ABC_d3_k15_f2",
        // ... more graph names
    ]
}
```
- Error Responses: All endpoints may return error responses in the following format:
```
{
    "error": "Error message describing what went wrong"
}
```
## Usage Example
Here is a complete example using curl:
```
# 1. First, analyze a graph
curl -X POST http://localhost:5000/analyze_graph \
-H "Content-Type: application/json" \
-d '{
    "start_protein": "MAPT",
    "k": 10,
    "depth": 2,
    "n_fold": 1
}'

# 2. Then, get path recommendations
curl -X POST http://localhost:5000/recommend_paths \
-H "Content-Type: application/json" \
-d '{
    "query": "Find protein paths that the end protein phosphorylates MAPT",
    "start_protein": "MAPT",
    "max_length": 4,
    "top_k": 10
}'

# 3. List all available graphs
curl http://localhost:5000/list_graphs
```

If app.py fails, you can use Pipeline.ipynb as a backup. You can walk through the Jupyter Notebook to obtain your PPI recommendation results.
## Credentials Setup
This application requires two types of credentials:
### Pre-configured Database Access
The Neo4j database credentials are pre-configured and managed by the application. You don't need to set these up. However, since constantly running the large-scale database would be extremely expensive (~3$/hour), once you want to reproduce the results or just explore PPI, please feel free to make your appointment in the issue. We will resume the database in the appointed time period.
### Required API Keys
You need to provide your own API keys for:
1. HuggingFace API: Get your API key from HuggingFace (Can be omitted if using ChatGPT series models)
2. OpenAI API: Get your API key from OpenAI Platform (Can be omitted if using other open-source models)

## Setting Up Your Credentials
1. Copy the example environment file:
```
cp .env.example .env
```
2. Update the `.env` file with your API keys:

```
HUGGINGFACE_KEY=your_huggingface_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Important note
1. Before running path recommendations, you must first analyze the graph using the `/analyze_graph` endpoint.
2. The API uses caching to improve performance for repeated queries.
3. Queries should be specific and clearly state the desired protein interactions.
4. The `n_fold` parameter in `analyze_graph` affects the diversity of the returned results.
5. Never commit your `.env` file to version control.
6. Keep your API keys secure and never share them
7. The Neo4j database credentials are managed by the application and should not be modified

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.