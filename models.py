# models.py
from neo4j import GraphDatabase
import numpy as np
from typing import List, Dict, Tuple, Optional
import networkx as nx

class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    @staticmethod
    def clean_embedding(embedding_list):
        embedding_list[0] = embedding_list[0].replace('[', '')
        embedding_list[-1] = embedding_list[-1].replace(']', '')
        return np.array([float(x) for x in embedding_list])

    def fetch_start_node_details(self, protein_name: str) -> Tuple[Dict, np.ndarray, int]:
        query = """
        MATCH (p:Protein)
        WHERE p.name < $name
        WITH COUNT(p) AS stepsToSkip
        MATCH (specificProtein:Protein {name: $name})
        RETURN specificProtein AS protein, stepsToSkip
        """
        with self.driver.session() as session:
            result = session.run(query, name=protein_name)
            for record in result:
                protein_details = record['protein']
                id = record['stepsToSkip']
                cleaned_embedding = self.clean_embedding(protein_details['embedding'])
                return protein_details, cleaned_embedding, id

    def fetch_interacting_proteins(self, protein_name: str) -> List[Dict]:
        query = """
        MATCH (p:Protein {name: $name})
        MATCH (interactor:Protein)-[r:INTERACTS_FULL|INTERACTS_PHY]->(p)
        RETURN interactor.id AS id, interactor.name AS name, interactor.embedding AS embedding,
               interactor.annotation AS annotation, TYPE(r) AS relationshipType, r AS properties
        """
        with self.driver.session() as session:
            result = session.run(query, name=protein_name)
            return [{
                "name": record["name"],
                "embedding": self.clean_embedding(record["embedding"]),
                "score": record["properties"].get('score', 'No score available'),
                "relationshipType": record["relationshipType"]
            } for record in result]

    def fetch_protein_details(self, protein_names: List[str]) -> Dict[str, Dict]:
        print(f"Fetching details for proteins: {protein_names}")
        query = """
        MATCH (p:Protein)
        WHERE p.name IN $proteinNames
        RETURN p.name AS name, p.annotation AS annotation
        """
        with self.driver.session() as session:
            result = session.run(query, proteinNames=protein_names)
            protein_details = {}
            for record in result:
                protein_details[record['name']] = {
                    'name': record['name'],
                    'annotation': record.get('annotation', 'No annotation available')
                }

            # Handle proteins not found
            for name in protein_names:
                if name not in protein_details:
                    print(f"No details found for protein: {name}")
                    protein_details[name] = {
                        'name': name,
                        'annotation': 'No annotation available'
                    }

            return protein_details

    def fetch_relationships_between_nodes(self, start_node: Dict, recommended_nodes: List[Dict]) -> List[Dict]:
        recommended_node_names = [node['name'] for node in recommended_nodes]
        query = """
        MATCH (start:Protein {name: $startName})
        MATCH (recommended:Protein) WHERE recommended.name IN $recommendedNodeNames
        OPTIONAL MATCH (start)-[r:INTERACTS_FULL | INTERACTS_PHY]->(recommended)
        RETURN start, recommended, TYPE(r) AS relationshipType, r AS properties
        """
        with self.driver.session() as session:
            result = session.run(query, startName=start_node['name'], recommendedNodeNames=recommended_node_names)
            relationships = []
            for record in result:
                start_node_name = record['start']['name']
                recommended_node_name = record['recommended']['name']
                recommended_node_annotation = record['recommended'].get('annotation', 'No annotation available')

                if record['properties']:
                    relationships.append({
                        'start': start_node_name,
                        'end': recommended_node_name,
                        'type': record['relationshipType'],
                        'properties': {
                            'score': record['properties'].get('score', 'No score available'),
                            'annotation': recommended_node_annotation
                        }
                    })

            return relationships