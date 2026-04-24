import os
import json
import logging
from typing import List, Dict, Any, Tuple
import asyncio

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from dotenv import load_dotenv
from embeddings.ingestion import EmbeddingGRPFunction
from sentence_transformers import CrossEncoder

from query.llmclient import LocalLLMClient
from embeddings.reranking import LocalReRanker




class GraphitiClient:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_pass = neo4j_pass

    def initialise(self):
            embedder = EmbeddingGRPFunction("BAAI/bge-small-en")

            client = LocalLLMClient("http://localhost:8000/query")

            cross_encoder = LocalReRanker("BAAI/bge-small-en")

            self.graphiti = Graphiti(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_uri, 
            llm_client = client,
            embedder = embedder,
            cross_encoder = cross_encoder
            )

