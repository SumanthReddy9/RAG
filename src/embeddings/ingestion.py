import chromadb
from embeddings.chunking import TextChunker
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import uuid
from sentence_transformers import SentenceTransformer


class EmbeddingGRPFunction:

    def __init__(self, model):
        self.model = SentenceTransformer(model)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


class Ingestion:
    def __init__(self, chunker: TextChunker):
        self.client = chromadb.PersistentClient("./db")
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en"
        )
        self.collection = self.client.get_or_create_collection(
            "docs", embedding_function=self.embedding_function
        )
        self.chunker = chunker
    
    def ingestData(self, text) -> int:
        chunks = self.chunker.chunk(text)
        if not chunks:
            return 0

        document_id = str(uuid.uuid4())
        num_chunks = 0
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{document_id}-{chunk_index}"
            metadata = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
            }
            self.collection.add(
                documents=[chunk],
                ids=[chunk_id],
                metadatas=[metadata],
            )
            num_chunks += 1
        return num_chunks
        
        
