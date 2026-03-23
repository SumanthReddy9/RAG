import chromadb
from embeddings.chunking import TextChunker
import uuid


class Ingestion:
    def __init__(self, chunker: TextChunker):
        self.client = chromadb.PersistentClient("./db")
        self.collection = self.client.get_or_create_collection("docs")
        self.chunker = chunker
    
    def ingestData(self, text) -> int:
        chunks = self.chunker.chunk(text)
        print(text)
        num_chunks = 0
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            self.collection.add(documents=[chunk], ids=[doc_id])
            num_chunks += 1
        return num_chunks
        
        
