from abc import ABC, abstractmethod
import chromadb
import ollama

class LLMCaller(ABC):
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./db")
        self.collection = self.client.get_or_create_collection("docs")

    @abstractmethod
    def call_llm(self, query):
        pass

    def get_context(self, query):
        results = self.collection.query(query_texts = [query], n_results = 5)
        chunks = results["documents"][0] 
        context = ' '.join(chunks)
        return context

class OllamaCaller(LLMCaller):
    
    def call_llm(self, query):
        context = self.get_context(query)
        answer = ollama.generate(
            model = "tinyllama",
            prompt = f"Context: \n{context}\n\nQuestion: {query}\n\nAnswer clearly and concisely:"
        )

        return {"answer": answer["response"]}
