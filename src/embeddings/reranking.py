from sentence_transformers import CrossEncoder
from abc import ABC, abstractmethod
import httpx

class CustomCrossEncoder(ABC):
    @abstractmethod
    def rerank(self, query, chunks):
        pass

class LLMClientReRanker(CustomCrossEncoder):
    def __init__(self, uri):
        self.uri = uri

    async def rerank(self, query, chunks):
        scores = []
        async with httpx.AsyncClinet(timeout = 60) as client:
            for chunk in chunks:
                response = await client.post(
                            self.uri,
                            json = {"q": f"What is the relevancy score(0-1) \nDoc: {chunk}\nQuery: {query}"}
                            )
                score = float(response.json()["answer"])
                scores.append((chunk, score))            
            ranked = sorted(scores, lambda x:x[1], reverse=True)

            return [c for c, _ in ranked]
    
class LocalReRanker(CustomCrossEncoder):
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, chunks):
        pairs = [(query, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(scores, lambda x:x[1], reverse=True)
        return [c for c, _ in ranked]
