import httpx

class LocalLLMClient:
    def __init__(self, uri):
        self.uri = uri
    
    async def complete(self, query):
        try:
            async with httpx.AsyncClient(timeout = 60) as client:
                response = await client.post(
                    self.uri,
                    json = {"q": query}
                )
                response.raise_for_status()
                return response.json()["answer"]
        except Exception as e:
            raise RuntimeError(f"LLM failed with response: {e}")