from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from embeddings.ingestion import Ingestion
from embeddings.chunking import SimpleTextChunker
from query.queryllm import OllamaCaller

app = FastAPI()


class QueryRequest(BaseModel):
    q: str

class AddKnowledge(BaseModel):
    k: str

@app.get("/")
def health_check():
    return {"Status: All good"}

@app.post("/query")
def query(req: QueryRequest):

    llm_caller = OllamaCaller()
    
    return llm_caller.call_llm(req.q)

@app.post("/add")
def add_knowledge(text: AddKnowledge):
    try:
        
        text_chunker = SimpleTextChunker(chunk_size=100, overlap=10)
        ingestion = Ingestion(text_chunker)
        n = ingestion.ingestData(text.k)

        return {
            "status": "success",
            "message": "content added successfully",
            "num_chunks": n
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    except:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 8000)