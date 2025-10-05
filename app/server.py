from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .agent import build_agent
from app.adapters import llm

app = FastAPI(title="PDF Ambient Agent (POC)")
agent = build_agent()

class AgentRequest(BaseModel):
    query: str

@app.post("/agent")
def run_agent(req: AgentRequest, debug: bool = Query(False)):
    try:
        result = agent.invoke({
            "query": req.query,
            "iterations": [],   # <-- seed
            "documents": []     # <-- seed
        })
        payload = {
            "answer": result["final_answer"],
            "confidence": result["confidence"],
            "iterations": len(result["iterations"]),
            "sources": result.get("sources", [])
        }
        if debug:
            payload["trace"] = {
                "iterations_list": result.get("iterations", []),
                "docs": result.get("documents", [])
            }
        return payload
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/llm_ping")
def llm_ping():
    out = llm.invoke("Respond with only: FINISH")
    return {"ok": True, "model": os.getenv("OLLAMA_MODEL", "llama3.1"), "out": out[:100]}