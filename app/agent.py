from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
from .adapters import pdf_rag
from .adapters import llm
from .utils import confidence, cite

MAX_STEPS = 3

def think_and_search(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")  # <-- avoid KeyError
    iters = len(state.get("iterations", []))
    if iters >= MAX_STEPS:
        return {
            "query": query,  # <-- keep it
            "iterations": [{"action": "FINISH", "reason": "max_iterations"}]
        }

    prev = "\n".join([
        f"  Search {i+1}: {it['search_query']} → {it['num_results']} docs"
        for i, it in enumerate(state.get("iterations", []))
        if it.get("action") == "SEARCH"
    ]) or "None yet"

    prompt = f"""You are planning PDF retrieval steps.
Original question: {query}
Previous searches:
{prev}
Respond with only one of:
- SEARCH: <specific query>
- FINISH"""
    resp = llm.invoke(prompt).strip()

    if resp.upper().startswith("FINISH"):
        return {
            "query": query,  # <-- keep it
            "iterations": [{"action": "FINISH", "reason": "sufficient_info"}]
        }

    if "SEARCH:" in resp.upper():
        q = resp.split("SEARCH:", 1)[1].strip()
        docs = pdf_rag.retrieve(q, top_k=3)
        return {
            "query": query,  # <-- keep it
            "iterations": [{"action": "SEARCH", "search_query": q, "num_results": len(docs)}],
            "documents": docs
        }

    return {
        "query": query,  # <-- keep it
        "iterations": [{"action": "FINISH", "reason": "parse_error"}]
    }


def should_continue(state: Dict[str, Any]) -> str:
    last = state["iterations"][-1]
    return "generate" if last["action"] == "FINISH" else "search"


def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")  # not strictly needed, but consistent
    docs = state.get("documents", [])
    if not docs:
        return {"query": query, "final_answer": "I couldn’t find relevant excerpts.", "confidence": 0.0, "sources": []}

    excerpts = "\n\n".join([f"[{cite(d)}]\n{d['content']}" for d in docs])
    prompt = f"""Answer using ONLY the PDF excerpts.
Question: {query}

Excerpts:
{excerpts}

Be concise (bullets allowed). End with:
Sources: (file, pages) …"""
    answer = llm.invoke(prompt)
    conf = confidence(docs)
    sources = [{"file": d["metadata"]["file"],
                "page_start": d["metadata"]["page_start"],
                "page_end": d["metadata"]["page_end"],
                "score": float(d["score"])} for d in docs]
    return {"query": query, "final_answer": answer, "confidence": conf, "sources": sources}

def build_agent():
    g = StateGraph(dict)
    g.add_node("search", think_and_search)
    g.add_node("generate", generate)
    g.set_entry_point("search")
    g.add_conditional_edges("search", should_continue, {"search":"search","generate":"generate"})
    g.add_edge("generate", END)
    return g.compile()