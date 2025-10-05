# app/agent.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
from .adapters import pdf_rag
from .adapters import llm
from .utils import confidence, cite

MAX_STEPS = 3  # hard cap so we never hit LangGraph's recursion_limit

def _last_search(state: Dict[str, Any]) -> str | None:
    its = state.get("iterations", [])
    for it in reversed(its):
        if it.get("action") == "SEARCH":
            return it.get("search_query")
    return None

def think_and_search(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    iterations: List[dict] = state.get("iterations", [])
    docs_so_far: List[dict] = state.get("documents", [])

    step_idx = len(iterations)
    if step_idx >= MAX_STEPS:
        return {
            "query": query,
            "iterations": iterations + [{"action": "FINISH", "reason": "max_iterations"}],
            "documents": docs_so_far,
        }

    prev = "\n".join(
        f"  Search {i+1}: {it['search_query']} → {it['num_results']} docs"
        for i, it in enumerate(iterations) if it.get("action") == "SEARCH"
    ) or "None yet"

    prompt = f"""You are planning PDF retrieval steps for a user question.

Original question: {query}

Previous searches:
{prev}

Decide ONE next action (uppercase only) and one line only:
- SEARCH: <specific, different query>  ← use this if you still need info
- FINISH                               ← use this if you can answer now

Rules:
- Do not repeat the exact same search twice.
- Prefer narrowing or complementing previous searches.
Respond with ONLY: "SEARCH: ..." or "FINISH" (no extra text).
"""

    resp = (llm.invoke(prompt) or "").strip()
    up = resp.upper()

    if up.startswith("FINISH"):
        return {
            "query": query,
            "iterations": iterations + [{"action": "FINISH", "reason": "sufficient_info"}],
            "documents": docs_so_far,
        }

    if "SEARCH:" in up:
        q = resp.split("SEARCH:", 1)[1].strip()
        # early-stop if planner repeats the same search term
        last_q = _last_search(state)
        if last_q and q.lower() == last_q.lower():
            return {
                "query": query,
                "iterations": iterations + [{"action": "FINISH", "reason": "repeated_search"}],
                "documents": docs_so_far,
            }

        found = pdf_rag.retrieve(q, top_k=3) or []
        # if we got no results twice in a row, finish
        nohit_prev = (len(iterations) >= 1 and iterations[-1].get("action") == "SEARCH" and iterations[-1].get("num_results", 0) == 0)
        if len(found) == 0 and nohit_prev:
            return {
                "query": query,
                "iterations": iterations + [
                    {"action": "SEARCH", "search_query": q, "num_results": 0},
                    {"action": "FINISH", "reason": "no_results"},
                ],
                "documents": docs_so_far,
            }

        return {
            "query": query,
            "iterations": iterations + [{"action": "SEARCH", "search_query": q, "num_results": len(found)}],
            # ✅ accumulate docs across steps
            "documents": docs_so_far + found,
        }

    # Fallback: parsing failed → finish
    return {
        "query": query,
        "iterations": iterations + [{"action": "FINISH", "reason": "parse_error"}],
        "documents": docs_so_far,
    }

def should_continue(state: Dict[str, Any]) -> str:
    last = state["iterations"][-1]
    return "generate" if last["action"] == "FINISH" else "search"

def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    docs = state.get("documents", [])
    if not docs:
        return {
            "query": query,
            "final_answer": "I couldn’t find relevant excerpts.",
            "confidence": 0.0,
            "sources": [],
            "iterations": state.get("iterations", []),
            "documents": docs,
        }

    excerpts = "\n\n".join(f"[{cite(d)}]\n{d['content']}" for d in docs)
    prompt = f"""Answer using ONLY the PDF excerpts.
Question: {query}

Excerpts:
{excerpts}

Be concise (bullets allowed). End with:
Sources: (file, pages) …"""
    answer = llm.invoke(prompt)
    conf = confidence(docs)
    sources = [
        {
            "file": d["metadata"].get("file"),
            "page_start": d["metadata"].get("page_start"),
            "page_end": d["metadata"].get("page_end"),
            "score": float(d.get("score", 0.0)),
        }
        for d in docs
    ]
    # keep full state for debug
    return {
        "query": query,
        "final_answer": answer,
        "confidence": conf,
        "sources": sources,
        "iterations": state.get("iterations", []),
        "documents": docs,
    }

def build_agent():
    g = StateGraph(dict)
    g.add_node("search", think_and_search)
    g.add_node("generate", generate)
    g.set_entry_point("search")
    g.add_conditional_edges("search", should_continue, {"search": "search", "generate": "generate"})
    g.add_edge("generate", END)
    return g.compile()