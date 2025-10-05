from typing import List, Dict, Any

def confidence(docs: List[Dict[str,Any]]) -> float:
    if not docs: return 0.0
    avg = sum(float(d.get("score",0)) for d in docs)/len(docs)
    src = min(len(docs)/3, 1.0)
    return avg*0.7 + src*0.3

def cite(d: Dict[str,Any]) -> str:
    m = d.get("metadata",{})
    f = m.get("file","unknown.pdf"); ps=m.get("page_start"); pe=m.get("page_end")
    if ps and pe and ps!=pe: return f"{f}, p. {ps}–{pe}"
    if ps: return f"{f}, p. {ps}"
    return f"{f}"