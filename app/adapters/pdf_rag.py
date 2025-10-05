import os, re
from typing import List, Dict, Any
from pypdf import PdfReader

PDF_DIR = os.getenv("PDF_DIR", os.path.join(os.getcwd(), "pdfs"))

_DOCS: List[Dict[str, Any]] = []  # [{id, content, score?, metadata:{file,page_start,page_end}}]

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _score(query: str, text: str) -> float:
    """Very simple keyword score: count of query tokens present / tokens"""
    q = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if not q: return 0.0
    tl = text.lower()
    hits = sum(1 for w in q if w in tl)
    # small boost for multi-word matches
    if " api " in f" {tl} " and " limit" in tl:
        hits += 1
    return min(hits / max(1, len(q)) + (0.05 * len(text) / 2000), 0.95)

def _load_pdfs_once():
    if _DOCS:  # already loaded
        return
    if not os.path.isdir(PDF_DIR):
        print(f"[pdf_rag] PDF_DIR not found: {PDF_DIR} — running with empty index.")
        return
    for name in os.listdir(PDF_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(PDF_DIR, name)
        try:
            reader = PdfReader(path)
        except Exception as e:
            print(f"[pdf_rag] failed to read {name}: {e}")
            continue
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = _normalize_ws(page.extract_text())
            except Exception:
                text = ""
            if not text:
                continue
            _DOCS.append({
                "id": f"{name}-p{i}",
                "content": text[:4000],  # cap page text
                "metadata": {"file": name, "page_start": i, "page_end": i},
            })
    print(f"[pdf_rag] loaded pages: {len(_DOCS)} from {PDF_DIR}")

def retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    In-memory search across PDF pages.
    Returns [{id, content, score, metadata:{file,page_start,page_end}}]
    """
    _load_pdfs_once()
    if not _DOCS:
        # graceful fallback: return empty; your agent will low-confidence finish
        return []
    scored = []
    for d in _DOCS:
        s = _score(query, d["content"])
        if s > 0:
            scored.append({**d, "score": s})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]