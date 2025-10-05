# app/adapters/llm.py
import os
import requests
from typing import Dict, Any

# Env-configurable; sensible defaults for local dev
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")

# Default generation options (good for a planner/agent)
DEFAULT_OPTIONS: Dict[str, Any] = {
    "temperature": 0,     # deterministic
    "num_ctx": 4096,      # roomy context for small prompts
    # you can add: "top_p": 1, "top_k": 40, "repeat_penalty": 1.1, etc.
}

class OllamaError(RuntimeError):
    pass

def invoke(prompt: str, *, timeout: int = 120, options: Dict[str, Any] | None = None) -> str:
    """
    Call Ollama's /api/generate and return the 'response' string.
    Raises OllamaError with a clear message if anything goes wrong.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,               # we want a single, complete response
        "options": {**DEFAULT_OPTIONS, **(options or {})},
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise OllamaError(f"Failed to reach Ollama at {OLLAMA_URL}: {e}") from e

    if resp.status_code >= 400:
        # Ollama often returns JSON with 'error'; keep body for debugging
        raise OllamaError(f"Ollama HTTP {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except ValueError as e:
        raise OllamaError(f"Ollama returned non-JSON body: {resp.text[:200]}") from e

    text = (data.get("response") or "").strip()
    if not text:
        raise OllamaError("Ollama returned an empty 'response'. Check model name and server logs.")

    return text