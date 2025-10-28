from __future__ import annotations
from typing import List, Dict

SYS = (
    "You are a helpful assistant. Use ONLY the provided context to answer the question. "
    "If the answer is not contained in the context, say you don't know."
)

def build_prompt(query: str, contexts: List[str]) -> str:
    ctx_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    return (
        f"{SYS}\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
