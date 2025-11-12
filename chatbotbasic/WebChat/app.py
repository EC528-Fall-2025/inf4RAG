"""
FastAPI backend for inf4RAG agentic chatbot.
- /health  : quick health check
- /chat    : main endpoint to run RAG + LLM (vLLM or Qwen-Agent)
Run:
  conda activate inf4rag_win
  python app.py
"""

from __future__ import annotations
import os
import uvicorn
import yaml
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_module import RAGStore
from agentic_workflow import VLLMChatClient, QwenAgentClient, build_messages


# -------------------------
# Load config.yaml
# -------------------------
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

LOG_LEVEL = CFG.get("server", {}).get("log_level", "info").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("inf4rag")


# -------------------------
# Init RAG
# -------------------------
rag_cfg = CFG.get("rag", {})
RAG_ENABLED_BY_DEFAULT: bool = bool(rag_cfg.get("enabled_by_default", True))

rag_store = RAGStore(
    docs_dir=rag_cfg.get("docs_dir", "./data/docs"),
    index_dir=rag_cfg.get("index_dir", "./data/index/faiss"),
    embedding_model=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
    chunk_size=int(rag_cfg.get("chunk_size", 800)),
    chunk_overlap=int(rag_cfg.get("chunk_overlap", 150)),
)
# Build or load index at startup (safe even if empty)
rag_store.build_or_load()


# -------------------------
# Init LLM clients
# -------------------------
vllm_cfg = CFG.get("vllm", {})
vllm_client = VLLMChatClient(
    base_url=vllm_cfg.get("base_url", "http://127.0.0.1:8000/v1"),
    api_key=vllm_cfg.get("api_key", "ec528"),
    model=vllm_cfg.get("model", "qwen-4b-instruct"),
)

qwen_cfg = CFG.get("qwen_agent", {})
QWEN_ENABLED = bool(qwen_cfg.get("enabled", False))
qwen_client = None
if QWEN_ENABLED:
    try:
        qwen_client = QwenAgentClient(
            model=qwen_cfg.get("model", "qwen-max-latest"),
            dashscope_api_key=qwen_cfg.get("dashscope_api_key", ""),
        )
        log.info("Qwen-Agent client initialized.")
    except Exception as e:
        log.warning("Qwen-Agent disabled: %s", e)
        qwen_client = None
        QWEN_ENABLED = False


# -------------------------
# FastAPI app & models
# -------------------------
app = FastAPI(title="inf4RAG Agentic Backend", version="0.1.0")


class ChatRequest(BaseModel):
    user_input: str = Field(..., description="User message.")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="OpenAI-style messages.")
    use_rag: Optional[bool] = Field(default=None, description="Override RAG switch; None=use default.")
    provider: str = Field(default="vllm", description="vllm | qwen")
    temperature: float = 0.2
    max_tokens: int = 1024
    top_k: int = 4


class ChatResponse(BaseModel):
    output: str
    retrieved: Optional[List[str]] = None
    provider: str = "vllm"
    model: str = ""
    used_rag: bool = False


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    # Decide whether to use RAG
    use_rag = RAG_ENABLED_BY_DEFAULT if req.use_rag is None else bool(req.use_rag)

    # Retrieve context if enabled
    chunks: List[str] = []
    if use_rag and req.user_input.strip():
        chunks = rag_store.retrieve(req.user_input, k=max(1, req.top_k))

    # Compose messages
    system_prompt = (
        "You are a helpful AI assistant. Use the provided context if relevant. "
        "Cite the context succinctly when applicable."
    )
    messages = build_messages(
        system_prompt=system_prompt,
        user_message=req.user_input,
        history=req.history,
        context_chunks=chunks if use_rag else None,
    )

    # Route to provider
    provider = (req.provider or "vllm").lower()

    if provider == "qwen" and QWEN_ENABLED and qwen_client is not None:
        try:
            text = qwen_client.chat(messages)
            return ChatResponse(
                output=text,
                retrieved=chunks if use_rag else None,
                provider="qwen",
                model=qwen_cfg.get("model", ""),
                used_rag=use_rag,
            )
        except Exception as e:
            # Fallback to vLLM if qwen fails
            log.warning("Qwen-Agent failed (%s); falling back to vLLM.", e)

    # Default: vLLM via OpenAI-compatible API
    text = vllm_client.chat(
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return ChatResponse(
        output=text,
        retrieved=chunks if use_rag else None,
        provider="vllm",
        model=vllm_cfg.get("model", ""),
        used_rag=use_rag,
    )


if __name__ == "__main__":
    host = CFG.get("server", {}).get("host", "0.0.0.0")
    port = int(CFG.get("server", {}).get("port", 7861))
    uvicorn.run("app:app", host=host, port=port, reload=False)
