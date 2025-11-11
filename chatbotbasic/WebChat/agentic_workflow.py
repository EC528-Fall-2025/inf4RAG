"""
Agentic workflow:
- Option A (default): call vLLM via OpenAI-compatible /v1/chat/completions
- Option B (optional): route to Qwen-Agent if enabled (best-effort; wrapped in try/except)
- If RAG enabled, prepend retrieved context to the system/user messages
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import requests


class VLLMChatClient:
    """Minimal OpenAI-compatible client for chat.completions."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "none"
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class QwenAgentClient:
    """
    Optional agent using qwen-agent (if installed). We keep it defensive:
    - If qwen-agent isn't available or errors out, we raise a clear exception.
    """

    def __init__(self, model: str, dashscope_api_key: str) -> None:
        self.model = model
        self.api_key = dashscope_api_key

        try:
            from qwen_agent.agents import Assistant
            from qwen_agent.llm import get_chat_model
        except Exception as e:
            raise RuntimeError("qwen-agent is not installed or failed to import") from e

        from qwen_agent.agents import Assistant
        from qwen_agent.llm import get_chat_model

        self.llm = get_chat_model(model=self.model, api_key=self.api_key)
        self.assistant = Assistant(self.llm)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Convert OpenAI-style message list into a single prompt for the agent
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        prompt = "\n\n".join(user_turns[-2:]) if user_turns else ""
        reply = self.assistant.run(prompt)
        return str(reply)


def build_messages(
    system_prompt: str,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    context_chunks: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Compose OpenAI-style messages array with optional retrieved context."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        if context_chunks:
            context_block = "\n\n".join([f"- {c}" for c in context_chunks])
            system_prompt = f"{system_prompt.strip()}\n\nRelevant Context:\n{context_block}"
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages
