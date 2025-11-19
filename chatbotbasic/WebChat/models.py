# -*- coding: utf-8 -*-
import os
import requests

class ChatModel:
    """
    Minimal wrapper around an OpenAI-compatible /chat/completions endpoint.
    """

    def __init__(self, base_url=None, api_key=None, model=None, timeout=60):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.getenv("OPENAI_MODEL", "Qwen2.5-7B-Instruct")
        self.timeout = timeout

    def _chat(self, messages, temperature=0.3, max_tokens=1024, stream=False):
        """
        Low-level helper: send a /chat/completions request and return the raw JSON.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def complete(self, system_prompt: str, user_messages, temperature=0.3, max_tokens=1024) -> str:
        """
        Non-streaming completion.
        system_prompt: optional system message text.
        user_messages: list of {role, content} messages (user/assistant).
        """
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(user_messages)
        try:
            data = self._chat(
                msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Model error] {e}"

    def generate(self, system_prompt: str, user_messages, temperature=0.3, max_tokens=1024):
        """
        Backwards-compatible interface for previous code paths.
        This simply yields a single full response string (non-streaming).
        """
        yield self.complete(system_prompt, user_messages, temperature, max_tokens)
