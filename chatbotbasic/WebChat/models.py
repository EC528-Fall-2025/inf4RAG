# -*- coding: utf-8 -*-
import os
import logging
from typing import Any, Dict, Optional

import requests
import yaml


class ChatModel:
    """Minimal wrapper around an OpenAI-compatible /chat/completions endpoint.

    This class mirrors the behavior of the previous `Model` / `OpenAIModel`
    implementation:

    * If a config.yaml file is available, it is used to configure the
      base URL, API key and default model name.
    * At start-up we call the `/v1/models` endpoint once to auto-detect
      the model name. If that fails we fall back to a default.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        config_path: Optional[str] = None,
    ) -> None:
        # Resolve config path (same convention as app.py)
        if config_path is None:
            config_path = os.getenv(
                "WEBCHAT_CONFIG",
                os.path.join(os.path.dirname(__file__), "config.yaml"),
            )

        config: Dict[str, Any] = {}
        if not (base_url and api_key and model):
            # Only load YAML if some values were not passed explicitly.
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    config = data
            except FileNotFoundError:
                logging.warning("ChatModel config file %s not found; using defaults.", config_path)
            except Exception as e:
                logging.warning(
                    "ChatModel failed to load %s: %s; using defaults.",
                    config_path,
                    e,
                )

        openstack_ip_port = config.get("openstack_ip_port")
        default_base = (
            f"http://{openstack_ip_port}/v1"
            if openstack_ip_port
            else "http://127.0.0.1:8000/v1"
        )

        self.base_url = (
            base_url
            or config.get("model_api_base")
            or os.getenv("OPENAI_BASE_URL")
            or default_base
        ).rstrip("/")

        self.api_key = api_key or config.get("api_key") or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.timeout = timeout

        default_model = (
            model
            or config.get("default_model")
            or os.getenv("OPENAI_MODEL", "Qwen2.5-7B-Instruct")
        )

        self.model = default_model
        self.connected = False

        try:
            self.model = self._autodetect_model_name(default_model)
            self.connected = True
            logging.info("ChatModel auto-detected model: %s", self.model)
        except Exception as e:
            logging.warning(
                "ChatModel could not auto-detect model; using fallback %s (%s)",
                default_model,
                e,
            )

    def _autodetect_model_name(self, fallback: str) -> str:
        """Call the OpenAI-compatible /models endpoint once and return
        the first model id.

        Raises on HTTP errors; the caller decides whether to swallow or
        propagate.
        """
        url = f"{self.base_url}/models"
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        models = payload.get("data") or payload.get("models") or []
        if isinstance(models, list) and models:
            first = models[0]
            if isinstance(first, dict):
                return first.get("id", fallback)
            return str(first)
        return fallback

    def _chat(
        self,
        messages,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stream: bool = False,
    ):
        """Low-level helper: send a /chat/completions request and return
        the raw JSON response.
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

    def complete(
        self,
        system_prompt: str,
        user_messages,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Non-streaming completion helper.

        `system_prompt`: optional system message text.
        `user_messages`: list of {role, content} messages.
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

    def generate(
        self,
        system_prompt: str,
        user_messages,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """Backwards-compatible generator-style interface.

        Yields a single full response string (non-streaming), matching
        the previous WebChat code paths.
        """
        yield self.complete(system_prompt, user_messages, temperature, max_tokens)
