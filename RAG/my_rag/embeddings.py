from __future__ import annotations
from typing import List
import numpy as np
import os


class Embeddings:
    """Simple pluggable embeddings facade supporting local and OpenAI."""

    def __init__(self, provider: str, model_name: str) -> None:
        self.provider = provider
        self.model_name = model_name
        self._model = None

        if provider == "local":
            from sentence_transformers import SentenceTransformer
            # lazy load
            self._model = SentenceTransformer(model_name)
        elif provider == "openai":
            from openai import OpenAI
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
            self._client = OpenAI()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.provider == "local":
            vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return vecs.astype(np.float32)
        else:
            # OpenAI
            resp = self._client.embeddings.create(model=self.model_name, input=texts)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            # normalize to match cosine/IP usage
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return (vecs / norms).astype(np.float32)
