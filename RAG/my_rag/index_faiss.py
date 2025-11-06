from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import os

import numpy as np
import faiss
import orjson as json

from .config import RAGConfig
from .index import ChunkMeta
from .embeddings import Embeddings


class RAGFaissIndex:
    """
    FAISS-backed semantic index using OpenAI embeddings.
    Requires OPENAI_API_KEY in environment.
    """

    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self.doc_texts: List[str] = []
        self.doc_meta: List[ChunkMeta] = []
        self.index: faiss.Index | None = None
        self.dim: int | None = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        emb = Embeddings(self.cfg.embedding_provider, self.cfg.embedding_model)
        return emb.embed(texts)

    def build(self, chunks: List[Tuple[str, ChunkMeta]]) -> None:
        self.doc_texts = [c[0] for c in chunks]
        self.doc_meta = [c[1] for c in chunks]

        if not self.doc_texts:
            raise ValueError("No texts to index")

        vecs = self._embed(self.doc_texts)
        self.dim = vecs.shape[1]
        # normalize for cosine similarity via inner product
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        assert self.index is not None, "Index is not built"
        q = self._embed([query]).astype(np.float32)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        idxs = I[0]
        sims = D[0]
        results: List[Tuple[int, float]] = []
        for i, s in zip(idxs, sims):
            if i == -1:
                continue
            results.append((int(i), float(s)))
        return results

    def save(self, index_dir: Path) -> None:
        assert self.index is not None
        index_dir.mkdir(parents=True, exist_ok=True)
        # FAISS index
        faiss.write_index(self.index, str(index_dir / self.cfg.faiss_index_file))
        # metadata
        meta_out = [m.__dict__ for m in self.doc_meta]
        blob = {"meta": meta_out, "texts": self.doc_texts, "dim": self.dim}
        (index_dir / self.cfg.meta_file).write_bytes(json.dumps(blob))

    @classmethod
    def load(cls, cfg: RAGConfig, index_dir: Path) -> "RAGFaissIndex":
        obj = cls(cfg)
        obj.index = faiss.read_index(str(index_dir / cfg.faiss_index_file))
        blob = json.loads((index_dir / cfg.meta_file).read_bytes())
        obj.doc_texts = blob["texts"]
        obj.doc_meta = [ChunkMeta(**m) for m in blob["meta"]]
        obj.dim = blob.get("dim")
        return obj
