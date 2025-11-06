from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import orjson as json

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import RAGConfig
from .index import ChunkMeta
from .embeddings import Embeddings


class RAGQdrantIndex:
    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self.doc_texts: List[str] = []
        self.doc_meta: List[ChunkMeta] = []
        self.dim: int | None = None
        self.dataset_id: str | None = None

    def _client(self) -> QdrantClient:
        return QdrantClient(url=self.cfg.qdrant_url, api_key=self.cfg.qdrant_api_key)

    def _collection_name(self, dataset_id: str) -> str:
        return f"{self.cfg.qdrant_collection_prefix}_{dataset_id}"

    def build(self, dataset_id: str, chunks: List[Tuple[str, ChunkMeta]]) -> None:
        self.dataset_id = dataset_id
        self.doc_texts = [c[0] for c in chunks]
        self.doc_meta = [c[1] for c in chunks]
        if not self.doc_texts:
            raise ValueError("No texts to index")

        emb = Embeddings(self.cfg.embedding_provider, self.cfg.embedding_model)
        vecs = emb.embed(self.doc_texts)
        self.dim = int(vecs.shape[1])

        client = self._client()
        cname = self._collection_name(dataset_id)
        # recreate collection (drops existing if present)
        client.recreate_collection(
            collection_name=cname,
            vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
        )

        points = []
        payloads = []
        for idx, (v, meta) in enumerate(zip(vecs, self.doc_meta)):
            points.append(v.tolist())
            payloads.append({
                "doc_path": meta.doc_path,
                "chunk_id": meta.chunk_id,
                "start_char": meta.start_char,
                "end_char": meta.end_char,
                "text": self.doc_texts[idx],
            })

        client.upsert(collection_name=cname, points=qmodels.Batch(
            vectors=points,
            payloads=payloads,
            ids=list(range(len(points)))
        ))

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        emb = Embeddings(self.cfg.embedding_provider, self.cfg.embedding_model)
        q = emb.embed([query])[0]
        client = self._client()
        assert self.dataset_id, "Qdrant index missing dataset_id"
        cname = self._collection_name(self.dataset_id)
        # Basic existence check: attempt to retrieve collection info
        try:
            client.get_collection(cname)
        except Exception as e:
            raise RuntimeError(f"Qdrant collection '{cname}' not found. Did you ingest the dataset? Original error: {e}")
        res = client.search(collection_name=cname, query_vector=q.tolist(), limit=top_k)
        out: List[Tuple[int, float]] = []
        for r in res:
            # we stored ids as 0..N-1 matching array order
            out.append((int(r.id), float(r.score)))
        return out

    def save(self, index_dir: Path) -> None:
        # persist only metadata and texts locally (vectors live in Qdrant)
        meta_out = [m.__dict__ for m in self.doc_meta]
        blob = {"meta": meta_out, "texts": self.doc_texts, "dim": self.dim, "dataset_id": self.dataset_id}
        (index_dir / self.cfg.meta_file).write_bytes(json.dumps(blob))

    @classmethod
    def load(cls, cfg: RAGConfig, index_dir: Path) -> "RAGQdrantIndex":
        obj = cls(cfg)
        blob = json.loads((index_dir / cfg.meta_file).read_bytes())
        obj.doc_texts = blob["texts"]
        obj.doc_meta = [ChunkMeta(**m) for m in blob["meta"]]
        obj.dim = blob.get("dim")
        obj.dataset_id = blob.get("dataset_id")
        return obj
