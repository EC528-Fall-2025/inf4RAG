from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .config import RAGConfig
from .file_io import ensure_dir, iter_files, read_text_file
from .chunker import SimpleTextSplitter
from .index import RAGIndex, ChunkMeta
from .index_faiss import RAGFaissIndex
from .index_qdrant import RAGQdrantIndex
from .prompts import build_prompt


@dataclass
class GenerateResult:
    """
    Container for query generation results.
    Includes retrieved document info, ranking metadata, and the final prompt.
    """
    dataset_id: str
    query: str
    top_k: int
    retrieved: List[Dict]
    prompt: str


class RAGPipeline:
    """
    The main orchestration class for the RAG (Retrieval-Augmented Generation) pipeline.

    Workflow:
        ingest  ->  build index  ->  search  ->  assemble generation prompt
    """

    def __init__(self, cfg: RAGConfig | None = None) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            cfg: Optional RAGConfig instance. If None, a default configuration is used.
        """
        self.cfg = cfg or RAGConfig()

    # ---------- Ingestion: Parse raw data and build index ----------
    def ingest(self, dataset_id: str) -> None:
        """
        Ingest the dataset and build its retrieval index.

        This function:
            - Reads all text files under the raw dataset directory.
            - Splits documents into text chunks.
            - Builds and saves the TF-IDF-based retrieval index.

        Args:
            dataset_id: The unique dataset identifier.
        """
        raw_dir = self.cfg.raw_dir(dataset_id)
        index_dir = self.cfg.index_dir(dataset_id)
        ensure_dir(index_dir)

        splitter = SimpleTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            min_chunk_chars=self.cfg.min_chunk_chars,
        )

        chunks: List[Tuple[str, ChunkMeta]] = []

        # Iterate through all allowed files and split text into chunks
        for fp in iter_files(raw_dir, self.cfg.include_exts):
            text = read_text_file(fp)
            offset = 0
            for i, chunk in enumerate(
                splitter.split(
                    text,
                    lowercase=self.cfg.lowercase,
                    normalize_ws=self.cfg.normalize_ws,
                )
            ):
                meta = ChunkMeta(
                    doc_path=str(fp),
                    chunk_id=i,
                    start_char=offset,
                    end_char=offset + len(chunk),
                )
                chunks.append((chunk, meta))
                offset += len(chunk)  # Approximate offset tracking

        # Build and persist the retrieval index
        if self.cfg.backend == "faiss":
            index = RAGFaissIndex(self.cfg)
            index.build(chunks)
            index.save(index_dir)
        elif self.cfg.backend == "qdrant":
            index = RAGQdrantIndex(self.cfg)
            index.build(dataset_id, chunks)
            index.save(index_dir)
        else:
            index = RAGIndex(self.cfg)
            index.build(chunks)
            index.save(index_dir)

    # ---------- Query: Retrieve and build prompt ----------
    def generate(
        self,
        dataset_id: str,
        query: str,
        top_k: int | None = None
    ) -> GenerateResult:
        """
        Perform retrieval for a given query and construct the generation prompt.

        Args:
            dataset_id: The dataset to query.
            query: The user’s question or input string.
            top_k: Optional number of top results to retrieve (defaults to config value).

        Returns:
            GenerateResult object containing retrieved metadata and constructed prompt.
        """
        index_dir = self.cfg.index_dir(dataset_id)
        if self.cfg.backend == "faiss":
            index = RAGFaissIndex.load(self.cfg, index_dir)
        elif self.cfg.backend == "qdrant":
            index = RAGQdrantIndex.load(self.cfg, index_dir)
        else:
            index = RAGIndex.load(self.cfg, index_dir)
        k = top_k or self.cfg.default_top_k

        # Perform similarity search
        hits = index.search(query, top_k=k)
        retrieved_texts: List[str] = []
        retrieved_meta: List[Dict] = []

        # Collect retrieved chunks and their metadata
        for rank, (idx, score) in enumerate(hits, start=1):
            meta = index.doc_meta[idx]
            text = index.doc_texts[idx]
            retrieved_texts.append(text)
            retrieved_meta.append({
                "rank": rank,
                "score": round(float(score), 6),
                "doc_path": meta.doc_path,
                "chunk_id": meta.chunk_id,
                "start_char": meta.start_char,
                "end_char": meta.end_char,
                "text": text,
            })

        # Construct the final prompt using the retrieved chunks
        prompt = build_prompt(query, retrieved_texts)

        return GenerateResult(
            dataset_id=dataset_id,
            query=query,
            top_k=k,
            retrieved=retrieved_meta,
            prompt=prompt,
        )
