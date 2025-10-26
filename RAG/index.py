from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import orjson as json

from .vectorize import TFIDFVectorizer
from .config import RAGConfig


@dataclass
class ChunkMeta:
    """
    Metadata for each text chunk.
    Stores the original document path, chunk index, and character offsets.
    """
    doc_path: str
    chunk_id: int
    start_char: int
    end_char: int


class RAGIndex:
    """
    The RAGIndex class handles building, saving, loading, and querying
    the document retrieval index for a Retrieval-Augmented Generation (RAG) system.

    Responsibilities:
        - Build TF-IDF embeddings for all text chunks.
        - Store and load index data (vectors, metadata, and vocabulary).
        - Perform top-k similarity searches for user queries.
    """

    def __init__(self, cfg: RAGConfig) -> None:
        """
        Initialize the index with a given configuration.

        Args:
            cfg: RAGConfig object containing directory paths and constants.
        """
        self.cfg = cfg
        self.vectorizer = TFIDFVectorizer()
        self.doc_texts: List[str] = []          # Chunk texts
        self.doc_meta: List[ChunkMeta] = []     # Metadata for each chunk
        self.matrix: np.ndarray | None = None   # TF-IDF matrix: [num_chunks, vocab_size]

    # ---------- Build ----------
    def build(self, chunks: List[Tuple[str, ChunkMeta]]) -> None:
        """
        Build the TF-IDF index from a list of text chunks.

        Args:
            chunks: List of (text, metadata) tuples representing document chunks.
        """
        self.doc_texts = [c[0] for c in chunks]
        self.doc_meta = [c[1] for c in chunks]

        # Fit vocabulary and compute IDF
        self.vectorizer.fit(self.doc_texts)

        # Compute TF-IDF vectors for all chunks
        self.matrix = self.vectorizer.transform(self.doc_texts)  # Shape: [N, V]

    # ---------- Search ----------
    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Search the index for the top-k most relevant chunks.

        Args:
            query: The query text string.
            top_k: Number of top results to retrieve.

        Returns:
            A list of (chunk_index, similarity_score) tuples sorted by similarity.
        """
        assert self.matrix is not None, "Index has not been built."

        # Encode query into a TF-IDF vector
        q_vec = self.vectorizer.transform([query])[0]  # Shape: [V]

        # Compute cosine similarity via dot product (since vectors are L2-normalized)
        sims = self.matrix @ q_vec  # Shape: [N]

        # Efficient partial sort for top-k results
        idx = np.argpartition(-sims, kth=min(top_k, sims.size - 1))[:top_k]
        idx_sorted = idx[np.argsort(-sims[idx])]

        return [(int(i), float(sims[int(i)])) for i in idx_sorted]

    # ---------- Persistence ----------
    def save(self, index_dir: Path) -> None:
        """
        Save the index to disk, including vocabulary, vectors, and metadata.

        Args:
            index_dir: Directory where index files will be stored.
        """
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save vocabulary and IDF values
        (index_dir / self.cfg.vocab_file).write_bytes(
            json.dumps(self.vectorizer.to_dict())
        )

        # Save vector matrix
        assert self.matrix is not None, "Matrix not initialized."
        np.savez_compressed(index_dir / self.cfg.index_file, matrix=self.matrix)

        # Save metadata and raw text chunks
        meta_out = [cm.__dict__ for cm in self.doc_meta]
        blob = {"meta": meta_out, "texts": self.doc_texts}
        (index_dir / self.cfg.meta_file).write_bytes(json.dumps(blob))

    @classmethod
    def load(cls, cfg: RAGConfig, index_dir: Path) -> RAGIndex:
        """
        Load a saved RAGIndex instance from disk.

        Args:
            cfg: RAGConfig object for file paths and naming.
            index_dir: Directory containing saved index files.

        Returns:
            A reconstructed RAGIndex instance with loaded data.
        """
        obj = cls(cfg)

        # Load vocabulary and IDF
        vec = json.loads((index_dir / cfg.vocab_file).read_bytes())
        obj.vectorizer = TFIDFVectorizer.from_dict(vec)

        # Load TF-IDF matrix
        arr = np.load(index_dir / cfg.index_file)
        obj.matrix = arr["matrix"]

        # Load metadata and text chunks
        blob = json.loads((index_dir / cfg.meta_file).read_bytes())
        obj.doc_texts = blob["texts"]
        obj.doc_meta = [ChunkMeta(**m) for m in blob["meta"]]

        return obj
