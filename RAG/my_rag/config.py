from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RAGConfig:
    # Root directory for storing persistent datasets and index files
    persistent_dir: Path = Path("persistent_data")

    # Supported file extensions (extendable to PDF/HTML, etc.)
    include_exts: tuple[str, ...] = (
        ".txt", ".md", ".markdown", ".py", ".rst",
        ".json", ".csv", ".yml", ".yaml"
    )

    # Text chunking configuration
    chunk_size: int = 800          # Number of characters per chunk
    chunk_overlap: int = 120       # Overlapping characters between chunks
    min_chunk_chars: int = 50      # Minimum valid chunk length; shorter chunks are discarded

    # Retrieval configuration
    default_top_k: int = 3         # Default number of results to retrieve

    # Preprocessing options
    lowercase: bool = True         # Convert text to lowercase
    normalize_ws: bool = True      # Normalize whitespace characters

    # Filenames for saving intermediate and final outputs
    vocab_file: str = "vocab.json"
    index_file: str = "index.npz"
    meta_file: str = "meta.json"

    def dataset_dir(self, dataset_id: str) -> Path:
        """Return the path to the dataset directory."""
        return self.persistent_dir / "datasets" / dataset_id

    def raw_dir(self, dataset_id: str) -> Path:
        """Return the path to the directory containing raw data."""
        return self.dataset_dir(dataset_id) / "raw"

    def index_dir(self, dataset_id: str) -> Path:
        """Return the path to the directory containing index files."""
        return self.dataset_dir(dataset_id) / "index"
