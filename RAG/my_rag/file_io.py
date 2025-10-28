from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator
import orjson as json

from .config import RAGConfig


def ensure_dir(path: Path) -> None:
    """Create a directory (and its parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def iter_files(root: Path, include_exts: Iterable[str]) -> Iterator[Path]:
    """
    Recursively iterate through all files in the given root directory
    and yield files whose extensions match the allowed list.
    """
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in include_exts:
            yield p


def read_text_file(path: Path) -> str:
    """
    Read the content of a text file as a UTF-8 string.

    Note:
        Even for CSV or JSON files, this function treats them as plain text.
        If UTF-8 decoding fails, it falls back to Latin-1 encoding.
    """
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def allocate_dataset(cfg: RAGConfig, user_id: str, dataset_name: str) -> tuple[str, Path]:
    """
    Allocate a new dataset directory for the given user and dataset name.

    Returns:
        (dataset_id, raw_dir)
        - dataset_id: Unique identifier for the dataset.
        - raw_dir: Directory path where uploaded or extracted files should be placed.
    """
    from uuid import uuid4

    # Generate a unique dataset identifier
    dataset_id = f"{user_id}-{dataset_name}-{uuid4().hex[:8]}"

    # Create the raw data directory
    raw_dir = cfg.raw_dir(dataset_id)
    ensure_dir(raw_dir)

    # Pre-create the index directory for vector indexing or embeddings
    ensure_dir(cfg.index_dir(dataset_id))

    # Write dataset metadata (includes user info and dataset name)
    meta = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "dataset_name": dataset_name,
    }
    (cfg.dataset_dir(dataset_id) / "dataset.json").write_bytes(
        json.dumps(meta, option=json.OPT_INDENT_2)
    )

    return dataset_id, raw_dir
