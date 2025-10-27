"""File processing utilities for handling uploads and zip extraction."""

from __future__ import annotations
from pathlib import Path
import zipfile
import tempfile
import shutil
from typing import Tuple

from ..config import RAGConfig


def safe_path_join(base: Path, path: str) -> Path:
    """Safely join paths to prevent directory traversal attacks."""
    full_path = (base / path).resolve()
    if not str(full_path).startswith(str(base.resolve())):
        raise ValueError(f"Path traversal detected: {path}")
    return full_path


def process_zip_upload(zip_file_path: Path, config: RAGConfig, dataset_id: str) -> Tuple[int, Path]:
    """Process uploaded zip file: extract, validate, and copy to raw_dir."""
    import zipfile
    import tempfile
    import shutil
    
    raw_dir = config.raw_dir(dataset_id)
    file_count = 0
    
    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extract_dir = temp_path / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        # Extract zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Copy files to raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in extract_dir.rglob("*"):
            if file_path.is_file():
                # Check if file extension is supported
                if file_path.suffix.lower() in config.include_exts:
                    # Create relative path to preserve structure
                    rel_path = file_path.relative_to(extract_dir)
                    target_path = raw_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(file_path, target_path)
                    file_count += 1
        
        if file_count == 0:
            raise ValueError("No supported files found in zip archive")
    
    return file_count, raw_dir
