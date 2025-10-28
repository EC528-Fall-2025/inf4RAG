"""Upload service layer."""

from __future__ import annotations
from pathlib import Path
import tempfile

from ..config import RAGConfig
from ..pipeline import RAGPipeline
from ..utils.file_processor import process_zip_upload


class UploadService:
    """Service for handling document uploads and ingestion."""
    
    def __init__(self, config: RAGConfig, pipeline: RAGPipeline):
        self.config = config
        self.pipeline = pipeline
    
    def upload_and_ingest(self, file_path: Path, dataset_id: str) -> dict:
        """
        Upload a zip file, process it, and build the index.
        
        Returns:
            dict with keys: status, message, dataset_id, file_count
        """
        try:
            # Process zip file
            file_count, raw_dir = process_zip_upload(
                file_path, 
                self.config, 
                dataset_id
            )
            
            # Build index
            self.pipeline.ingest(dataset_id)
            
            return {
                "status": "success",
                "message": f"Successfully processed {file_count} files",
                "dataset_id": dataset_id,
                "file_count": file_count
            }
            
        except ValueError as e:
            return {
                "status": "error",
                "message": str(e),
                "dataset_id": dataset_id
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Upload failed: {str(e)}",
                "dataset_id": dataset_id
            }
