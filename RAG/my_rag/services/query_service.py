"""Query service layer."""

from __future__ import annotations
from typing import Optional

from ..config import RAGConfig
from ..pipeline import RAGPipeline, GenerateResult


class QueryService:
    """Service for handling RAG queries."""
    
    def __init__(self, config: RAGConfig, pipeline: RAGPipeline):
        self.config = config
        self.pipeline = pipeline
    
    def execute_query(
        self, 
        query: str, 
        dataset_id: str = "default-dataset",
        top_k: Optional[int] = None
    ) -> dict:
        """
        Execute RAG query and return results.
        
        Returns:
            dict with status, retrieved documents, and prompt
        """
        # Validate
        if not query or not isinstance(query, str):
            return {
                "status": "error",
                "message": "query is required and must be a string"
            }
        
        # Check if dataset exists
        index_dir = self.config.index_dir(dataset_id)
        if not index_dir.exists():
            return {
                "status": "error",
                "message": f"Dataset '{dataset_id}' not found. Please upload documents first."
            }
        
        # Execute query
        try:
            result: GenerateResult = self.pipeline.generate(
                dataset_id=dataset_id,
                query=query,
                top_k=top_k or self.config.default_top_k
            )
            
            return {
                "status": "success",
                "dataset_id": result.dataset_id,
                "query": result.query,
                "top_k": result.top_k,
                "retrieved_documents": result.retrieved,
                "prompt": result.prompt,
                "retrieved_count": len(result.retrieved)
            }
            
        except FileNotFoundError:
            return {
                "status": "error",
                "message": f"Index for dataset '{dataset_id}' not found"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Query failed: {str(e)}"
            }
