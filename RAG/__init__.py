"""
RAG - Retrieval-Augmented Generation Package
"""

from .config import RAGConfig
from .pipeline import RAGPipeline, GenerateResult

__version__ = "1.0.0"

__all__ = ['RAGConfig', 'RAGPipeline', 'GenerateResult']
