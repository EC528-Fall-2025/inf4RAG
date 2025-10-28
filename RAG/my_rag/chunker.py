from __future__ import annotations
import re
from typing import List


class SimpleTextSplitter:
    """
    A simple character-level text splitter.
    This can later be replaced with a more advanced version
    that splits by tokens (e.g., tiktoken) or by sentences/paragraphs.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        min_chunk_chars: int = 50
    ) -> None:
        """
        Initialize the text splitter with chunk configuration.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between consecutive chunks.
            min_chunk_chars: Minimum number of characters required for a valid chunk.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    def _normalize(
        self,
        text: str,
        lowercase: bool = True,
        normalize_ws: bool = True
    ) -> str:
        """
        Normalize text by applying lowercase and whitespace cleanup.

        Args:
            text: Input text string.
            lowercase: Whether to convert text to lowercase.
            normalize_ws: Whether to normalize whitespace characters.

        Returns:
            Normalized text string.
        """
        if lowercase:
            text = text.lower()
        if normalize_ws:
            # Replace multiple whitespace characters with a single space
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def split(
        self,
        text: str,
        lowercase: bool = True,
        normalize_ws: bool = True
    ) -> List[str]:
        """
        Split text into overlapping chunks based on character length.

        Args:
            text: The input text to split.
            lowercase: Whether to apply lowercase normalization.
            normalize_ws: Whether to normalize whitespace before splitting.

        Returns:
            A list of text chunks, each with up to `chunk_size` characters.
        """
        text = self._normalize(text, lowercase, normalize_ws)
        chunks: List[str] = []

        if not text:
            return chunks

        n = len(text)
        start = 0

        # Iterate over text and extract overlapping chunks
        while start < n:
            end = min(n, start + self.chunk_size)
            chunk = text[start:end]

            # Only keep chunks that meet the minimum length requirement
            if len(chunk) >= self.min_chunk_chars:
                chunks.append(chunk)

            # Stop if we've reached the end of the text
            if end == n:
                break

            # Move the window with overlap
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end

        return chunks
