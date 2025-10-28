from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np

Token = str


def simple_tokenize(text: str) -> List[Token]:
    """
    A simple alphanumeric tokenizer.
    Splits text into tokens consisting of letters, digits, or underscores.

    Note:
        This is a lightweight implementation that can be replaced
        with a more advanced tokenizer (e.g., spaCy, NLTK, tiktoken).
    """
    return [t for t in re_split(text)]


# Precompiled regular expression for performance
_re = re.compile(r"[a-zA-Z0-9_]+")


def re_split(text: str) -> List[str]:
    """Return all alphanumeric tokens found in the given text."""
    return _re.findall(text)


class TFIDFVectorizer:
    """
    Lightweight TF-IDF vectorizer implemented purely with NumPy.

    Features:
        - No dependency on scikit-learn.
        - Computes IDF using the formula:
              idf = log((N + 1) / (df + 1)) + 1
        - Applies L2 normalization to each resulting vector.
    """

    def __init__(self) -> None:
        """Initialize an empty vocabulary and IDF vector."""
        self.vocab: Dict[Token, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, texts: List[str]) -> None:
        """
        Build vocabulary and compute IDF values based on the given corpus.

        Args:
            texts: List of input documents as strings.
        """
        df_counter: Dict[Token, int] = defaultdict(int)

        # Count how many documents contain each token
        for text in texts:
            toks = set(simple_tokenize(text))
            for t in toks:
                df_counter[t] += 1

        # Assign an index to each token
        self.vocab = {t: i for i, t in enumerate(sorted(df_counter.keys()))}

        N = len(texts)
        df = np.zeros(len(self.vocab), dtype=np.float32)

        # Fill document frequency values
        for t, i in self.vocab.items():
            df[i] = df_counter[t]

        # Compute IDF vector
        self.idf = np.log((N + 1) / (df + 1)) + 1.0

    def _tfidf_vec(self, text: str) -> np.ndarray:
        """
        Compute the TF-IDF vector for a single text input.

        Args:
            text: Input document as a string.

        Returns:
            A normalized TF-IDF vector as a NumPy array.
        """
        assert self.idf is not None, "TFIDFVectorizer must be fitted before use."

        counts = Counter(simple_tokenize(text))
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        length = sum(counts.values()) or 1

        # Compute term frequency (TF)
        for t, c in counts.items():
            idx = self.vocab.get(t)
            if idx is not None:
                tf = c / length
                vec[idx] = tf * self.idf[idx]

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform a list of texts into a TF-IDF matrix.

        Args:
            texts: List of input documents.

        Returns:
            A 2D NumPy array where each row is a normalized TF-IDF vector.
        """
        return np.stack([self._tfidf_vec(t) for t in texts], axis=0)

    # Serialization / deserialization helpers (used with index storage)
    def to_dict(self) -> Dict:
        """
        Serialize the vectorizer to a dictionary for saving.
        """
        return {
            "vocab": self.vocab,
            "idf": self.idf.tolist() if self.idf is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> TFIDFVectorizer:
        """
        Reconstruct a TFIDFVectorizer instance from a dictionary.

        Args:
            d: Dictionary returned by `to_dict()`.

        Returns:
            TFIDFVectorizer instance with restored vocabulary and IDF.
        """
        obj = cls()
        obj.vocab = {k: int(v) for k, v in d["vocab"].items()}
        obj.idf = (
            np.array(d["idf"], dtype=np.float32)
            if d.get("idf") is not None
            else None
        )
        return obj
