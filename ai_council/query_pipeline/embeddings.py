"""EmbeddingEngine — fast, dependency-free text embeddings.

Default backend: **hash-based TF-IDF projection** into a fixed 384-dim float32
space.  Works out-of-the-box with no model downloads and no external APIs.

Architecture
------------
1. Tokenise input into n-grams (unigram + bigram).
2. Hash each n-gram to a bucket in [0, dim).
3. Accumulate signed ±1 contributions (feature hashing / Vowpal Wabbit style).
4. L2-normalise the result vector.

This gives a deterministic, consistent embedding that captures approximate
term-overlap similarity — sufficient for topic classification with a seeded
vector store.

Swap-in backends (future)
--------------------------
- ``SentenceTransformerBackend`` — uses ``sentence-transformers`` library.
- ``OpenAIEmbeddingBackend``     — uses ``openai.embeddings.create``.

Both implement the same ``EmbeddingBackend`` ABC so the engine is backend-agnostic.
"""

from __future__ import annotations

import hashlib
import logging
import re
import struct
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Backend ABC
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingBackend(ABC):
    """Abstract embedding backend."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> np.ndarray: ...

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])


# ─────────────────────────────────────────────────────────────────────────────
# Hash-based backend (default, zero deps)
# ─────────────────────────────────────────────────────────────────────────────

class HashEmbeddingBackend(EmbeddingBackend):
    """Deterministic feature-hashing embedding (Vowpal-Wabbit style).

    Time complexity — O(n_tokens) per document.
    No model download, no external API, no GPU.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    # ── tokenisation ────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    # ── hashing ──────────────────────────────────────────────────────────────

    def _hash_token(self, token: str) -> tuple[int, int]:
        """Return (bucket_index, sign) for a token.

        Uses the first 8 bytes of SHA-256 so the distribution is uniform.
        """
        digest = hashlib.sha256(token.encode()).digest()
        val = struct.unpack_from(">Q", digest)[0]          # unsigned 64-bit
        bucket = val % self._dim
        sign = 1 if (val >> 32) & 1 else -1               # sign from upper 32-bits
        return bucket, sign

    # ── encoding ─────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        tokens = self._tokenise(text)
        vec = np.zeros(self._dim, dtype=np.float32)
        if not tokens:
            return vec

        for token in tokens:
            bucket, sign = self._hash_token(token)
            vec[bucket] += sign

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ─────────────────────────────────────────────────────────────────────────────
# (Optional) SentenceTransformer backend — imported lazily
# ─────────────────────────────────────────────────────────────────────────────

class SentenceTransformerBackend(EmbeddingBackend):
    """Wraps ``sentence-transformers``; lazy import so it's truly optional."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(model_name)
            self._dim_val: int = self._model.get_sentence_embedding_dimension()
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

    @property
    def dim(self) -> int:
        return self._dim_val

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingEngine — public interface
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """High-level embedding engine with per-query LRU cache.

    Args:
        backend:    An :class:`EmbeddingBackend` instance.
        cache_size: Max number of embeddings to keep in memory (LRU eviction).

    Example::

        engine = EmbeddingEngine.default()
        vec = engine.embed("Explain quicksort algorithm")
        assert vec.shape == (384,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5   # unit norm
    """

    def __init__(
        self,
        backend: Optional[EmbeddingBackend] = None,
        cache_size: int = 1024,
    ):
        self._backend: EmbeddingBackend = backend or HashEmbeddingBackend()
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._hits = 0
        self._misses = 0

    # ── factories ───────────────────────────────────────────────────────────

    @classmethod
    def default(cls, dim: int = 384, cache_size: int = 1024) -> "EmbeddingEngine":
        """Build with the hash-based backend (no deps required)."""
        return cls(backend=HashEmbeddingBackend(dim=dim), cache_size=cache_size)

    @classmethod
    def from_config(cls, backend: str = "hash", model_name: str = "hash-384",
                    dim: int = 384, cache_size: int = 1024) -> "EmbeddingEngine":
        if backend == "sentence_transformers":
            return cls(backend=SentenceTransformerBackend(model_name), cache_size=cache_size)
        # Default: hash
        return cls(backend=HashEmbeddingBackend(dim=dim), cache_size=cache_size)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        return self._backend.dim

    # ── public API ──────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-norm float32 embedding for *text* (cached)."""
        key = text.strip()
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        vec = self._backend.encode(key)

        # LRU eviction
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = vec
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts; uses cache per item."""
        return np.stack([self.embed(t) for t in texts])

    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        self._hits = self._misses = 0
