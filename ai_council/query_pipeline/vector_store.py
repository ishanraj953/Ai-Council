"""VectorStore — similarity search over topic exemplar embeddings.

Default backend: brute-force L2 search with NumPy (FAISS-compatible interface).
Optional FAISS backend activated automatically when ``faiss`` is importable.

Design decisions
----------------
* Topic registry is built at construction time by embedding ~20 exemplar
  phrases per topic. At query time only a single ``np.dot`` call is needed.
* L2 distance ≡ cosine distance for unit-norm vectors (since ‖a-b‖²= 2-2a·b),
  so we use ``np.dot`` for speed.
* The store is intentionally **read-heavy / write-light**: exemplars are added
  once at startup; runtime queries only call ``search_topk``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    topic_id: str
    distance: float          # lower = more similar (L2)
    similarity: float        # 1 - normalised distance, in [0, 1]
    context_chunks: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in seed topics
# ─────────────────────────────────────────────────────────────────────────────

SEED_TOPICS: Dict[str, Dict] = {
    "coding": {
        "exemplars": [
            "write a Python function", "implement binary search", "debug this code",
            "fix the syntax error", "how to sort a list", "create a REST API",
            "explain recursion with code", "write unit tests for this function",
            "refactor this class", "implement a linked list", "write SQL query",
            "javascript async await", "how to use decorators", "OOP in Python",
            "implement quicksort algorithm", "write a web scraper",
            "create a database schema", "how to handle exceptions", "git merge conflict",
            "write a bash script",
        ],
        "context_chunks": [
            "Programming and software development tasks",
            "Code generation, debugging, and implementation",
        ],
    },
    "math": {
        "exemplars": [
            "solve this equation", "calculate the integral", "prove this theorem",
            "find the derivative", "matrix multiplication", "probability of event",
            "linear algebra basics", "statistics mean median mode",
            "solve differential equation", "calculate eigenvalues",
            "combinatorics permutation", "number theory prime factorization",
            "geometry area of circle", "trigonometry sin cos tan",
            "calculus limit definition", "algebra quadratic formula",
            "set theory union intersection", "graph theory shortest path",
            "numerical methods Newton Raphson", "Fourier transform",
        ],
        "context_chunks": [
            "Mathematical computation and proofs",
            "Algebra, calculus, statistics, and numerical methods",
        ],
    },
    "general_qa": {
        "exemplars": [
            "what is the capital of France", "who invented the telephone",
            "when did World War 2 end", "what is the speed of light",
            "how many planets in solar system", "what is the population of India",
            "who wrote Harry Potter", "what causes earthquakes", "explain photosynthesis",
            "what is democracy", "how does the immune system work",
            "what is the water cycle", "history of ancient Rome",
            "what is climate change", "how does wifi work",
            "what is quantum physics", "explain the theory of relativity",
            "what is DNA", "how does the heart pump blood",
            "what is machine learning",
        ],
        "context_chunks": [
            "General knowledge and factual questions",
            "History, science, geography, and general trivia",
        ],
    },
    "reasoning": {
        "exemplars": [
            "analyze the trade-offs", "compare and contrast approaches",
            "what are the pros and cons", "reason step by step",
            "evaluate this argument", "what is the logical conclusion",
            "identify the fallacy", "if-then reasoning",
            "causal analysis of the problem", "predict the outcome",
            "critically evaluate this claim", "what are the implications",
            "argue for and against", "systematic approach to decision",
            "root cause analysis", "inductive vs deductive reasoning",
            "evaluate evidence for claim", "what assumptions are being made",
            "synthesize these viewpoints", "what is the strongest counterargument",
        ],
        "context_chunks": [
            "Complex reasoning, analysis, and critical thinking",
            "Multi-step logic, arguments, and evaluation",
        ],
    },
    "research": {
        "exemplars": [
            "find information about", "research the topic of",
            "summarize recent papers on", "what does the literature say",
            "gather data on this subject", "investigate the causes of",
            "literature review on AI", "survey of methods for",
            "compare studies on", "review academic papers about",
            "what research exists on", "explore the history of",
            "collect evidence for", "what are the latest findings",
            "systematic review of", "meta-analysis of studies",
            "find sources about", "research methodology for",
            "bibliography on the topic", "what do experts say about",
        ],
        "context_chunks": [
            "Information gathering and literature review",
            "Academic research, surveys, and data collection",
        ],
    },
    "creative": {
        "exemplars": [
            "write a short story", "compose a poem", "create a song",
            "write a creative essay", "imagine a world where",
            "write dialogue for characters", "create an advertisement",
            "write a movie plot", "compose a haiku", "creative writing prompt",
            "write a children's book", "create a fictional character",
            "write a comedy sketch", "brainstorm creative ideas",
            "design a logo concept", "write marketing copy",
            "create a narrative", "write a product description",
            "compose a speech", "write a blog post",
        ],
        "context_chunks": [
            "Creative writing, storytelling, and content generation",
            "Fiction, poetry, marketing copy, and imaginative tasks",
        ],
    },
    "data_analysis": {
        "exemplars": [
            "analyze this dataset", "visualize the data", "find trends in the data",
            "calculate statistics", "data cleaning and preprocessing",
            "predict future values", "cluster this data", "feature engineering",
            "train a machine learning model", "evaluate model performance",
            "correlation between variables", "time series analysis",
            "anomaly detection", "classification problem", "regression analysis",
            "pivot table analysis", "exploratory data analysis",
            "data pipeline design", "ETL process", "analyze CSV file",
        ],
        "context_chunks": [
            "Data science, analytics, and machine learning",
            "Statistical analysis, visualization, and predictive modeling",
        ],
    },
    "debugging": {
        "exemplars": [
            "why is this code not working", "fix this bug", "traceback error",
            "null pointer exception", "memory leak", "performance issue",
            "why does this test fail", "stack overflow error", "segmentation fault",
            "AttributeError in Python", "TypeError debug", "runtime error",
            "investigate slow query", "debug network issue", "fix broken pipeline",
            "why is the API returning 500", "authentication error",
            "dependency conflict", "environment setup problem", "docker issue",
            # Extra exemplars to disambiguate AttributeError / error-on-line queries
            "debug the AttributeError exception",
            "error on line 42 in Python",
            "exception traceback Python debug",
            "fix the AttributeError in my code",
            "Python throws AttributeError",
            "error message traceback debug fix",
            "why does Python raise AttributeError",
            "investigate the error on this line",
        ],
        "context_chunks": [
            "Bug investigation, error diagnosis, and troubleshooting",
            "Runtime errors, stack traces, and fix suggestions",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """In-memory vector store with top-k L2 nearest-neighbour search.

    Uses NumPy brute-force by default. If ``faiss`` is installed it is used
    transparently for faster search at large scale.

    Args:
        engine:     :class:`~.embeddings.EmbeddingEngine` used to embed exemplars.
        use_faiss:  If True, attempt to use FAISS; fall back to NumPy silently.

    Example::

        from ai_council.query_pipeline.embeddings import EmbeddingEngine
        engine = EmbeddingEngine.default()
        vs = VectorStore(engine)
        vs.seed_default_topics()
        results = vs.search_topk(engine.embed("write a Python function"), k=5)
        assert results[0].topic_id == "coding"
    """

    def __init__(self, engine, *, use_faiss: bool = True):
        self._engine = engine
        self._topic_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None  # shape (N, dim)
        self._context_map: Dict[str, List[str]] = {}
        self._use_faiss = use_faiss and self._faiss_available()
        self._faiss_index = None
        self._n_vectors = 0

    # ── FAISS probe ──────────────────────────────────────────────────────────

    @staticmethod
    def _faiss_available() -> bool:
        try:
            import faiss  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Public API ───────────────────────────────────────────────────────────

    def add_topic(
        self,
        topic_id: str,
        exemplar_texts: List[str],
        context_chunks: Optional[List[str]] = None,
    ) -> None:
        """Embed *exemplar_texts* and add them to the store under *topic_id*."""
        if not exemplar_texts:
            return

        vecs = self._engine.embed_batch(exemplar_texts)   # (N, dim)

        labels = [topic_id] * len(exemplar_texts)
        self._topic_ids.extend(labels)
        self._context_map[topic_id] = context_chunks or []

        if self._embeddings is None:
            self._embeddings = vecs
        else:
            self._embeddings = np.vstack([self._embeddings, vecs])

        self._n_vectors = len(self._topic_ids)
        self._faiss_index = None  # invalidate; rebuilt lazily on next search

        logger.debug("VectorStore: added %d exemplars for topic '%s'.", len(exemplar_texts), topic_id)

    def seed_default_topics(self) -> None:
        """Populate the store with the 8 built-in topics."""
        for topic_id, data in SEED_TOPICS.items():
            self.add_topic(
                topic_id=topic_id,
                exemplar_texts=data["exemplars"],
                context_chunks=data.get("context_chunks", []),
            )
        logger.info("VectorStore: seeded %d topics, %d total exemplars.", len(SEED_TOPICS), self._n_vectors)

    def search_topk(self, query_vec: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Return the *k* nearest topics for *query_vec*.

        Args:
            query_vec: A unit-norm float32 1-D array of length ``engine.dim``.
            k:         Number of nearest neighbours to return.

        Returns:
            List of :class:`SearchResult` sorted by ascending distance
            (most similar first).
        """
        if self._embeddings is None or self._n_vectors == 0:
            return []

        t0 = time.perf_counter()

        if self._use_faiss:
            results = self._search_faiss(query_vec, k)
        else:
            results = self._search_numpy(query_vec, k)

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.debug("VectorStore.search_topk: %d results in %.2f ms.", len(results), elapsed_ms)
        return results

    # ── NumPy search (default) ───────────────────────────────────────────────

    def _search_numpy(self, query_vec: np.ndarray, k: int) -> List[SearchResult]:
        # For unit-norm vectors: L2² = 2 - 2·dot  →  maximise dot = minimise L2
        dots = self._embeddings @ query_vec                     # (N,)
        # Retrieve k * 3 candidates to ensure all topics can appear before dedup
        cand_size = min(k * 3, len(dots))
        top_k_idx = np.argpartition(dots, -cand_size)[-cand_size:]
        top_k_idx = top_k_idx[np.argsort(dots[top_k_idx])[::-1]]

        seen_topics: Dict[str, SearchResult] = {}
        for idx in top_k_idx:
            tid = self._topic_ids[idx]
            dot = float(dots[idx])
            l2_dist = float(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))
            sim = (dot + 1.0) / 2.0  # map [-1,1] → [0,1]

            if tid not in seen_topics or sim > seen_topics[tid].similarity:
                seen_topics[tid] = SearchResult(
                    topic_id=tid,
                    distance=l2_dist,
                    similarity=sim,
                    context_chunks=self._context_map.get(tid, []),
                )

        return sorted(seen_topics.values(), key=lambda r: r.distance)[:k]

    # ── FAISS search (optional fast path) ────────────────────────────────────

    def _build_faiss_index(self) -> None:
        import faiss  # type: ignore
        dim = self._embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.ascontiguousarray(self._embeddings, dtype=np.float32))
        self._faiss_index = index

    def _search_faiss(self, query_vec: np.ndarray, k: int) -> List[SearchResult]:
        import faiss  # type: ignore  # noqa: F401

        if self._faiss_index is None:
            self._build_faiss_index()

        q = np.ascontiguousarray(query_vec[np.newaxis, :], dtype=np.float32)
        distances, indices = self._faiss_index.search(q, min(k * 2, self._n_vectors))

        seen_topics: Dict[str, SearchResult] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            tid = self._topic_ids[idx]
            sim = max(0.0, 1.0 - float(dist) / 4.0)

            if tid not in seen_topics or sim > seen_topics[tid].similarity:
                seen_topics[tid] = SearchResult(
                    topic_id=tid,
                    distance=float(dist),
                    similarity=sim,
                    context_chunks=self._context_map.get(tid, []),
                )

        return sorted(seen_topics.values(), key=lambda r: r.distance)[:k]

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "n_vectors": self._n_vectors,
            "n_topics": len(set(self._topic_ids)),
            "backend": "faiss" if (self._use_faiss and self._faiss_available()) else "numpy",
            "dim": self._embeddings.shape[1] if self._embeddings is not None else 0,
        }
