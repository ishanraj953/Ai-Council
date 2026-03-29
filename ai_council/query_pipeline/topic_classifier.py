"""TopicClassifier — classify a user query into a topic using embedding + top-k NN vote.

Algorithm
---------
1. Embed the input text via :class:`~.embeddings.EmbeddingEngine`.
2. Retrieve the top-k nearest exemplars from :class:`~.vector_store.VectorStore`.
3. Majority-vote the topic labels; weight by similarity score.
4. Return :class:`ClassificationResult` with topic, confidence, context chunks, latency.

Target latency: <50 ms (typically <5 ms with hash embeddings + numpy search).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .embeddings import EmbeddingEngine
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Outcome of topic classification for a single query."""
    topic: str
    confidence: float              # 0.0–1.0
    context_chunks: List[str] = field(default_factory=list)
    runner_up: Optional[str] = None
    runner_up_confidence: float = 0.0
    latency_ms: float = 0.0
    top_k_results: List[dict] = field(default_factory=list)   # raw NN results


class TopicClassifier:
    """Classify user queries into pre-registered topics.

    Args:
        engine:   :class:`~.embeddings.EmbeddingEngine` for query embedding.
        store:    :class:`~.vector_store.VectorStore` pre-seeded with topics.
        top_k:    Number of nearest neighbours to retrieve for voting.
        threshold: Minimum weighted vote share to assign a topic; otherwise
                   falls back to ``"general_qa"``.

    Example::

        engine = EmbeddingEngine.default()
        store  = VectorStore(engine)
        store.seed_default_topics()
        clf = TopicClassifier(engine, store)
        result = clf.classify("write a Python quicksort function")
        assert result.topic == "coding"
        assert result.confidence > 0.5
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        store: VectorStore,
        *,
        top_k: int = 5,
        threshold: float = 0.20,
        fallback_topic: str = "general_qa",
    ):
        self._engine = engine
        self._store = store
        self._top_k = top_k
        self._threshold = threshold
        self._fallback = fallback_topic

        # Classification statistics
        self._total = 0
        self._topic_counts: Dict[str, int] = defaultdict(int)

    # ── Main API ─────────────────────────────────────────────────────────────

    def classify(self, text: str) -> ClassificationResult:
        """Classify *text* into a topic.

        Returns a :class:`ClassificationResult` with the winning topic and
        its confidence score.  If no topic exceeds *threshold* the fallback
        topic is returned with ``confidence = 0.0``.
        """
        t0 = time.perf_counter()

        query_vec = self._engine.embed(text)
        results = self._store.search_topk(query_vec, k=self._top_k)

        latency_ms = (time.perf_counter() - t0) * 1_000

        if not results:
            return ClassificationResult(
                topic=self._fallback,
                confidence=0.0,
                latency_ms=latency_ms,
            )

        # Weighted vote: weight per exemplar = similarity score
        vote_weights: Dict[str, float] = defaultdict(float)
        for r in results:
            vote_weights[r.topic_id] += r.similarity

        total_weight = sum(vote_weights.values()) or 1.0
        ranked = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)

        winner_topic, winner_weight = ranked[0]
        winner_confidence = winner_weight / total_weight

        # Context chunks from the winning topic
        # Use the SearchResult with the highest similarity for that topic
        context_chunks: List[str] = []
        for r in results:
            if r.topic_id == winner_topic:
                context_chunks = r.context_chunks
                break

        # Runner-up
        runner_up = runner_up_conf = None
        if len(ranked) > 1:
            runner_up, ru_weight = ranked[1]
            runner_up_conf = ru_weight / total_weight

        # Apply threshold
        if winner_confidence < self._threshold:
            winner_topic = self._fallback
            winner_confidence = 0.0

        self._total += 1
        self._topic_counts[winner_topic] += 1

        logger.debug(
            "[TopicClassifier] topic='%s' confidence=%.3f latency=%.2fms",
            winner_topic, winner_confidence, latency_ms,
        )

        return ClassificationResult(
            topic=winner_topic,
            confidence=winner_confidence,
            context_chunks=context_chunks,
            runner_up=runner_up,
            runner_up_confidence=runner_up_conf or 0.0,
            latency_ms=latency_ms,
            top_k_results=[
                {"topic": r.topic_id, "similarity": r.similarity, "distance": r.distance}
                for r in results
            ],
        )

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_classified": self._total,
            "topic_distribution": dict(self._topic_counts),
        }
