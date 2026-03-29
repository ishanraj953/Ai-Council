"""QueryPipeline — end-to-end cost-optimized query processing orchestrator.

Pipeline stages (0-indexed):

    Stage 0  SanitizationFilter      → block injections
    Stage 1  QueryCache.lookup()     → short-circuit on cache hit
    Stage 2  EmbeddingEngine.embed() → dense query vector
    Stage 3  VectorStore.search()    → top-k nearest exemplars
    Stage 4  TopicClassifier         → topic label + context
    Stage 5  SmartQueryDecomposer    → sub-queries + dependency order
    Stage 6  ModelRouter.route_all() → tier assignment per sub-query
    Stage 7  TokenOptimizer          → compressed prompt per sub-query
    Stage 8  Execution               → stub (pluggable via execute_fn)
    Stage 9  ResponseAggregator      → merge + cost report
    Stage 10 QueryCache.store()      → persist result

Usage::

    pipeline = QueryPipeline.build()
    result   = pipeline.process("Explain quicksort and give Python code")
    print(result.cost_report)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .cache import CacheStats, QueryCache
from .config import PipelineConfig
from .embeddings import EmbeddingEngine
from .model_router import ModelRouter, ModelTier, RouterResult, RoutingDecision
from .query_decomposer import DecompositionResult, SmartQueryDecomposer, SubQuery
from .token_optimizer import OptimizedPrompt, TokenOptimizer
from .topic_classifier import ClassificationResult, TopicClassifier
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubQueryResult:
    """Execution outcome for a single sub-query."""
    sub_query: SubQuery
    routing: RoutingDecision
    optimized_prompt: OptimizedPrompt
    response: Any = None            # filled by execute_fn
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class CostReport:
    """Cost comparison: optimized pipeline vs always-expensive baseline."""
    baseline_cost_usd: float
    optimized_cost_usd: float
    total_savings_usd: float
    savings_pct: float
    cheap_count: int
    mid_count: int
    expensive_count: int
    token_compression_ratios: List[float] = field(default_factory=list)

    @property
    def avg_compression(self) -> float:
        if not self.token_compression_ratios:
            return 1.0
        return sum(self.token_compression_ratios) / len(self.token_compression_ratios)

    def pretty(self) -> str:
        lines = [
            "=== Cost Report ===",
            f"  Baseline (all-expensive): ${self.baseline_cost_usd:.6f}",
            f"  Optimized:                ${self.optimized_cost_usd:.6f}",
            f"  Savings:                  ${self.total_savings_usd:.6f} ({self.savings_pct:.1f}%)",
            f"  Model tier breakdown: cheap={self.cheap_count}, mid={self.mid_count}, expensive={self.expensive_count}",
            f"  Avg token compression:    {self.avg_compression:.2%} of original",
        ]
        return "\n".join(lines)


@dataclass
class LatencyBreakdown:
    cache_lookup_ms: float = 0.0
    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    classification_ms: float = 0.0
    decomposition_ms: float = 0.0
    routing_ms: float = 0.0
    optimization_ms: float = 0.0
    execution_ms: float = 0.0
    aggregation_ms: float = 0.0
    total_overhead_ms: float = 0.0

    def summary(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PipelineResult:
    """Full result of a pipeline run."""
    query: str
    final_response: Any
    classification: Optional[ClassificationResult]
    decomposition: Optional[DecompositionResult]
    router_result: Optional[RouterResult]
    sub_query_results: List[SubQueryResult]
    cost_report: CostReport
    latency: LatencyBreakdown
    from_cache: bool = False
    success: bool = True
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Default stub executor
# ─────────────────────────────────────────────────────────────────────────────

async def _stub_executor(
    sub_query: SubQuery,
    routing: RoutingDecision,
    optimized_prompt: OptimizedPrompt,
) -> str:
    """Default executor: returns a placeholder response (replace in production)."""
    return (
        f"[STUB] response for sub-query {sub_query.index}: '{sub_query.text[:60]}...' "
        f"via {routing.model_id} (tier={routing.tier.value})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# QueryPipeline
# ─────────────────────────────────────────────────────────────────────────────

class QueryPipeline:
    """End-to-end cost-optimized query processing pipeline.

    Args:
        config:       :class:`~.config.PipelineConfig` controlling all stages.
        execute_fn:   Async callable ``(sub_query, routing, prompt) -> str``
                      invoked per sub-query. Defaults to the stub executor.
        sanitizer:    Optional callable ``(text) -> bool``; returns ``False``
                      to block unsafe input. Defaults to no sanitization.

    Example::

        pipeline = QueryPipeline.build()
        result = pipeline.process("Explain quicksort and give Python code")
        print(result.cost_report.pretty())
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        execute_fn: Optional[Callable] = None,
        sanitizer: Optional[Callable[[str], bool]] = None,
    ):
        self._cfg = config or PipelineConfig._defaults()
        self._execute = execute_fn or _stub_executor
        self._sanitizer = sanitizer

        # ── Component initialisation ─────────────────────────────────────────
        emb_cfg = self._cfg.embedding
        self._engine = EmbeddingEngine.from_config(
            backend=emb_cfg.backend,
            model_name=emb_cfg.model_name,
            dim=emb_cfg.dim,
            cache_size=emb_cfg.cache_size,
        )

        vs_cfg = self._cfg.vector_store
        self._store = VectorStore(self._engine, use_faiss=(vs_cfg.backend == "faiss"))
        self._store.seed_default_topics()

        self._classifier = TopicClassifier(
            self._engine,
            self._store,
            top_k=5,
            threshold=0.15,
        )

        self._decomposer = SmartQueryDecomposer(
            max_sub_queries=self._cfg.max_sub_queries
        )

        self._router = ModelRouter.from_pipeline_config(self._cfg)

        self._optimizer = TokenOptimizer()

        cache_cfg = self._cfg.cache
        self._cache = QueryCache(
            max_memory_entries=cache_cfg.max_memory_entries,
            ttl_seconds=cache_cfg.ttl_seconds,
            persist=cache_cfg.persist,
            persist_path=cache_cfg.persist_path,
        ) if cache_cfg.enabled else None

        logger.info(
            "QueryPipeline ready: embedding=%s vector_store=%s cache=%s",
            emb_cfg.backend, vs_cfg.backend,
            "enabled" if self._cache else "disabled",
        )

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        config_path: Optional[str] = None,
        execute_fn: Optional[Callable] = None,
        sanitizer: Optional[Callable[[str], bool]] = None,
    ) -> "QueryPipeline":
        """Build a `QueryPipeline` from a YAML config file (or defaults)."""
        from pathlib import Path
        cfg = PipelineConfig.from_yaml(Path(config_path) if config_path else None)
        return cls(config=cfg, execute_fn=execute_fn, sanitizer=sanitizer)

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(self, query: str, session_id: str = "anonymous") -> PipelineResult:
        """Synchronous wrapper around :meth:`process_async`."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self.process_async(query, session_id))
                    return future.result()
        except Exception:
            pass
        return asyncio.run(self.process_async(query, session_id))

    async def process_async(self, query: str, session_id: str = "anonymous") -> PipelineResult:
        """Run all pipeline stages and return a :class:`PipelineResult`."""
        t_total = time.perf_counter()
        latency = LatencyBreakdown()

        # ── Stage 0: Sanitization ─────────────────────────────────────────────
        if self._sanitizer and not self._sanitizer(query):
            return self._blocked_result(query, "Input blocked by sanitization filter.")

        # ── Stage 1: Cache lookup ─────────────────────────────────────────────
        t0 = time.perf_counter()
        if self._cache is not None:
            cached = self._cache.lookup(query)
            latency.cache_lookup_ms = (time.perf_counter() - t0) * 1_000
            if cached is not None:
                logger.info("[Pipeline] Cache HIT for session=%s", session_id)
                latency.total_overhead_ms = (time.perf_counter() - t_total) * 1_000
                return PipelineResult(
                    query=query,
                    final_response=cached,
                    classification=None,
                    decomposition=None,
                    router_result=None,
                    sub_query_results=[],
                    cost_report=CostReport(0, 0, 0, 0, 0, 0, 0),
                    latency=latency,
                    from_cache=True,
                )

        # ── Stage 2: Embedding ────────────────────────────────────────────────
        t0 = time.perf_counter()
        query_vec = self._engine.embed(query)
        latency.embedding_ms = (time.perf_counter() - t0) * 1_000

        # ── Stage 3: Vector search ────────────────────────────────────────────
        t0 = time.perf_counter()
        nn_results = self._store.search_topk(query_vec, k=5)
        latency.vector_search_ms = (time.perf_counter() - t0) * 1_000

        # ── Stage 4: Topic classification ─────────────────────────────────────
        t0 = time.perf_counter()
        classification = self._classifier.classify(query)
        latency.classification_ms = (time.perf_counter() - t0) * 1_000

        logger.info(
            "[Pipeline] classified topic='%s' conf=%.2f latency=%.1fms session=%s",
            classification.topic, classification.confidence,
            classification.latency_ms, session_id,
        )

        # ── Stage 5: Decomposition ────────────────────────────────────────────
        t0 = time.perf_counter()
        decomposition = self._decomposer.decompose(query, topic_hint=classification.topic)
        latency.decomposition_ms = (time.perf_counter() - t0) * 1_000

        logger.info(
            "[Pipeline] decomposed into %d sub-queries (total_complexity=%d)",
            len(decomposition.sub_queries), decomposition.total_complexity,
        )

        # ── Stage 6: Model routing ────────────────────────────────────────────
        t0 = time.perf_counter()
        router_result = self._router.route_all(decomposition.sub_queries)
        latency.routing_ms = (time.perf_counter() - t0) * 1_000

        # ── Stages 7+8: Token optimization + Execution ───────────────────────
        t0 = time.perf_counter()
        sub_results = await self._execute_sub_queries(
            decomposition, router_result, classification.context_chunks
        )
        latency.optimization_ms = sum(
            sr.optimized_prompt.optimized_tokens / 1_000 for sr in sub_results
        )
        latency.execution_ms = (time.perf_counter() - t0) * 1_000

        # ── Stage 9: Aggregation ──────────────────────────────────────────────
        t0 = time.perf_counter()
        final_response = self._aggregate(sub_results)
        cost_report = self._build_cost_report(router_result, sub_results)
        latency.aggregation_ms = (time.perf_counter() - t0) * 1_000

        # ── Stage 10: Cache store ─────────────────────────────────────────────
        if self._cache is not None:
            self._cache.store(query, final_response)

        latency.total_overhead_ms = (time.perf_counter() - t_total) * 1_000

        logger.info(
            "[Pipeline] done in %.1fms | savings=%.2f%% ($%.6f) | session=%s",
            latency.total_overhead_ms, cost_report.savings_pct,
            cost_report.total_savings_usd, session_id,
        )

        return PipelineResult(
            query=query,
            final_response=final_response,
            classification=classification,
            decomposition=decomposition,
            router_result=router_result,
            sub_query_results=sub_results,
            cost_report=cost_report,
            latency=latency,
            from_cache=False,
            success=True,
        )

    # ── Sub-query execution ───────────────────────────────────────────────────

    async def _execute_sub_queries(
        self,
        decomp: DecompositionResult,
        router_result: RouterResult,
        context_chunks: List[str],
    ) -> List[SubQueryResult]:
        """Execute sub-queries in topological order (sequential for dependencies, parallel for independents)."""
        decisions_by_idx = {d.sub_query_index: d for d in router_result.decisions}
        results: List[Optional[SubQueryResult]] = [None] * len(decomp.sub_queries)

        # Group by execution wave (each wave can run in parallel)
        waves = self._make_execution_waves(decomp)

        for wave in waves:
            tasks = []
            for idx in wave:
                sq = decomp.sub_queries[idx]
                routing = decisions_by_idx[idx]
                tasks.append(self._execute_one(sq, routing, context_chunks))

            wave_results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, res in zip(wave, wave_results):
                if isinstance(res, Exception):
                    sq = decomp.sub_queries[idx]
                    routing = decisions_by_idx[idx]
                    results[idx] = SubQueryResult(
                        sub_query=sq,
                        routing=routing,
                        optimized_prompt=self._optimizer.optimize(
                            sq.text, sq.text, context_chunks,
                            budget_tokens=routing.token_budget,
                        ),
                        success=False,
                        error=str(res),
                    )
                else:
                    results[idx] = res

        return [r for r in results if r is not None]

    async def _execute_one(
        self,
        sub_query: SubQuery,
        routing: RoutingDecision,
        context_chunks: List[str],
    ) -> SubQueryResult:
        t0 = time.perf_counter()

        opt_prompt = self._optimizer.optimize(
            query=sub_query.text,
            prompt=sub_query.text,
            context_chunks=context_chunks,
            budget_tokens=routing.token_budget,
        )

        try:
            response = await self._execute(sub_query, routing, opt_prompt)
            success = True
            err = None
        except Exception as exc:
            response = None
            success = False
            err = str(exc)
            logger.warning("[Pipeline] sub-query %d failed: %s", sub_query.index, exc)

        return SubQueryResult(
            sub_query=sub_query,
            routing=routing,
            optimized_prompt=opt_prompt,
            response=response,
            success=success,
            error=err,
            latency_ms=(time.perf_counter() - t0) * 1_000,
        )

    # ── Wave builder (parallel groups) ───────────────────────────────────────

    def _make_execution_waves(self, decomp: DecompositionResult) -> List[List[int]]:
        """Build execution waves from topological order + dependency sets."""
        order = decomp.execution_order
        deps = {sq.index: set(sq.depends_on) for sq in decomp.sub_queries}
        completed: set = set()
        waves: List[List[int]] = []

        remaining = list(order)
        while remaining:
            wave = [i for i in remaining if deps[i].issubset(completed)]
            if not wave:
                # Cycle guard — execute rest sequentially
                wave = [remaining[0]]
            waves.append(wave)
            for i in wave:
                completed.add(i)
                remaining.remove(i)

        return waves

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, results: List[SubQueryResult]) -> Dict[str, Any]:
        """Merge sub-query responses into a final response dict."""
        sub_responses = []
        for r in results:
            sub_responses.append({
                "index": r.sub_query.index,
                "query": r.sub_query.text,
                "model": r.routing.model_id,
                "tier": r.routing.tier.value,
                "complexity": r.sub_query.complexity_score,
                "response": r.response,
                "success": r.success,
                "tokens_saved": r.optimized_prompt.tokens_saved,
                "latency_ms": r.latency_ms,
            })
        return {
            "sub_query_responses": sub_responses,
            "n_sub_queries": len(sub_responses),
            "all_success": all(r.success for r in results),
        }

    # ── Cost report ───────────────────────────────────────────────────────────

    def _build_cost_report(
        self,
        router_result: RouterResult,
        sub_results: List[SubQueryResult],
    ) -> CostReport:
        compression_ratios = [
            sr.optimized_prompt.compression_ratio for sr in sub_results
        ]
        return CostReport(
            baseline_cost_usd=router_result.baseline_cost_usd,
            optimized_cost_usd=router_result.total_estimated_cost_usd,
            total_savings_usd=router_result.total_savings_usd,
            savings_pct=router_result.savings_pct,
            cheap_count=router_result.cheap_count,
            mid_count=router_result.mid_count,
            expensive_count=router_result.expensive_count,
            token_compression_ratios=compression_ratios,
        )

    # ── Error helpers ──────────────────────────────────────────────────────────

    def _blocked_result(self, query: str, reason: str) -> PipelineResult:
        return PipelineResult(
            query=query,
            final_response={"error": reason},
            classification=None,
            decomposition=None,
            router_result=None,
            sub_query_results=[],
            cost_report=CostReport(0, 0, 0, 0, 0, 0, 0),
            latency=LatencyBreakdown(),
            success=False,
            error=reason,
        )

    # ── Stats / observability ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline-wide observability stats."""
        stats: Dict[str, Any] = {
            "vector_store": self._store.stats(),
            "embedding_cache": self._engine.cache_stats(),
            "topic_classifier": self._classifier.stats(),
        }
        if self._cache:
            cs = self._cache.stats()
            stats["query_cache"] = {
                "hits": cs.hits,
                "misses": cs.misses,
                "hit_rate": cs.hit_rate,
                "size": cs.size,
            }
        return stats
