"""
Cost-Optimized Query Processing System for AI Council.

Pipeline (left-to-right):

    User Input
        → QueryCache          (short-circuit on cache hit)
        → EmbeddingEngine     (dense vector representation)
        → VectorStore         (top-k nearest-neighbour search)
        → TopicClassifier     (topic label + context chunks)
        → SmartQueryDecomposer (sub-queries + dependency graph)
        → ModelRouter         (cheap / mid / expensive tier)
        → TokenOptimizer      (prompt compression + RAG cherry-pick)
        → Execution           (parallel, via existing orchestration)
        → ResponseAggregator  (merge + CostReport)
        → QueryCache.store()
        → PipelineResult

Public API::

    from ai_council.query_pipeline import QueryPipeline, PipelineConfig

    pipeline = QueryPipeline.from_config()
    result   = await pipeline.process("Explain quicksort and give Python code")
    print(result.cost_report)
"""

from .config import PipelineConfig
from .embeddings import EmbeddingEngine
from .vector_store import VectorStore, SearchResult
from .topic_classifier import TopicClassifier, ClassificationResult
from .query_decomposer import SmartQueryDecomposer, DecompositionResult, SubQuery
from .model_router import ModelRouter, RoutingDecision, ModelTier
from .token_optimizer import TokenOptimizer, OptimizedPrompt
from .cache import QueryCache, CacheStats
from .pipeline import QueryPipeline, PipelineResult, CostReport

__all__ = [
    # top-level pipeline
    "QueryPipeline",
    "PipelineResult",
    "CostReport",
    "PipelineConfig",
    # individual components (composable)
    "EmbeddingEngine",
    "VectorStore",
    "SearchResult",
    "TopicClassifier",
    "ClassificationResult",
    "SmartQueryDecomposer",
    "DecompositionResult",
    "SubQuery",
    "ModelRouter",
    "RoutingDecision",
    "ModelTier",
    "TokenOptimizer",
    "OptimizedPrompt",
    "QueryCache",
    "CacheStats",
]
