"""PipelineConfig — configuration dataclass for the cost-optimized query pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default config path relative to repo root
_DEFAULT_CONFIG = Path(__file__).parents[2] / "config" / "query_pipeline.yaml"


@dataclass
class EmbeddingConfig:
    backend: str = "hash"          # "hash" | "sentence_transformers" | "openai"
    model_name: str = "hash-384"   # used by non-hash backends
    dim: int = 384
    cache_size: int = 1024         # max cached embeddings (LRU)


@dataclass
class VectorStoreConfig:
    backend: str = "numpy"         # "numpy" | "faiss"
    persist_path: str = "~/.ai_council/vector_store"
    n_exemplars_per_topic: int = 20


@dataclass
class RoutingTierConfig:
    name: str = "cheap"
    complexity_max: int = 3        # inclusive upper bound (0-10 scale)
    preferred_models: List[str] = field(default_factory=list)
    token_budget: int = 1024
    fallback_tier: Optional[str] = None


@dataclass
class CacheConfig:
    enabled: bool = True
    max_memory_entries: int = 512
    ttl_seconds: int = 3600        # 1 hour
    persist: bool = False          # requires diskcache
    persist_path: str = "~/.ai_council/cache/query_pipeline"


@dataclass
class PipelineConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    routing_tiers: List[RoutingTierConfig] = field(default_factory=list)
    cache: CacheConfig = field(default_factory=CacheConfig)
    max_sub_queries: int = 8
    target_classification_ms: float = 50.0
    target_pipeline_overhead_ms: float = 200.0

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "PipelineConfig":
        """Load config from YAML (or JSON) file; falls back to defaults."""
        resolved = Path(path) if path else _DEFAULT_CONFIG

        if not resolved.exists():
            logger.warning("query_pipeline config '%s' not found — using defaults.", resolved)
            return cls._defaults()

        try:
            raw = resolved.read_text(encoding="utf-8")
            data = _parse(resolved, raw)
        except Exception as exc:
            logger.error("Failed to parse '%s': %s — using defaults.", resolved, exc)
            return cls._defaults()

        # Navigate into the 'query_pipeline' key if present
        section: Dict[str, Any] = data.get("query_pipeline", data)

        emb_data = section.get("embedding", {})
        vs_data = section.get("vector_store", {})
        cache_data = section.get("cache", {})
        tier_data: List[Dict] = section.get("routing_tiers", [])

        tiers = [
            RoutingTierConfig(
                name=t.get("name", "cheap"),
                complexity_max=t.get("complexity_max", 3),
                preferred_models=t.get("preferred_models", []),
                token_budget=t.get("token_budget", 1024),
                fallback_tier=t.get("fallback_tier"),
            )
            for t in tier_data
        ] or cls._defaults().routing_tiers

        return cls(
            embedding=EmbeddingConfig(**{k: emb_data[k] for k in emb_data if hasattr(EmbeddingConfig, k)}),
            vector_store=VectorStoreConfig(**{k: vs_data[k] for k in vs_data if hasattr(VectorStoreConfig, k)}),
            routing_tiers=tiers,
            cache=CacheConfig(**{k: cache_data[k] for k in cache_data if hasattr(CacheConfig, k)}),
            max_sub_queries=section.get("max_sub_queries", 8),
            target_classification_ms=section.get("target_classification_ms", 50.0),
            target_pipeline_overhead_ms=section.get("target_pipeline_overhead_ms", 200.0),
        )

    @staticmethod
    def _defaults() -> "PipelineConfig":
        return PipelineConfig(
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            routing_tiers=[
                RoutingTierConfig(name="cheap",     complexity_max=3,  token_budget=1024,  fallback_tier="mid"),
                RoutingTierConfig(name="mid",       complexity_max=6,  token_budget=2048,  fallback_tier="expensive"),
                RoutingTierConfig(name="expensive", complexity_max=10, token_budget=4096,  fallback_tier=None),
            ],
            cache=CacheConfig(),
        )


def _parse(path: Path, raw: str) -> Dict[str, Any]:
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(raw) or {}
        except ImportError:
            pass  # fall through to JSON
    return json.loads(raw)
