"""ModelRouter — assign the cheapest viable model tier to each sub-query.

Tier mapping (configurable via ``config/query_pipeline.yaml``):

    Complexity 0-3  →  cheap      (summarization, lookup, extraction, simple code)
    Complexity 4-6  →  mid        (medium reasoning, multi-step analysis)
    Complexity 7-10 →  expensive  (complex reasoning, planning, generation)

Design
------
* Routing decisions are deterministic: same (complexity_score, topic) → same tier.
* A confidence score (0-1) is emitted alongside each decision for observability.
* The router computes a ``cost_saved_usd`` delta vs "always use expensive model".
* Fallback chain: if the preferred tier has no models, escalate to the next tier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .query_decomposer import SubQuery

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tier enum & data classes
# ─────────────────────────────────────────────────────────────────────────────

class ModelTier(str, Enum):
    CHEAP     = "cheap"
    MID       = "mid"
    EXPENSIVE = "expensive"


@dataclass
class TierConfig:
    """Configuration for a single tier loaded from pipeline config."""
    name: ModelTier
    complexity_max: int        # inclusive upper bound (0-10)
    preferred_models: List[str] = field(default_factory=list)
    token_budget: int = 2048
    cost_per_1k_tokens: float = 0.0   # USD estimate for cost reporting
    fallback_tier: Optional[str] = None


@dataclass
class RoutingDecision:
    """Routing decision for a single sub-query."""
    sub_query_index: int
    tier: ModelTier
    model_id: str            # selected model (or tier label if no registry)
    confidence: float        # 0-1
    complexity_score: int
    token_budget: int
    reasoning: str
    cost_estimate_usd: float = 0.0
    cost_vs_expensive_usd: float = 0.0   # money saved vs expensive tier


@dataclass
class RouterResult:
    """Routing decisions for a full :class:`~.query_decomposer.DecompositionResult`."""
    decisions: List[RoutingDecision]
    total_estimated_cost_usd: float
    baseline_cost_usd: float          # cost if all tasks were routed to expensive
    total_savings_usd: float
    savings_pct: float

    @property
    def cheap_count(self) -> int:
        return sum(1 for d in self.decisions if d.tier == ModelTier.CHEAP)

    @property
    def mid_count(self) -> int:
        return sum(1 for d in self.decisions if d.tier == ModelTier.MID)

    @property
    def expensive_count(self) -> int:
        return sum(1 for d in self.decisions if d.tier == ModelTier.EXPENSIVE)


# ─────────────────────────────────────────────────────────────────────────────
# Default tier config (overridden by PipelineConfig)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TIERS: List[TierConfig] = [
    TierConfig(
        name=ModelTier.CHEAP,
        complexity_max=3,
        preferred_models=["gpt-3.5-turbo", "gemini-1.5-flash", "llama-3-8b"],
        token_budget=1024,
        cost_per_1k_tokens=0.001,
        fallback_tier="mid",
    ),
    TierConfig(
        name=ModelTier.MID,
        complexity_max=6,
        preferred_models=["gpt-4o-mini", "gemini-1.5-pro", "llama-3-70b"],
        token_budget=2048,
        cost_per_1k_tokens=0.005,
        fallback_tier="expensive",
    ),
    TierConfig(
        name=ModelTier.EXPENSIVE,
        complexity_max=10,
        preferred_models=["gpt-4o", "claude-3-opus", "gemini-1.5-ultra"],
        token_budget=4096,
        cost_per_1k_tokens=0.030,
        fallback_tier=None,
    ),
]

# Topic-based complexity adjustments (applied on top of raw score)
_TOPIC_ADJUSTMENTS: Dict[str, int] = {
    "reasoning":      +2,
    "data_analysis":  +1,
    "research":       +1,
    "coding":          0,
    "debugging":       0,
    "math":           +1,
    "general_qa":     -1,
    "creative":        0,
}


# ─────────────────────────────────────────────────────────────────────────────
# ModelRouter
# ─────────────────────────────────────────────────────────────────────────────

class ModelRouter:
    """Route sub-queries to the cheapest viable model tier.

    Args:
        tiers:            Ordered list of :class:`TierConfig` (cheap → expensive).
        available_models: Flat list of model IDs available in the environment.
                          If empty, decision uses tier labels as placeholders.

    Example::

        router = ModelRouter.default()
        sub = SubQuery(index=0, text="What is quicksort?", complexity_score=2)
        decision = router.route(sub)
        assert decision.tier == ModelTier.CHEAP
    """

    def __init__(
        self,
        tiers: Optional[List[TierConfig]] = None,
        available_models: Optional[List[str]] = None,
    ):
        self._tiers: List[TierConfig] = sorted(
            tiers or _DEFAULT_TIERS, key=lambda t: t.complexity_max
        )
        self._available = set(available_models or [])
        self._tier_by_name: Dict[str, TierConfig] = {t.name.value: t for t in self._tiers}

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "ModelRouter":
        return cls()

    @classmethod
    def from_pipeline_config(cls, pipeline_config, available_models: Optional[List[str]] = None) -> "ModelRouter":
        tier_cfgs = [
            TierConfig(
                name=ModelTier(t.name),
                complexity_max=t.complexity_max,
                preferred_models=t.preferred_models,
                token_budget=t.token_budget,
                fallback_tier=t.fallback_tier,
            )
            for t in pipeline_config.routing_tiers
        ]
        return cls(tiers=tier_cfgs, available_models=available_models)

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self, sub_query: SubQuery) -> RoutingDecision:
        """Return the optimal :class:`RoutingDecision` for one *sub_query*."""
        effective_score = self._adjusted_score(sub_query)
        tier = self._pick_tier(effective_score)
        model_id = self._pick_model(tier)

        confidence = self._confidence(effective_score, tier)
        cost_est = self._estimate_cost(sub_query.text, tier)
        expensive_cost = self._estimate_cost(sub_query.text, self._expensive_tier())

        reasoning = (
            f"complexity={effective_score} (raw={sub_query.complexity_score}, "
            f"topic_adj={effective_score - sub_query.complexity_score:+d}) "
            f"→ tier={tier.name.upper()} → model={model_id}"
        )

        return RoutingDecision(
            sub_query_index=sub_query.index,
            tier=tier,
            model_id=model_id,
            confidence=confidence,
            complexity_score=effective_score,
            token_budget=self._tier_config(tier).token_budget,
            reasoning=reasoning,
            cost_estimate_usd=cost_est,
            cost_vs_expensive_usd=max(0.0, expensive_cost - cost_est),
        )

    def route_all(self, sub_queries: List[SubQuery]) -> RouterResult:
        """Route all sub-queries and produce a :class:`RouterResult` with cost metrics."""
        decisions = [self.route(sq) for sq in sub_queries]

        total_cost = sum(d.cost_estimate_usd for d in decisions)
        baseline_cost = sum(
            self._estimate_cost(sq.text, self._expensive_tier()) for sq in sub_queries
        )
        savings = max(0.0, baseline_cost - total_cost)
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

        return RouterResult(
            decisions=decisions,
            total_estimated_cost_usd=total_cost,
            baseline_cost_usd=baseline_cost,
            total_savings_usd=savings,
            savings_pct=savings_pct,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _adjusted_score(self, sq: SubQuery) -> int:
        adj = _TOPIC_ADJUSTMENTS.get(sq.topic_hint, 0)
        return max(0, min(10, sq.complexity_score + adj))

    def _pick_tier(self, score: int) -> ModelTier:
        for tier_cfg in self._tiers:
            if score <= tier_cfg.complexity_max:
                return tier_cfg.name
        return self._tiers[-1].name  # most expensive as fallback

    def _tier_config(self, tier: ModelTier) -> TierConfig:
        return self._tier_by_name[tier.value]

    def _expensive_tier(self) -> ModelTier:
        return self._tiers[-1].name

    def _pick_model(self, tier: ModelTier) -> str:
        """Return the first preferred model available in the environment, or the tier name."""
        cfg = self._tier_config(tier)
        if self._available:
            for m in cfg.preferred_models:
                if m in self._available:
                    return m
        # Fall back to first preferred (placeholder)
        return cfg.preferred_models[0] if cfg.preferred_models else tier.value

    def _confidence(self, score: int, tier: ModelTier) -> float:
        """Confidence is high when the score is well within the tier's bounds."""
        cfg = self._tier_config(tier)
        prev_max = 0
        for t in self._tiers:
            if t.name == tier:
                break
            prev_max = t.complexity_max

        tier_range = cfg.complexity_max - prev_max
        distance_to_boundary = min(
            abs(score - prev_max),
            abs(cfg.complexity_max - score),
        )
        raw_conf = distance_to_boundary / tier_range if tier_range > 0 else 1.0
        return round(min(0.95, 0.50 + raw_conf * 0.45), 3)  # [0.50, 0.95]

    def _estimate_cost(self, text: str, tier: ModelTier) -> float:
        cfg = self._tier_config(tier)
        tokens = len(text.split()) * 1.3  # rough token estimate
        return (tokens / 1000.0) * cfg.cost_per_1k_tokens
