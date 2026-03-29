"""SmartQueryDecomposer — break complex queries into ordered, typed sub-queries.

Responsibilities
----------------
1. Determine whether a query warrants decomposition (based on complexity signals).
2. Split the query into atomic :class:`SubQuery` objects.
3. Score each sub-query with a ``ComplexityScore`` (0–10).
4. Build a lightweight dependency graph (topological order).
5. Expose a deterministic API: same input → same decomposition.

This module wraps the existing ``BasicTaskDecomposer`` for sub-query extraction
and augments the result with complexity scoring and dependency ordering.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class ComplexityLevel(Enum):
    TRIVIAL   = "trivial"    # score 0-1
    SIMPLE    = "simple"     # score 2-3
    MODERATE  = "moderate"   # score 4-5
    COMPLEX   = "complex"    # score 6-7
    VERY_HIGH = "very_high"  # score 8-10


@dataclass
class SubQuery:
    """A single atomic sub-query produced by decomposition."""
    index: int                            # 0-based position in decomposition
    text: str                             # sub-query content
    topic_hint: str = "general_qa"       # classification hint
    complexity_score: int = 3             # 0-10
    complexity_level: ComplexityLevel = ComplexityLevel.SIMPLE
    depends_on: List[int] = field(default_factory=list)   # indices of dependencies
    rationale: str = ""                   # why this sub-query exists


@dataclass
class DecompositionResult:
    """Full decomposition of a user query."""
    original_query: str
    sub_queries: List[SubQuery]
    total_complexity: int                 # sum of sub-query complexity scores
    execution_order: List[int]            # topological sort of indices
    is_simple: bool = False               # True → only one sub-query, no decomposition needed
    decomposition_rationale: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Complexity scoring
# ─────────────────────────────────────────────────────────────────────────────

_HIGH_COMPLEXITY_SIGNALS = [
    r"\banalyze\b", r"\banalyse\b", r"\bcompare\b", r"\bcontrast\b",
    r"\bevaluate\b", r"\bcritically\b", r"\bjustify\b", r"\bargue\b",
    r"\bdesign\b", r"\barchitect\b", r"\boptimize\b", r"\boptimise\b",
    r"\bsynthesize\b", r"\bpredict\b", r"\bforecast\b", r"\bplan\b",
    r"\bstrategy\b", r"\bprove\b", r"\bderive\b",
]

_MEDIUM_COMPLEXITY_SIGNALS = [
    r"\bexplain\b", r"\bdescribe\b", r"\bsummarize\b", r"\bsummarise\b",
    r"\bimplement\b", r"\bwrite\b", r"\bcode\b", r"\bcreate\b",
    r"\bbuild\b", r"\bdevelop\b", r"\bgenerate\b", r"\bsolve\b",
    r"\bcalculate\b",
]

_LOW_COMPLEXITY_SIGNALS = [
    r"\bwhat\s+is\b", r"\bwho\s+is\b", r"\bwhen\s+was\b", r"\bwhere\s+is\b",
    r"\bdefine\b", r"\blist\b", r"\bname\b", r"\bgive\s+me\b",
]

_CONJUNCTION_SPLITS = [
    r",\s*and\s+",            # Oxford comma: ", and"
    r"\s*,\s*and\s+",         # same with optional leading space
    r"\s+and\s+then\s+",
    r"\s+and\s+also\s+",
    r"\s+also\s+",
    r"\s+additionally\s+",
    r"\s+furthermore\s+",
    r"\s+moreover\s+",
    r"\s+finally\s+",
    r"\s*;\s*",
    r"\s+then\s+",
]

_NUMBERED_STEP = re.compile(r"^\s*\d+[\.\)]\s+")
_BULLET_STEP   = re.compile(r"^\s*[-*•]\s+")


def _score_text_complexity(text: str) -> int:
    """Return complexity score 0-10 for a single text fragment."""
    low = text.lower()
    score = 3  # baseline

    hi_hits  = sum(1 for p in _HIGH_COMPLEXITY_SIGNALS   if re.search(p, low))
    md_hits  = sum(1 for p in _MEDIUM_COMPLEXITY_SIGNALS if re.search(p, low))
    lo_hits  = sum(1 for p in _LOW_COMPLEXITY_SIGNALS    if re.search(p, low))

    score += hi_hits * 2
    score += md_hits * 1
    score -= lo_hits * 1

    # Token count bonus
    token_count = len(text.split())
    if token_count > 40:
        score += 2
    elif token_count > 20:
        score += 1
    elif token_count < 8:
        score -= 1

    return max(0, min(10, score))


def _complexity_level(score: int) -> ComplexityLevel:
    if score <= 1:   return ComplexityLevel.TRIVIAL
    if score <= 3:   return ComplexityLevel.SIMPLE
    if score <= 5:   return ComplexityLevel.MODERATE
    if score <= 7:   return ComplexityLevel.COMPLEX
    return ComplexityLevel.VERY_HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Split strategies
# ─────────────────────────────────────────────────────────────────────────────

def _split_by_numbered_items(text: str) -> List[str]:
    parts = re.split(r"\n\s*\d+[\.\)]\s+", text)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]
    return []


def _split_by_bullets(text: str) -> List[str]:
    parts = re.split(r"\n\s*[-*•]\s+", text)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]
    return []


def _split_by_conjunctions(text: str) -> List[str]:
    combined = "|".join(_CONJUNCTION_SPLITS)
    parts = re.split(combined, text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip() and len(p.split()) >= 3]


def _split_by_comma_list(text: str) -> List[str]:
    """Split 'Explain X, compare Y, and give Z' at commas (with optional 'and').

    This is the most common multi-task query format.  We split on:
      - ``', and '``  (Oxford comma + conjunction)
      - ``', '``      (plain comma between substantial fragments)

    A fragment is kept only if it has >= 2 words; trivial slivers are dropped.
    We only engage if the whole text contains at least 2 commas OR a ', and '.
    """
    # Only trigger when we see comma patterns suggesting a list
    has_comma_and = bool(re.search(r",\s*and\s+", text, re.IGNORECASE))
    comma_count = text.count(",")
    if not has_comma_and and comma_count < 2:
        return []

    # Split on ', and ' first to get the last chunk correctly
    step1 = re.split(r",\s*and\s+", text, flags=re.IGNORECASE)
    # Then split each chunk further on remaining commas
    parts: List[str] = []
    for chunk in step1:
        sub = re.split(r",\s+", chunk)
        parts.extend(s.strip() for s in sub if s.strip())

    return [p for p in parts if len(p.split()) >= 2]


def _split_by_commas_with_verbs(text: str) -> List[str]:
    """Split 'X, Y, and Z' style when each fragment contains a verb."""
    parts = re.split(r",\s*(?:and\s+)?", text, flags=re.IGNORECASE)
    verb_parts = [p.strip() for p in parts
                  if p.strip() and re.search(r"\b(is|are|was|were|do|does|did|will|can|could|should|would|explain|write|show|list|give|find|create|analyze|analyse|compare|solve|calculate)\b", p.lower())]
    return verb_parts if len(verb_parts) >= 2 else []


# ─────────────────────────────────────────────────────────────────────────────
# SmartQueryDecomposer
# ─────────────────────────────────────────────────────────────────────────────

class SmartQueryDecomposer:
    """Decompose complex user queries into ordered, complexity-scored sub-queries.

    Args:
        max_sub_queries: Hard cap on the number of sub-queries returned.
        min_fragment_len: Sub-query fragments shorter than this are dropped.

    Example::

        decomposer = SmartQueryDecomposer()
        result = decomposer.decompose(
            "Explain quicksort, compare it with mergesort, and give Python code"
        )
        # result.sub_queries  →  3 SubQuery objects
        # result.execution_order  →  [0, 1, 2]  (sequential)
    """

    def __init__(self, max_sub_queries: int = 8, min_fragment_len: int = 2):
        self._max = max_sub_queries
        self._min_len = min_fragment_len

    # ── Public API ────────────────────────────────────────────────────────────

    def decompose(self, query: str, topic_hint: str = "general_qa") -> DecompositionResult:
        """Decompose *query* and return a :class:`DecompositionResult`."""
        query = query.strip()
        if not query:
            return DecompositionResult(
                original_query=query,
                sub_queries=[],
                total_complexity=0,
                execution_order=[],
                is_simple=True,
            )

        fragments, rationale = self._split(query)

        if not fragments or len(fragments) == 1:
            # Simple query — no decomposition needed
            sq = self._make_sub_query(0, query, topic_hint)
            return DecompositionResult(
                original_query=query,
                sub_queries=[sq],
                total_complexity=sq.complexity_score,
                execution_order=[0],
                is_simple=True,
                decomposition_rationale="Single atomic query — no decomposition required.",
            )

        # Cap and score
        fragments = fragments[:self._max]
        sub_queries = [self._make_sub_query(i, f, topic_hint) for i, f in enumerate(fragments)]

        # Dependency graph: sequential for now (each step depends on previous)
        # where the next sub-query semantically requires the previous
        self._assign_dependencies(sub_queries)

        exec_order = self._topological_sort(sub_queries)
        total_complexity = sum(sq.complexity_score for sq in sub_queries)

        return DecompositionResult(
            original_query=query,
            sub_queries=sub_queries,
            total_complexity=total_complexity,
            execution_order=exec_order,
            is_simple=False,
            decomposition_rationale=rationale,
        )

    # ── Splitting logic ───────────────────────────────────────────────────────

    def _split(self, text: str) -> Tuple[List[str], str]:
        # 1. Numbered items (most reliable)
        parts = _split_by_numbered_items(text)
        if parts:
            return self._filter(parts), "Detected numbered list structure."

        # 2. Bullet points
        parts = _split_by_bullets(text)
        if parts:
            return self._filter(parts), "Detected bullet-point structure."

        # 3. Comma-list: 'X, Y, and Z' (Oxford comma) — most common multi-task format
        parts = _split_by_comma_list(text)
        if len(parts) >= 2:
            return self._filter(parts), "Split on comma-separated clause list."

        # 4. Conjunction splitting
        parts = _split_by_conjunctions(text)
        if len(parts) >= 2:
            return self._filter(parts), "Split on conjunction / sequence words."

        # 5. Comma + verb splitting
        parts = _split_by_commas_with_verbs(text)
        if len(parts) >= 2:
            return self._filter(parts), "Split on comma-separated verb clauses."

        # 6. No decomposition
        return [text], "No clear split points found; treating as atomic."

    def _filter(self, parts: List[str]) -> List[str]:
        return [p for p in parts if len(p.split()) >= self._min_len]

    # ── Sub-query construction ────────────────────────────────────────────────

    def _make_sub_query(self, index: int, text: str, topic_hint: str) -> SubQuery:
        score = _score_text_complexity(text)
        return SubQuery(
            index=index,
            text=text,
            topic_hint=topic_hint,
            complexity_score=score,
            complexity_level=_complexity_level(score),
            depends_on=[],
        )

    # ── Dependency assignment ─────────────────────────────────────────────────

    def _assign_dependencies(self, sub_queries: List[SubQuery]) -> None:
        """Assign sequential dependencies where sub-query n depends on n-1.

        Sub-queries that reference preceding context (via pronouns or
        demonstrative references) get an explicit dependency edge.
        """
        reference_patterns = re.compile(
            r"\b(this|these|that|those|it|them|its|their|above|previous|"
            r"the result|the output|the code|the answer|the analysis)\b",
            re.IGNORECASE,
        )
        for i, sq in enumerate(sub_queries):
            if i == 0:
                continue
            if reference_patterns.search(sq.text):
                sq.depends_on = [i - 1]

    # ── Topological sort ──────────────────────────────────────────────────────

    @staticmethod
    def _topological_sort(sub_queries: List[SubQuery]) -> List[int]:
        """Kahn's algorithm for topological sort of the dependency graph."""
        n = len(sub_queries)
        in_degree = [0] * n
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}

        for sq in sub_queries:
            for dep in sq.depends_on:
                adj[dep].append(sq.index)
                in_degree[sq.index] += 1

        queue = [i for i in range(n) if in_degree[i] == 0]
        order: List[int] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbour in adj[node]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        # If cycle detected (shouldn't happen), fall back to sequential
        if len(order) != n:
            logger.warning("Dependency cycle detected; falling back to sequential order.")
            return list(range(n))

        return order
