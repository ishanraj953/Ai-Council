"""TokenOptimizer — reduce prompt size before expensive model calls.

Strategies applied **in order** (each feeds into the next):

1. **RAG cherry-pick**: score each context chunk by relevance to the query;
   keep only top-N chunks.
2. **Context trimming**: enforce a hard token budget by progressively
   dropping the least-relevant chunks.
3. **Prompt compression**: lightweight rule-based cleanup — strip redundant
   whitespace, boilerplate phrases, repeated content.
4. **Budget enforcement**: hard-truncate at the sentence boundary closest
   to the token budget.

Tokenisation
------------
Uses a simple word-based tokeniser by default (1 word ≈ 1.3 tokens).
Swap in ``tiktoken`` or a HuggingFace tokeniser via the ``tokenizer``
constructor argument.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Simple word-based tokeniser (swappable)
TokenizerFn = Callable[[str], int]


def _word_tokenizer(text: str) -> int:
    """Approximate token count: 1.3 tokens per whitespace-separated word."""
    return int(len(text.split()) * 1.3)


# ─────────────────────────────────────────────────────────────────────────────
# Boilerplate patterns for prompt compression
# ─────────────────────────────────────────────────────────────────────────────

_BOILERPLATE_PATTERNS = [
    (re.compile(r"\bAs an AI (?:language model|assistant),?\s*", re.I), ""),
    (re.compile(r"\bCertainly[!,.]?\s*(?:I(?:'d| would) be happy to[^.]*\.?)?\s*", re.I), ""),
    (re.compile(r"\bOf course[!,.]?\s*", re.I), ""),
    (re.compile(r"\bAbsolutely[!,.]?\s*", re.I), ""),
    (re.compile(r"\bSure[!,.]?\s*", re.I), ""),
    (re.compile(r"\bGreat question[!.]?\s*", re.I), ""),
    (re.compile(r"Please note that\s+", re.I), ""),
    (re.compile(r"\s{2,}", re.DOTALL), " "),         # Collapse whitespace
    (re.compile(r"\n{3,}", re.DOTALL), "\n\n"),       # Max 2 consecutive newlines
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizedPrompt:
    """Result of token optimization for a single prompt/context pair."""
    prompt: str
    original_tokens: int
    optimized_tokens: int
    compression_ratio: float         # optimized / original  (lower = more compressed)
    chunks_kept: int
    chunks_dropped: int
    strategies_applied: List[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.optimized_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# TokenOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TokenOptimizer:
    """Apply a cascade of optimizations to a (prompt, context_chunks) pair.

    Args:
        tokenizer:      Function ``(text) -> int`` counting tokens.
                        Defaults to the word-based approximation.
        max_chunk_drop: Max fraction of chunks to drop during RAG cherry-pick.

    Example::

        opt = TokenOptimizer()
        result = opt.optimize(
            query="What is quicksort?",
            prompt="Explain the quicksort algorithm in detail.",
            context_chunks=["Quicksort uses pivot...", "Merge sort divides..."],
            budget_tokens=512,
        )
        assert result.optimized_tokens <= 512
    """

    def __init__(
        self,
        tokenizer: Optional[TokenizerFn] = None,
        max_chunk_drop: float = 0.7,
    ):
        self._tok: TokenizerFn = tokenizer or _word_tokenizer
        self._max_chunk_drop = max_chunk_drop

    # ── Main API ─────────────────────────────────────────────────────────────

    def optimize(
        self,
        query: str,
        prompt: str,
        context_chunks: Optional[List[str]] = None,
        budget_tokens: int = 2048,
    ) -> OptimizedPrompt:
        """Return an :class:`OptimizedPrompt` within *budget_tokens*.

        Args:
            query:          Original user query (used for relevance scoring).
            prompt:         The constructed LLM prompt (system + user message).
            context_chunks: Optional RAG-retrieved context passages.
            budget_tokens:  Hard token budget for the final output.
        """
        context_chunks = context_chunks or []
        strategies: List[str] = []

        original_prompt = prompt
        original_tokens = self._tok(prompt) + sum(self._tok(c) for c in context_chunks)

        # ── Step 1: RAG cherry-pick ──────────────────────────────────────────
        selected_chunks, dropped = self._rag_cherry_pick(query, context_chunks, budget_tokens)
        if dropped > 0:
            strategies.append(f"rag_cherry_pick(dropped={dropped})")

        # ── Step 2: Prompt compression ───────────────────────────────────────
        compressed_prompt = self._compress_prompt(prompt)
        if compressed_prompt != prompt:
            strategies.append("prompt_compression")
            prompt = compressed_prompt

        # ── Step 3: Assemble and trim ─────────────────────────────────────────
        assembled = self._assemble(prompt, selected_chunks)
        assembled_tokens = self._tok(assembled)

        # ── Step 4: Budget enforcement ────────────────────────────────────────
        if assembled_tokens > budget_tokens:
            assembled = self._hard_trim(assembled, budget_tokens)
            strategies.append(f"hard_trim(budget={budget_tokens})")

        final_tokens = self._tok(assembled)
        ratio = final_tokens / original_tokens if original_tokens > 0 else 1.0

        logger.debug(
            "[TokenOptimizer] %d → %d tokens (%.0f%% of original). strategies=%s",
            original_tokens, final_tokens, ratio * 100, strategies,
        )

        return OptimizedPrompt(
            prompt=assembled,
            original_tokens=original_tokens,
            optimized_tokens=final_tokens,
            compression_ratio=ratio,
            chunks_kept=len(selected_chunks),
            chunks_dropped=dropped,
            strategies_applied=strategies,
        )

    # ── Strategy implementations ──────────────────────────────────────────────

    def _rag_cherry_pick(
        self, query: str, chunks: List[str], budget_tokens: int
    ) -> tuple[List[str], int]:
        """Keep only the most query-relevant chunks that fit in the budget.

        Two pruning passes:
        1. **Relevance gate**: drop chunks with a negative relevance score
           (zero query-term overlap, i.e. completely off-topic).
        2. **Budget gate**: from the remaining chunks, stop adding once the
           reserved token slot is full (70% of *budget_tokens*).

        At least one chunk is always kept (the highest-scored one).
        """
        if not chunks:
            return [], 0

        # Score each chunk by term-overlap with the query
        query_terms = set(re.findall(r"[a-z0-9']+", query.lower()))
        scores = []
        for chunk in chunks:
            chunk_terms = set(re.findall(r"[a-z0-9']+", chunk.lower()))
            overlap = len(query_terms & chunk_terms)
            idf_penalty = len(chunk_terms - query_terms) * 0.1  # penalise irrelevant terms
            scores.append(overlap - idf_penalty)

        # Rank by descending relevance score
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

        reserved = int(budget_tokens * 0.7)  # 70% of budget for context
        selected: List[str] = []
        used_tokens = 0

        for rank, idx in enumerate(ranked):
            chunk_score = scores[idx]
            chunk_tokens = self._tok(chunks[idx])

            # Pass 1 – Relevance gate: skip chunks that are clearly irrelevant
            # (keep at least the top-1 regardless of score)
            if rank > 0 and chunk_score < 0:
                continue

            # Pass 2 – Budget gate
            if used_tokens + chunk_tokens <= reserved:
                selected.append(chunks[idx])
                used_tokens += chunk_tokens
            elif len(selected) == 0:
                # Always keep the top chunk (truncate if needed)
                selected.append(self._hard_trim(chunks[idx], reserved))
                break
            # else: skip this chunk (over budget)

        dropped = len(chunks) - len(selected)
        # Restore original order for coherence
        original_order = sorted(selected, key=lambda c: chunks.index(c))
        return original_order, dropped

    def _compress_prompt(self, prompt: str) -> str:
        """Apply lightweight rule-based prompt compression."""
        result = prompt
        for pattern, replacement in _BOILERPLATE_PATTERNS:
            result = pattern.sub(replacement, result)
        return result.strip()

    def _assemble(self, prompt: str, chunks: List[str]) -> str:
        if not chunks:
            return prompt
        context_block = "\n\n".join(f"[Context]\n{c}" for c in chunks)
        return f"{context_block}\n\n{prompt}"

    def _hard_trim(self, text: str, budget_tokens: int) -> str:
        """Truncate *text* at the sentence boundary nearest to *budget_tokens*."""
        # Split into sentences and greedily fill
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: List[str] = []
        used = 0
        for sent in sentences:
            t = self._tok(sent)
            if used + t > budget_tokens:
                break
            result.append(sent)
            used += t
        trimmed = " ".join(result)
        if not trimmed and text:
            # Safety: hard character-level truncation
            char_limit = budget_tokens * 4   # ~4 chars/token
            trimmed = text[:char_limit]
        return trimmed

    # ── Utility ───────────────────────────────────────────────────────────────

    def token_count(self, text: str) -> int:
        return self._tok(text)
