"""QueryCache — two-level LRU cache for query results.

Level 1: In-memory ``OrderedDict`` LRU (always available).
Level 2: ``diskcache`` persistence (optional, activated when installed).

Cache keys are SHA-256 hashes of the normalised query text, so the cache
is resilient to minor whitespace/punctuation variations.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _normalise(query: str) -> str:
    """Normalise a query for cache key generation."""
    return " ".join(query.lower().split())


def _make_key(query: str) -> str:
    return hashlib.sha256(_normalise(query).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CachedResponse:
    query_key: str
    result: Any
    stored_at: float = field(default_factory=time.time)
    ttl_seconds: int = 3600
    hit_count: int = 0

    def is_expired(self) -> bool:
        return (time.time() - self.stored_at) > self.ttl_seconds


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


# ─────────────────────────────────────────────────────────────────────────────
# QueryCache
# ─────────────────────────────────────────────────────────────────────────────

class QueryCache:
    """Two-level LRU query cache.

    Args:
        max_memory_entries: Maximum entries in the in-memory LRU.
        ttl_seconds:        Default time-to-live for cached entries.
        persist:            Enable diskcache persistence (requires ``diskcache``).
        persist_path:       Path for the diskcache directory.

    Example::

        cache = QueryCache(max_memory_entries=256, ttl_seconds=3600)
        cache.store("What is quicksort?", {"answer": "..."})
        hit = cache.lookup("What is quicksort?")
        assert hit is not None
    """

    def __init__(
        self,
        max_memory_entries: int = 512,
        ttl_seconds: int = 3600,
        persist: bool = False,
        persist_path: str = "~/.ai_council/cache/query_pipeline",
    ):
        self._max = max_memory_entries
        self._ttl = ttl_seconds
        self._mem: OrderedDict[str, CachedResponse] = OrderedDict()
        self._stats = CacheStats()
        self._disk: Optional[Any] = None

        if persist:
            self._disk = self._init_disk(persist_path)

    # ── Disk cache init ───────────────────────────────────────────────────────

    @staticmethod
    def _init_disk(path: str) -> Optional[Any]:
        try:
            import diskcache  # type: ignore
            import os
            resolved = os.path.expanduser(path)
            dc = diskcache.Cache(resolved, size_limit=256 * 1024 * 1024)
            logger.info("QueryCache: diskcache persisted to '%s'.", resolved)
            return dc
        except ImportError:
            logger.warning("QueryCache: diskcache not installed; memory-only mode.")
            return None
        except Exception as exc:
            logger.warning("QueryCache: failed to init diskcache (%s); memory-only mode.", exc)
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[Any]:
        """Return the cached result for *query*, or ``None`` on a miss/expiry."""
        key = _make_key(query)

        # Level 1: memory
        if key in self._mem:
            entry = self._mem[key]
            if entry.is_expired():
                del self._mem[key]
                self._stats.evictions += 1
            else:
                self._mem.move_to_end(key)
                entry.hit_count += 1
                self._stats.hits += 1
                logger.debug("QueryCache HIT (memory) for key=%s...", key[:12])
                return entry.result

        # Level 2: disk
        if self._disk is not None:
            try:
                data = self._disk.get(key)
                if data is not None:
                    # Promote to memory
                    self._mem_store(key, data, self._ttl)
                    self._stats.hits += 1
                    logger.debug("QueryCache HIT (disk) for key=%s...", key[:12])
                    return data
            except Exception as exc:
                logger.warning("QueryCache disk lookup failed: %s", exc)

        self._stats.misses += 1
        return None

    def store(self, query: str, result: Any, ttl: Optional[int] = None) -> None:
        """Cache *result* under *query* for *ttl* seconds."""
        key = _make_key(query)
        effective_ttl = ttl if ttl is not None else self._ttl

        self._mem_store(key, result, effective_ttl)

        if self._disk is not None:
            try:
                self._disk.set(key, result, expire=effective_ttl)
            except Exception as exc:
                logger.warning("QueryCache disk store failed: %s", exc)

        logger.debug("QueryCache stored key=%s... (ttl=%ds)", key[:12], effective_ttl)

    def invalidate(self, query: str) -> bool:
        """Remove a single entry. Returns True if it existed."""
        key = _make_key(query)
        found = False
        if key in self._mem:
            del self._mem[key]
            found = True
        if self._disk is not None:
            try:
                found = self._disk.delete(key) or found
            except Exception:
                pass
        return found

    def clear(self) -> None:
        """Clear all cached entries (memory + disk)."""
        self._mem.clear()
        if self._disk is not None:
            try:
                self._disk.clear()
            except Exception:
                pass
        logger.info("QueryCache cleared.")

    def stats(self) -> CacheStats:
        self._stats.size = len(self._mem)
        return self._stats

    # ── Internals ─────────────────────────────────────────────────────────────

    def _mem_store(self, key: str, result: Any, ttl: int) -> None:
        if key in self._mem:
            self._mem.move_to_end(key)
        else:
            if len(self._mem) >= self._max:
                # Evict LRU
                evicted_key = next(iter(self._mem))
                del self._mem[evicted_key]
                self._stats.evictions += 1
        self._mem[key] = CachedResponse(
            query_key=key,
            result=result,
            ttl_seconds=ttl,
        )
