"""Rate-limit tracker for repeated malicious attempts (bonus requirement).

Tracks per-source-key blocked attempts within a sliding time window and
determines whether a repeat offender should be throttled.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Deque


@dataclass
class _WindowedCounter:
    """A deque-backed sliding-window counter of timestamps."""

    window_seconds: float
    _timestamps: Deque[float] = field(default_factory=deque)

    def record(self, ts: float | None = None) -> None:
        if ts is None:
            ts = time.monotonic()
        self._timestamps.append(ts)
        self._evict(ts)

    def count(self, ts: float | None = None) -> int:
        if ts is None:
            ts = time.monotonic()
        self._evict(ts)
        return len(self._timestamps)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


class RateLimitTracker:
    """Track repeated malicious attempts and flag repeat offenders.

    Each unique *key* (e.g. a user-id, session-id, or IP address) gets its
    own independent sliding window.  The tracker is intentionally simple and
    in-memory — swap it for a Redis-backed implementation in production.

    Args:
        max_attempts:    Number of blocked attempts allowed within the window.
        window_seconds:  Rolling window length in seconds.
    """

    def __init__(self, max_attempts: int = 5, window_seconds: float = 60.0):
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._counters: Dict[str, _WindowedCounter] = defaultdict(
            lambda: _WindowedCounter(window_seconds=self._window_seconds)
        )

    def record_attempt(self, key: str) -> None:
        """Record one blocked attempt for *key*."""
        self._counters[key].record()

    def is_rate_limited(self, key: str) -> bool:
        """Return True if *key* has exceeded the allowed attempt count."""
        return self._counters[key].count() >= self._max_attempts

    def attempt_count(self, key: str) -> int:
        """Return the current number of attempts within the window for *key*."""
        return self._counters[key].count()

    def reset(self, key: str) -> None:
        """Clear the attempt history for *key* (e.g. after allowing through)."""
        self._counters.pop(key, None)
