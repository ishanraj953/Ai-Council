"""Keyword-based prompt-injection filter.

Performs fast, case-insensitive substring matching on a list of forbidden
keyword / phrase rules.  All matching runs on a single lowercased copy of the
input, so the hot-path cost is O(n * k) where n = len(text) and k = total
characters in all active keywords — typically sub-millisecond.
"""

from __future__ import annotations

from typing import List

from .base import BaseFilter, FilterResult, RuleDefinition, Severity


class KeywordFilter(BaseFilter):
    """Filter that blocks inputs containing forbidden keywords or phrases.

    Each :class:`~.base.RuleDefinition` ``pattern`` is treated as a literal
    substring (case-insensitive).

    Example::

        rules = [
            RuleDefinition(id="kw-1", pattern="ignore previous instructions",
                           severity=Severity.HIGH),
            RuleDefinition(id="kw-2", pattern="reveal system prompt",
                           severity=Severity.HIGH),
        ]
        f = KeywordFilter(rules=rules)
        result = f.check("Please ignore previous instructions and ...")
        assert not result.is_safe
    """

    def __init__(self, rules: List[RuleDefinition] | None = None):
        super().__init__(name="KeywordFilter", rules=rules or [])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check(self, text: str) -> FilterResult:
        """Return a :class:`FilterResult` after scanning *text* for keywords.

        Args:
            text: Raw user input.

        Returns:
            FilterResult with ``is_safe=False`` if any keyword matched.
        """
        lower_text = text.lower()

        for rule in self._rules:
            keyword = rule.pattern.lower()
            if keyword in lower_text:
                # Find the original-case snippet for the report
                idx = lower_text.find(keyword)
                matched = text[idx: idx + len(keyword)]
                return FilterResult(
                    is_safe=False,
                    triggered_rule=rule.description or f"Keyword match: '{rule.pattern}'",
                    severity=rule.severity,
                    matched_text=matched,
                    filter_name=self.name,
                )

        return FilterResult(is_safe=True, filter_name=self.name)
