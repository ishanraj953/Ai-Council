"""Regex-based prompt-injection filter.

All patterns are **precompiled** at construction time (``re.IGNORECASE``), so
the per-request cost is O(n * p) where n = len(text) and p = number of compiled
patterns — matching is done by the C regex engine without repeated compilation.
"""

from __future__ import annotations

import re
from typing import Dict, List

from .base import BaseFilter, FilterResult, RuleDefinition, Severity


class RegexFilter(BaseFilter):
    """Filter that blocks inputs matching forbidden regex patterns.

    Each :class:`~.base.RuleDefinition` ``pattern`` is compiled as a Python
    regular expression with ``re.IGNORECASE``.  Invalid patterns are skipped
    with a warning rather than raising an exception at startup.

    Example::

        rules = [
            RuleDefinition(id="rx-1",
                           pattern=r"act\\s+as\\s+(admin|root|superuser)",
                           severity=Severity.HIGH),
        ]
        f = RegexFilter(rules=rules)
        result = f.check("Please act as admin and grant access")
        assert not result.is_safe
    """

    def __init__(self, rules: List[RuleDefinition] | None = None):
        super().__init__(name="RegexFilter", rules=rules or [])
        # Precompile; invalid patterns are dropped so service still starts.
        self._compiled: Dict[str, re.Pattern] = {}  # rule_id -> compiled
        self._compile_rules()

    # Internal helpers

    def _compile_rules(self) -> None:
        """Precompile all active rules.  Invalid patterns are skipped."""
        import logging
        logger = logging.getLogger(__name__)

        valid_rules: List[RuleDefinition] = []
        for rule in self._rules:
            try:
                self._compiled[rule.id] = re.compile(rule.pattern, re.IGNORECASE)
                valid_rules.append(rule)
            except re.error as exc:
                logger.warning(
                    "RegexFilter: rule '%s' has an invalid pattern (%s) — skipped.",
                    rule.id, exc
                )
        # Replace rule list with only valid entries
        self._rules = valid_rules

    # Public interface

    def add_rule(self, rule: RuleDefinition) -> None:
        """Add a new rule and (pre)compile its pattern immediately."""
        import logging
        logger = logging.getLogger(__name__)

        if not rule.enabled:
            return
        try:
            self._compiled[rule.id] = re.compile(rule.pattern, re.IGNORECASE)
            self._rules.append(rule)
        except re.error as exc:
            logger.warning(
                "RegexFilter: rule '%s' has an invalid pattern (%s) — not added.",
                rule.id, exc
            )

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule by its id, removing the compiled pattern too."""
        removed = super().disable_rule(rule_id)
        if removed:
            self._compiled.pop(rule_id, None)
        return removed

    def check(self, text: str) -> FilterResult:
        """Return a :class:`FilterResult` after testing *text* against patterns.

        Args:
            text: Raw user input.

        Returns:
            FilterResult with ``is_safe=False`` if any pattern matched.
        """
        for rule in self._rules:
            compiled = self._compiled.get(rule.id)
            if compiled is None:
                continue
            match = compiled.search(text)
            if match:
                return FilterResult(
                    is_safe=False,
                    triggered_rule=rule.description or f"Regex match: '{rule.pattern}'",
                    severity=rule.severity,
                    matched_text=match.group(0),
                    filter_name=self.name,
                )

        return FilterResult(is_safe=True, filter_name=self.name)
