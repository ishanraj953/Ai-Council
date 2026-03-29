"""Abstract base classes and shared data types for the sanitization layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Severity(str, Enum):
    """Severity level assigned to a matched rule."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class FilterResult:
    """Encapsulates the outcome of a single filter check.

    Attributes:
        is_safe:        True when no threat was detected.
        triggered_rule: Human-readable description of the rule that matched.
        severity:       Severity level of the detected threat.
        matched_text:   The portion of the input that triggered the rule.
        filter_name:    Name of the filter that produced this result.
    """

    is_safe: bool = True
    triggered_rule: Optional[str] = None
    severity: Optional[Severity] = None
    matched_text: Optional[str] = None
    filter_name: str = ""

    # Structured error payload returned to callers when the input is blocked.
    @property
    def error_response(self) -> dict:
        """Return a structured error dict when the input was blocked."""
        if self.is_safe:
            return {}
        return {
            "error": "Unsafe input detected. Request blocked due to potential prompt injection.",
            "details": {
                "filter": self.filter_name,
                "rule": self.triggered_rule,
                "severity": self.severity.value if self.severity else None,
            },
        }


@dataclass
class RuleDefinition:
    """A single configurable detection rule.

    Attributes:
        id:       Unique identifier for the rule.
        pattern:  The keyword or regex pattern string.
        severity: Severity when this rule fires.
        enabled:  Whether this rule is active.
        description: Human-readable explanation of the rule.
    """

    id: str
    pattern: str
    severity: Severity = Severity.HIGH
    enabled: bool = True
    description: str = ""


class BaseFilter(ABC):
    """Abstract base class that every filter must implement.

    Subclasses should be lightweight; their :meth:`check` method is called
    synchronously in the hot path and must complete in well under 5 ms for
    typical inputs.
    """

    def __init__(self, name: str, rules: List[RuleDefinition]):
        self._name = name
        self._rules: List[RuleDefinition] = [r for r in rules if r.enabled]

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def check(self, text: str) -> FilterResult:
        """Inspect *text* and return a :class:`FilterResult`.

        Args:
            text: The raw user input to inspect.

        Returns:
            FilterResult with ``is_safe=True`` when no threat was detected.
        """

    def add_rule(self, rule: RuleDefinition) -> None:
        """Dynamically add a rule at runtime."""
        if rule.enabled:
            self._rules.append(rule)

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule by its id.  Returns True if the rule was found."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.id != rule_id]
        return len(self._rules) < before
