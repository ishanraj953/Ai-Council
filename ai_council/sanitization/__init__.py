"""
Sanitization Filter Layer for AI Council.

Provides prompt injection detection and blocking before prompt construction.

Public API:
    SanitizationFilter  – main entry point; chains multiple BaseFilter instances
    BaseFilter          – abstract base for all filter implementations
    KeywordFilter       – exact / substring keyword matching
    RegexFilter         – precompiled regex pattern matching
    FilterResult        – result dataclass returned by every filter
    Severity            – enum for LOW / MEDIUM / HIGH rule severity
"""

from .base import BaseFilter, FilterResult, Severity
from .keyword_filter import KeywordFilter
from .regex_filter import RegexFilter
from .sanitization_filter import SanitizationFilter

__all__ = [
    "SanitizationFilter",
    "BaseFilter",
    "KeywordFilter",
    "RegexFilter",
    "FilterResult",
    "Severity",
]
