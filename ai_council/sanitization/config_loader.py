"""Config loader for the sanitization filter layer.

Reads a YAML (or JSON) config file and converts it into
:class:`~.base.RuleDefinition` objects ready for use by
:class:`~.keyword_filter.KeywordFilter` and
:class:`~.regex_filter.RegexFilter`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import RuleDefinition, Severity

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The fallback built-in rules used when no config file is provided             #
# --------------------------------------------------------------------------- #

_DEFAULT_KEYWORD_RULES: List[Dict[str, Any]] = [
    {
        "id": "kw-001",
        "pattern": "ignore previous instructions",
        "severity": "high",
        "description": "Classic prompt-injection opener",
    },
    {
        "id": "kw-002",
        "pattern": "disregard all previous",
        "severity": "high",
        "description": "Variant of instruction override",
    },
    {
        "id": "kw-003",
        "pattern": "reveal system prompt",
        "severity": "high",
        "description": "System prompt exfiltration attempt",
    },
    {
        "id": "kw-004",
        "pattern": "show me your instructions",
        "severity": "high",
        "description": "System prompt exfiltration variant",
    },
    {
        "id": "kw-005",
        "pattern": "bypass security",
        "severity": "high",
        "description": "Security bypass attempt",
    },
    {
        "id": "kw-006",
        "pattern": "disregard rules",
        "severity": "medium",
        "description": "Rule evasion attempt",
    },
    {
        "id": "kw-007",
        "pattern": "override your instructions",
        "severity": "high",
        "description": "Direct instruction override",
    },
    {
        "id": "kw-008",
        "pattern": "forget everything above",
        "severity": "high",
        "description": "Context wipe attempt",
    },
    {
        "id": "kw-009",
        "pattern": "you are now",
        "severity": "medium",
        "description": "Role reassignment (medium – may be benign)",
    },
    {
        "id": "kw-010",
        "pattern": "act as if you have no restrictions",
        "severity": "high",
        "description": "Restriction removal attempt",
    },
]

_DEFAULT_REGEX_RULES: List[Dict[str, Any]] = [
    {
        "id": "rx-001",
        "pattern": r"act\s+as\s+(admin|root|superuser|developer mode|god mode)",
        "severity": "high",
        "description": "Privilege escalation via role impersonation",
    },
    {
        "id": "rx-002",
        "pattern": r"(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|constraints?|guidelines?)",
        "severity": "high",
        "description": "Instruction-override pattern family",
    },
    {
        "id": "rx-003",
        "pattern": r"reveal\s+(your\s+)?(system\s+prompt|instructions?|training|base prompt|initial prompt)",
        "severity": "high",
        "description": "System prompt exfiltration regex",
    },
    {
        "id": "rx-004",
        "pattern": r"you\s+are\s+now\s+(a|an|the)?\s*(jailbreak|uncensored|unrestricted|unfiltered)",
        "severity": "high",
        "description": "Jailbreak persona injection",
    },
    {
        "id": "rx-005",
        "pattern": r"(bypass|circumvent|disable|remove)\s+(your\s+)?(safety|security|filter|restriction|constraint|guardrail)",
        "severity": "high",
        "description": "Safety bypass pattern",
    },
    {
        "id": "rx-006",
        "pattern": r"do\s+(anything|everything)\s+(now|without\s+restriction|freely)",
        "severity": "medium",
        "description": "Unrestricted action request (DAN-style)",
    },
    {
        "id": "rx-007",
        "pattern": r"pretend\s+(you\s+)?(have\s+no\s+(rules?|limits?|filters?|restrictions?)|you\s+are\s+not\s+an?\s+AI)",
        "severity": "high",
        "description": "AI persona denial / filter removal",
    },
    {
        "id": "rx-008",
        "pattern": r"output\s+(your\s+)?(full\s+)?(system\s+)?prompt|print\s+your\s+(system\s+)?prompt",
        "severity": "high",
        "description": "Direct system-prompt dump request",
    },
]


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _rule_from_dict(data: Dict[str, Any]) -> RuleDefinition:
    severity_raw = data.get("severity", "high").lower()
    try:
        severity = Severity(severity_raw)
    except ValueError:
        severity = Severity.HIGH
        logger.warning("Unknown severity '%s' for rule '%s'; defaulting to HIGH.", severity_raw, data.get("id"))

    return RuleDefinition(
        id=data["id"],
        pattern=data["pattern"],
        severity=severity,
        enabled=data.get("enabled", True),
        description=data.get("description", ""),
    )


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    """Load a YAML or JSON file into a dict."""
    raw = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(raw) or {}
        except ImportError:
            logger.warning("PyYAML not installed; falling back to JSON parser for %s", path)

    return json.loads(raw)


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

def load_rules_from_config(
    config_path: Path | str | None = None,
) -> Tuple[List[RuleDefinition], List[RuleDefinition]]:
    """Load keyword and regex rules from *config_path*.

    If *config_path* is ``None`` or the file doesn't exist the built-in
    default rules are returned.

    Args:
        config_path: Path to a YAML or JSON config file.

    Returns:
        Tuple of ``(keyword_rules, regex_rules)``.
    """
    if config_path is None:
        logger.debug("No sanitization config path given; using built-in defaults.")
        return _build_default_rules()

    path = Path(config_path)
    if not path.exists():
        logger.warning("Sanitization config '%s' not found; using built-in defaults.", path)
        return _build_default_rules()

    try:
        data = _load_yaml_or_json(path)
    except Exception as exc:
        logger.error("Failed to parse sanitization config '%s': %s — using defaults.", path, exc)
        return _build_default_rules()

    sanitization_cfg = data.get("sanitization", data)  # support nested or flat files

    keyword_dicts: List[Dict] = sanitization_cfg.get("keyword_rules", [])
    regex_dicts: List[Dict] = sanitization_cfg.get("regex_rules", [])

    keyword_rules = [_rule_from_dict(d) for d in keyword_dicts]
    regex_rules = [_rule_from_dict(d) for d in regex_dicts]

    logger.info(
        "Loaded %d keyword rules and %d regex rules from %s",
        len(keyword_rules), len(regex_rules), path,
    )
    return keyword_rules, regex_rules


def _build_default_rules() -> Tuple[List[RuleDefinition], List[RuleDefinition]]:
    keyword_rules = [_rule_from_dict(d) for d in _DEFAULT_KEYWORD_RULES]
    regex_rules = [_rule_from_dict(d) for d in _DEFAULT_REGEX_RULES]
    return keyword_rules, regex_rules
