"""Data schema for the rule engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class Detection:
    alias: str
    bbox: Sequence[float]
    score: float = 0.0
    roi_tags: Sequence[str] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Detection":
        return cls(
            alias=str(data.get("alias", "")),
            bbox=data.get("bbox") or data.get("xyxy") or (),
            score=float(data.get("score", 0.0)),
            roi_tags=tuple(data.get("roi_tags", ()) or ()),
        )


@dataclass(frozen=True)
class Rule:
    name: str
    expr: Dict[str, Any]


@dataclass(frozen=True)
class RuleConfig:
    rule_id: str
    name: str
    enabled: bool
    conditions: Dict[str, Any]


def _normalize_expr(expr: Mapping[str, Any]) -> Dict[str, Any]:
    if "conditions" in expr and isinstance(expr["conditions"], Mapping):
        expr = dict(expr["conditions"])
    if "all" in expr:
        return {"and": [_normalize_expr(item) for item in expr["all"]]}
    if "any" in expr:
        return {"or": [_normalize_expr(item) for item in expr["any"]]}
    if "and" in expr:
        return {"and": [_normalize_expr(item) for item in expr["and"]]}
    if "or" in expr:
        return {"or": [_normalize_expr(item) for item in expr["or"]]}
    if "not" in expr and isinstance(expr["not"], Mapping):
        return {"not": _normalize_expr(expr["not"])}

    leaf = dict(expr)
    if bool(leaf.pop("not", False)):
        return {"not": _normalize_expr(leaf)}
    return leaf


def normalize_rules(rules: Any) -> List[Rule]:
    if isinstance(rules, Mapping):
        expr = _normalize_expr(dict(rules.get("expr") or rules))
        return [Rule(name=str(rules.get("name", "rule_0")), expr=expr)]

    out: List[Rule] = []
    for idx, item in enumerate(rules or []):
        if isinstance(item, Rule):
            out.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"Rule at index {idx} must be object")
        if not item.get("enabled", True):
            continue
        expr = _normalize_expr(dict(item.get("expr") or item.get("conditions") or item))
        name = str(item.get("name", f"rule_{idx}"))
        out.append(Rule(name=name, expr=expr))
    return out
