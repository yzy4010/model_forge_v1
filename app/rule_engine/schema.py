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


def normalize_rules(rules: Any) -> List[Rule]:
    if isinstance(rules, Mapping):
        return [Rule(name=str(rules.get("name", "rule_0")), expr=dict(rules.get("expr") or rules))]

    out: List[Rule] = []
    for idx, item in enumerate(rules or []):
        if isinstance(item, Rule):
            out.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"Rule at index {idx} must be object")
        expr = dict(item.get("expr") or item)
        name = str(item.get("name", f"rule_{idx}"))
        out.append(Rule(name=name, expr=expr))
    return out
