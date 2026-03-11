"""Rule expression compiler and evaluator."""

from __future__ import annotations

from typing import Any, Mapping

from app.rule_engine.conditions import alias_condition, count_condition, duration_condition, roi_condition
from app.rule_engine.operators import op_and, op_not, op_or


class ExpressionCompiler:
    def __init__(self, duration_state):
        self._duration_state = duration_state

    def compile(self, expr: Mapping[str, Any]):
        if "and" in expr:
            return op_and(self.compile(item) for item in expr["and"])
        if "or" in expr:
            return op_or(self.compile(item) for item in expr["or"])
        if "not" in expr:
            return op_not(self.compile(expr["not"]))
        if "alias" in expr:
            return alias_condition(expr["alias"])
        if "roi" in expr:
            return roi_condition(expr["roi"])
        if "count" in expr:
            return count_condition(expr["count"])
        if "duration" in expr:
            return duration_condition(expr["duration"], self.compile, self._duration_state)
        raise ValueError(f"Unsupported expression: {expr}")
