"""Logical operators for expression evaluation."""

from __future__ import annotations

from typing import Callable, Iterable


Predicate = Callable[[dict], bool]


def op_and(children: Iterable[Predicate]) -> Predicate:
    children = tuple(children)
    return lambda ctx: all(child(ctx) for child in children)


def op_or(children: Iterable[Predicate]) -> Predicate:
    children = tuple(children)
    return lambda ctx: any(child(ctx) for child in children)


def op_not(child: Predicate) -> Predicate:
    return lambda ctx: not child(ctx)
