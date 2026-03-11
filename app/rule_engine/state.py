"""Thread-safe state store for temporal rule evaluation."""

from __future__ import annotations

import threading
from typing import Dict


class DurationState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._true_since: Dict[str, float] = {}

    def update(self, key: str, is_true: bool, now: float) -> float:
        with self._lock:
            if is_true:
                self._true_since.setdefault(key, now)
                return now - self._true_since[key]
            self._true_since.pop(key, None)
            return 0.0
