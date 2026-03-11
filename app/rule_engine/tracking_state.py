"""Cross-frame tracking state manager for RuleEngine V3.

This module is intentionally independent from the rule engine runtime so it can
be used as an optional capability. It keeps per-track temporal/object state that
can be queried by duration/stay/behavior logic in upper layers.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict, List, Mapping, MutableMapping, Optional


class TrackStateManager:
    """Thread-safe manager for per-``track_id`` cross-frame state.

    State shape (per track):
        {
            "track_id": int,
            "first_seen": float,
            "last_seen": float,
            "frames": int,
            "bbox": [x1, y1, x2, y2],
            "objects": {alias: bool},
            "roi": optional
        }

    The manager supports empty inputs and missing ``track_id`` entries. Cleanup
    is based on ``max_missing_seconds`` and can run either explicitly or as part
    of ``update``.
    """

    def __init__(self, max_missing_seconds: float = 5):
        """Initialize tracking state manager.

        Args:
            max_missing_seconds: Expiration threshold for stale tracks.
        """
        self.max_missing_seconds = float(max_missing_seconds)
        self.tracks: MutableMapping[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def update(self, person_objects: Optional[List[Mapping[str, Any]]]) -> None:
        """Update track states from one frame of association results.

        Args:
            person_objects: List of person-like association dictionaries with
                shape similar to::

                    {
                        "track_id": 3,
                        "bbox": [x1, y1, x2, y2],
                        "objects": {
                            "helmet": [...],
                            "vest": [...]
                        }
                    }

                Empty input is allowed; in this case only cleanup is performed.
        """
        now = time.monotonic()
        with self._lock:
            # O(N): process current frame objects once.
            for person in person_objects or []:
                track_id = person.get("track_id")
                if track_id is None:
                    continue

                try:
                    track_key = int(track_id)
                except (TypeError, ValueError):
                    continue

                bbox = list(person.get("bbox") or [])
                if len(bbox) != 4:
                    bbox = []

                incoming_objects = person.get("objects") or {}
                incoming_aliases = {
                    str(alias): bool(items)
                    for alias, items in incoming_objects.items()
                    if str(alias).strip()
                }

                if track_key not in self.tracks:
                    self.tracks[track_key] = {
                        "track_id": track_key,
                        "first_seen": now,
                        "last_seen": now,
                        "frames": 1,
                        "bbox": bbox,
                        "objects": dict(incoming_aliases),
                        "roi": person.get("roi"),
                    }
                    continue

                state = self.tracks[track_key]
                state["frames"] = int(state.get("frames", 0)) + 1
                state["last_seen"] = now
                state["bbox"] = bbox
                if "roi" in person:
                    state["roi"] = person.get("roi")

                objects_state: Dict[str, bool] = state.setdefault("objects", {})
                known_aliases = set(objects_state.keys()) | set(incoming_aliases.keys())
                for alias in known_aliases:
                    objects_state[alias] = incoming_aliases.get(alias, False)

            self._cleanup_locked(now)

    def cleanup(self) -> None:
        """Remove tracks that have not been updated recently."""
        with self._lock:
            self._cleanup_locked(time.monotonic())

    def _cleanup_locked(self, now: float) -> None:
        """Internal cleanup. Caller must hold ``self._lock``."""
        expire_before = now - self.max_missing_seconds
        stale_ids = [
            track_id
            for track_id, state in self.tracks.items()
            if float(state.get("last_seen", 0.0)) < expire_before
        ]
        for track_id in stale_ids:
            self.tracks.pop(track_id, None)

    def get_track(self, track_id: Any) -> Optional[Dict[str, Any]]:
        """Return a copy of one track state, or ``None`` when unavailable."""
        try:
            track_key = int(track_id)
        except (TypeError, ValueError):
            return None

        with self._lock:
            state = self.tracks.get(track_key)
            return deepcopy(state) if state is not None else None

    def get_duration(self, track_id: Any) -> float:
        """Return current existence duration (seconds) for a track.

        Duration is measured as ``now - first_seen``. Returns ``0.0`` when
        track does not exist.
        """
        try:
            track_key = int(track_id)
        except (TypeError, ValueError):
            return 0.0

        with self._lock:
            state = self.tracks.get(track_key)
            if not state:
                return 0.0
            return max(0.0, time.monotonic() - float(state.get("first_seen", 0.0)))
