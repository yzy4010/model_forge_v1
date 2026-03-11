"""Cross-frame tracking state manager for RuleEngine V3."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict, List, Mapping, MutableMapping, Optional


class TrackStateManager:
    def __init__(self, max_missing_seconds: float = 5.0):
        self.max_missing_seconds = float(max_missing_seconds)
        self.tracks: MutableMapping[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def update(self, person_objects: Optional[List[Mapping[str, Any]]]) -> None:
        now = time.monotonic()
        with self._lock:
            for person in person_objects or []:
                track_id = person.get("track_id")
                if track_id is None:
                    continue
                try:
                    track_key = int(track_id)
                except (TypeError, ValueError):
                    continue

                state = self.tracks.get(track_key)
                if state is None:
                    state = {
                        "track_id": track_key,
                        "first_seen": now,
                        "last_seen": now,
                        "frames": 0,
                        "bbox": [],
                        "objects": {},
                        "roi_tags": tuple(),
                    }
                    self.tracks[track_key] = state

                state["frames"] = int(state.get("frames", 0)) + 1
                state["last_seen"] = now
                state["bbox"] = list(person.get("bbox") or [])
                state["roi_tags"] = tuple(person.get("roi_tags") or ())

                incoming_objects = person.get("objects") or {}
                state["objects"] = {
                    str(alias): bool(items)
                    for alias, items in incoming_objects.items()
                    if str(alias).strip()
                }

            self._cleanup_locked(now)

    def _cleanup_locked(self, now: float) -> None:
        expire_before = now - self.max_missing_seconds
        for track_id in list(self.tracks.keys()):
            if float(self.tracks[track_id].get("last_seen", 0.0)) < expire_before:
                self.tracks.pop(track_id, None)

    def cleanup(self) -> None:
        with self._lock:
            self._cleanup_locked(time.monotonic())

    def get_track(self, track_id: Any) -> Optional[Dict[str, Any]]:
        try:
            track_key = int(track_id)
        except (TypeError, ValueError):
            return None
        with self._lock:
            state = self.tracks.get(track_key)
            return deepcopy(state) if state is not None else None

    def get_duration(self, track_id: Any) -> float:
        try:
            track_key = int(track_id)
        except (TypeError, ValueError):
            return 0.0
        with self._lock:
            state = self.tracks.get(track_key)
            if state is None:
                return 0.0
            return max(0.0, time.monotonic() - float(state.get("first_seen", 0.0)))
