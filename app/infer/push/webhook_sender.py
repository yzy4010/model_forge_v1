from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger("model_forge.infer.push")


class WebhookSender:
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_ms: int = 2000,
        max_queue: int = 256,
    ) -> None:
        if max_queue <= 0:
            raise ValueError("max_queue must be positive")
        self._url = url
        self._headers = {"Content-Type": "application/json"}
        if headers:
            self._headers.update(headers)
        self._timeout_s = max(timeout_ms, 1) / 1000.0
        self._queue: Deque[str] = deque()
        self._max_queue = max_queue
        self._dropped_count = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def enqueue(self, event_dict: Dict[str, Any]) -> None:
        payload = json.dumps(event_dict, ensure_ascii=False, separators=(",", ":"))
        with self._condition:
            if len(self._queue) >= self._max_queue:
                self._queue.popleft()
                self._dropped_count += 1
                logger.warning(
                    "WebhookSender queue full; dropped oldest event. dropped_count=%s",
                    self._dropped_count,
                )
            self._queue.append(payload)
            self._condition.notify()

    def stop(self, timeout_s: Optional[float] = None) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=timeout_s)

    def _send_loop(self) -> None:
        while not self._stop_event.is_set():
            payload = self._next_payload()
            if payload is None:
                continue
            self._post_payload(payload)

    def _next_payload(self) -> Optional[str]:
        with self._condition:
            while not self._queue and not self._stop_event.is_set():
                self._condition.wait(timeout=0.5)
            if self._stop_event.is_set():
                return None
            return self._queue.popleft() if self._queue else None

    def _post_payload(self, payload: str) -> None:
        request = urllib.request.Request(
            self._url,
            data=payload.encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                status_code = response.getcode()
                if status_code == 200:
                    logger.info("WebhookSender delivered event (status=%s)", status_code)
                else:
                    logger.warning(
                        "WebhookSender delivered non-200 status=%s", status_code
                    )
        except urllib.error.URLError as exc:
            logger.warning("WebhookSender failed to deliver event: %s", exc)
        except Exception:
            logger.exception("WebhookSender unexpected failure during delivery")
            time.sleep(0.1)
