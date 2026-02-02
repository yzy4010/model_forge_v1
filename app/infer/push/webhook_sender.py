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
SLOW_REQUEST_MS = 500


class WebhookSender:
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_ms: int = 5000,
        max_queue: int = 2000,
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
        self._sent_count = 0
        self._failed_count = 0
        self._last_error: Optional[str] = None
        self._last_latency_ms: Optional[int] = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def sent_count(self) -> int:
        with self._lock:
            return self._sent_count

    @property
    def failed_count(self) -> int:
        with self._lock:
            return self._failed_count

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    @property
    def last_latency_ms(self) -> Optional[int]:
        with self._lock:
            return self._last_latency_ms

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
        t0 = time.time()
        request = urllib.request.Request(
            self._url,
            data=payload.encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                status_code = response.getcode()
                lat_ms = int((time.time() - t0) * 1000)
                if 200 <= status_code < 300:
                    with self._lock:
                        self._sent_count += 1
                        self._last_error = None
                        self._last_latency_ms = lat_ms
                    logger.info("WebhookSender delivered event (status=%s)", status_code)
                else:
                    with self._lock:
                        self._failed_count += 1
                        self._last_error = f"status={status_code}"
                        self._last_latency_ms = lat_ms
                    logger.warning(
                        "WebhookSender delivered non-2xx status=%s (lat_ms=%s)",
                        status_code,
                        lat_ms,
                    )
                if lat_ms >= SLOW_REQUEST_MS:
                    logger.warning(
                        "WebhookSender slow delivery (lat_ms=%s, status=%s)",
                        lat_ms,
                        status_code,
                    )
        except urllib.error.URLError as exc:
            lat_ms = int((time.time() - t0) * 1000)
            with self._lock:
                self._failed_count += 1
                self._last_error = repr(exc)
                self._last_latency_ms = lat_ms
                queue_len = len(self._queue)
                dropped_count = self._dropped_count
                failed_count = self._failed_count
            logger.warning(
                "WebhookSender failed (err=%s, lat_ms=%s, queue=%s, dropped=%s, failed=%s, timeout_s=%.2f)",
                type(exc).__name__,
                lat_ms,
                queue_len,
                dropped_count,
                failed_count,
                self._timeout_s,
            )
            time.sleep(0.05)
        except Exception as exc:
            lat_ms = int((time.time() - t0) * 1000)
            with self._lock:
                self._failed_count += 1
                self._last_error = repr(exc)
                self._last_latency_ms = lat_ms
                queue_len = len(self._queue)
                dropped_count = self._dropped_count
                failed_count = self._failed_count
            logger.exception(
                "WebhookSender unexpected failure during delivery (lat_ms=%s, queue=%s, dropped=%s, failed=%s)",
                lat_ms,
                queue_len,
                dropped_count,
                failed_count,
            )
            time.sleep(0.1)
