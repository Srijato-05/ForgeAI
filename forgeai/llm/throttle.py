"""Thread-safe sliding-window rate limiter for RPM and TPM constraints."""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Enforces requests-per-minute (RPM) and tokens-per-minute (TPM) limits.

    Passing ``rpm <= 0`` or ``tpm <= 0`` disables that respective limit.
    """

    def __init__(self, rpm: int = 0, tpm: int = 0, window_seconds: float = 60.0) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._req_timestamps: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._req_timestamps and self._req_timestamps[0] <= cutoff:
            self._req_timestamps.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def acquire(self, estimated_tokens: int = 0) -> None:
        """Block until a request with ``estimated_tokens`` fits within RPM/TPM."""
        if self.rpm <= 0 and self.tpm <= 0:
            return

        tokens_needed = max(0, estimated_tokens)
        while True:
            wait_for = 0.0
            with self._lock:
                now = time.monotonic()
                self._prune(now)

                if self.rpm > 0 and len(self._req_timestamps) >= self.rpm:
                    wait_for = max(
                        wait_for,
                        self._req_timestamps[0] + self.window_seconds - now + 0.01,
                    )

                if self.tpm > 0 and tokens_needed > 0:
                    used_tokens = sum(t for _, t in self._token_events)
                    if used_tokens + tokens_needed > self.tpm and self._token_events:
                        wait_for = max(
                            wait_for,
                            self._token_events[0][0] + self.window_seconds - now + 0.01,
                        )

                if wait_for <= 0.0:
                    self._req_timestamps.append(now)
                    if tokens_needed > 0:
                        self._token_events.append((now, tokens_needed))
                    return

            time.sleep(min(wait_for, 1.0))

    def record_actual_tokens(self, actual_tokens: int) -> None:
        """Record actual token consumption after a response arrives."""
        if self.tpm <= 0 or actual_tokens <= 0:
            return
        with self._lock:
            self._token_events.append((time.monotonic(), actual_tokens))

