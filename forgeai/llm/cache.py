"""Disk prompt-hash cache and JSONL call tracer."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CachedLLMResponse(BaseModel):
    """Payload persisted in the disk cache."""

    content: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0


class DiskCache:
    """Content-addressed disk cache keyed by request parameters."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def make_key(
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        json_mode: bool,
        temperature: float,
    ) -> str:
        """Compute deterministic SHA-256 hash for an LLM call."""
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "messages": messages,
                "json_mode": json_mode,
                "temperature": temperature,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> CachedLLMResponse | None:
        """Retrieve a cached response if present and valid."""
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            return CachedLLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: str, value: CachedLLMResponse) -> None:
        """Persist a response to disk."""
        path = self._path_for(key)
        with self._lock:
            path.write_text(value.model_dump_json(indent=2), encoding="utf-8")


class TraceEntry(BaseModel):
    """Single JSONL record for an LLM invocation."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    node: str = ""
    role: str
    provider: str
    model: str
    prompt_hash: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool


class TraceLogger:
    """Appends JSONL trace records and tracks cumulative token/latency totals."""

    def __init__(self, trace_path: Path | None = None) -> None:
        self.trace_path = Path(trace_path) if trace_path is not None else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.total_tokens: int = 0
        self.total_latency_ms: float = 0.0
        self.call_count: int = 0

    def log(self, entry: TraceEntry) -> None:
        """Record a call entry in memory and append it to the JSONL file if configured."""
        with self._lock:
            self.call_count += 1
            self.total_tokens += entry.tokens_in + entry.tokens_out
            self.total_latency_ms += entry.latency_ms
            if self.trace_path is not None:
                with self.trace_path.open("a", encoding="utf-8") as fh:
                    fh.write(entry.model_dump_json() + "\n")

