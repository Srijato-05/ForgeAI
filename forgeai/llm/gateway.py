"""OpenAI-compatible LLM gateway supporting Groq, OpenRouter, SambaNova, and Ollama."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel

from forgeai.config import ForgeSettings, ModelRole, ProviderName
from forgeai.llm.cache import CachedLLMResponse, DiskCache, TraceEntry, TraceLogger
from forgeai.llm.throttle import RateLimiter


class LLMResult(BaseModel):
    """Normalized response returned by LLMGateway."""

    content: str
    model: str
    provider: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return max(1, chars // 4)


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, TimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return False


def _recover_tool_use_failed(exc: Exception) -> str | None:
    """Wrap Groq's 400 tool_use_failed `failed_generation` into _AgentStepOutput JSON."""
    if getattr(exc, "status_code", None) != 400:
        return None
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    err = body.get("error") if isinstance(body.get("error"), dict) else body
    if err.get("code") != "tool_use_failed":
        return None
    raw = err.get("failed_generation")
    if not isinstance(raw, str):
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "name" not in parsed:
        return None
    args = parsed.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return json.dumps(
        {
            "thought": "Invoking tool.",
            "tool_call": {"name": str(parsed["name"]), "arguments": args},
            "final_answer": None,
            "citations": [],
            "refused": False,
            "refusal_reason": None,
        }
    )


class LLMGateway:
    """Unified gateway over OpenAI-compatible endpoints with caching, throttling, and tracing."""

    def __init__(
        self,
        settings: ForgeSettings,
        cache: DiskCache | None = None,
        tracer: TraceLogger | None = None,
        rate_limiter: RateLimiter | None = None,
        client_factory: Callable[[ProviderName, str, str], Any] | None = None,
        max_attempts: int = 4,
        base_backoff_seconds: float = 0.5,
    ) -> None:
        self.settings = settings
        self.cache = cache if cache is not None else DiskCache(settings.cache_dir)
        self.tracer = tracer
        self._client_factory = client_factory
        self._clients: dict[ProviderName, Any] = {}
        self.max_attempts = max_attempts
        self.base_backoff_seconds = base_backoff_seconds

        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        elif settings.forge_provider == "ollama":
            self.rate_limiter = RateLimiter(rpm=0, tpm=0)
        else:
            self.rate_limiter = RateLimiter(rpm=settings.rpm_limit, tpm=settings.tpm_limit)

    def _get_client(self, provider: ProviderName) -> Any:
        if provider not in self._clients:
            base_url = self.settings.get_base_url(provider)
            api_key = self.settings.get_api_key(provider)
            if self._client_factory is not None:
                self._clients[provider] = self._client_factory(provider, base_url, api_key)
            else:
                self._clients[provider] = OpenAI(base_url=base_url, api_key=api_key or "missing-key")
        return self._clients[provider]

    def complete(
        self,
        role: ModelRole,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        json_mode: bool = False,
        node: str = "",
        provider: ProviderName | None = None,
        use_cache: bool = True,
    ) -> LLMResult:
        """Execute a chat completion with caching, rate limiting, and backoff."""
        target_provider = provider or self.settings.forge_provider
        model = self.settings.get_model(role, target_provider)
        cache_key = DiskCache.make_key(target_provider, model, messages, json_mode, temperature)

        if use_cache and self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                result = LLMResult(
                    content=cached.content,
                    model=cached.model,
                    provider=cached.provider,
                    tokens_in=cached.tokens_in,
                    tokens_out=cached.tokens_out,
                    latency_ms=0.0,
                    cached=True,
                )
                if self.tracer is not None:
                    self.tracer.log(
                        TraceEntry(
                            node=node,
                            role=role,
                            provider=target_provider,
                            model=model,
                            prompt_hash=cache_key,
                            tokens_in=cached.tokens_in,
                            tokens_out=cached.tokens_out,
                            latency_ms=0.0,
                            cached=True,
                        )
                    )
                return result

        client = self._get_client(target_provider)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.settings.max_output_tokens > 0:
            kwargs["max_tokens"] = self.settings.max_output_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        estimated_tokens = _estimate_tokens(messages)
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            self.rate_limiter.acquire(estimated_tokens)
            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(**kwargs)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
                tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
                self.rate_limiter.record_actual_tokens(tokens_out)

                if use_cache and self.cache is not None:
                    self.cache.set(
                        cache_key,
                        CachedLLMResponse(
                            content=content,
                            model=model,
                            provider=target_provider,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                        ),
                    )
                if self.tracer is not None:
                    self.tracer.log(
                        TraceEntry(
                            node=node,
                            role=role,
                            provider=target_provider,
                            model=model,
                            prompt_hash=cache_key,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            latency_ms=latency_ms,
                            cached=False,
                        )
                    )

                return LLMResult(
                    content=content,
                    model=model,
                    provider=target_provider,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_ms=latency_ms,
                    cached=False,
                )
            except Exception as exc:
                last_exc = exc
                recovered = _recover_tool_use_failed(exc)
                if recovered is not None:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    if self.tracer is not None:
                        self.tracer.log(
                            TraceEntry(
                                node=node,
                                role=role,
                                provider=target_provider,
                                model=model,
                                prompt_hash=cache_key,
                                tokens_in=0,
                                tokens_out=0,
                                latency_ms=latency_ms,
                                cached=False,
                            )
                        )
                    return LLMResult(
                        content=recovered,
                        model=model,
                        provider=target_provider,
                        tokens_in=0,
                        tokens_out=0,
                        latency_ms=latency_ms,
                        cached=False,
                    )
                if "response_format" in kwargs and getattr(exc, "status_code", None) == 400:
                    kwargs.pop("response_format", None)
                    if attempt + 1 < self.max_attempts:
                        continue
                if getattr(exc, "status_code", None) == 429:
                    msg = str(exc).lower()
                    if "otpm" in msg or "reduce max_tokens" in msg:
                        kwargs["max_tokens"] = max(400, min(int(kwargs.get("max_tokens", 900)), 750) - 100)
                if attempt + 1 >= self.max_attempts or not _is_retryable_error(exc):
                    raise
                time.sleep(self.base_backoff_seconds * (2**attempt))

        assert last_exc is not None
        raise last_exc

