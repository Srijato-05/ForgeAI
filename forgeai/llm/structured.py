"""Pydantic-validated structured output helper with repair retries."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from forgeai.config import ModelRole
from forgeai.llm.gateway import LLMGateway

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class StructuredOutputError(Exception):
    """Raised when an LLM fails to produce schema-valid JSON within max_retries."""


def _strip_code_fences(raw: str) -> str:
    text = _THINK_RE.sub("", raw).strip()
    match = _CODE_FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    if not (text.startswith("{") and text.endswith("}")):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def _unwrap_defs(obj: Any, def_names: set[str]) -> Any:
    """Unwrap single-key `$defs` wrapper dicts sometimes emitted by 3B models."""
    if isinstance(obj, list):
        return [_unwrap_defs(item, def_names) for item in obj]
    if isinstance(obj, dict):
        if len(obj) == 1:
            sole_key = next(iter(obj))
            if sole_key in def_names and isinstance(obj[sole_key], dict):
                return _unwrap_defs(obj[sole_key], def_names)
        return {k: _unwrap_defs(v, def_names) for k, v in obj.items()}
    return obj


def call_structured(
    gateway: LLMGateway,
    role: ModelRole,
    messages: list[dict[str, Any]],
    schema: type[T],
    max_retries: int = 3,
    temperature: float = 0.0,
    node: str = "",
) -> T:
    """Call the LLM and parse/validate output into ``schema``, repairing on errors."""
    raw_schema = schema.model_json_schema()
    def_names = set(raw_schema.get("$defs", {}).keys()) | {schema.__name__}
    schema_json = json.dumps(raw_schema, ensure_ascii=False)
    schema_instruction = (
        "Respond ONLY with a single valid JSON object matching this JSON Schema. "
        "Do not include markdown or extra commentary.\n"
        f"Schema: {schema_json}"
    )

    working_messages = [dict(m) for m in messages]
    if working_messages and working_messages[0].get("role") == "system":
        working_messages[0]["content"] = f"{working_messages[0]['content']}\n\n{schema_instruction}"
    else:
        working_messages.insert(0, {"role": "system", "content": schema_instruction})

    last_error = ""
    for attempt in range(max_retries):
        result = gateway.complete(
            role=role,
            messages=working_messages,
            temperature=temperature,
            json_mode=True,
            node=node,
            use_cache=(attempt == 0),
        )
        cleaned = _strip_code_fences(result.content)
        try:
            parsed = json.loads(cleaned)
            unwrapped = _unwrap_defs(parsed, def_names)
            return schema.model_validate(unwrapped)
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            working_messages.append({"role": "assistant", "content": result.content})
            working_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON output failed validation with this error:\n"
                        f"{last_error}\n\n"
                        "Return ONLY a corrected JSON object matching the schema."
                    ),
                }
            )

    raise StructuredOutputError(
        f"Failed to produce valid {schema.__name__} after {max_retries} attempts. Last error: {last_error}"
    )

