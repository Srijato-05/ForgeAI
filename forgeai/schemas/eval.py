"""Schemas for generated evaluation cases, agent responses, and layered verdicts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EvalCategory = Literal["single_hop", "multi_doc", "ambiguous", "out_of_scope", "tool_required"]
EvalSplit = Literal["dev", "holdout"]


class EvalCase(BaseModel):
    """A single auto-generated and filtered test case."""

    id: str
    category: EvalCategory
    split: EvalSplit
    question: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    should_refuse: bool
    expected_tool: str | None = None
    expected_value: str | None = None
    rationale: str = ""


class AgentResponse(BaseModel):
    """Structured output returned by the generic agent runtime."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    refused: bool
    refusal_reason: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0


class CaseVerdict(BaseModel):
    """Per-case layered evaluation score combining deterministic, NLI, and LLM checks."""

    case_id: str
    passed: bool
    score: float
    l0_schema_refusal_ok: bool
    l1_citation_hit: bool
    l2_fact_coverage: float
    l3_tool_ok: bool | None = None
    l4_nli_groundedness: float
    l5_llm_correctness: float
    notes: str = ""

