"""Schemas for failure diagnosis and the targeted critic-optimizer loop."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FailureCategory = Literal["retrieval_miss", "hallucination", "wrong_refusal", "prompt_defect"]


class FailureClassification(BaseModel):
    """Root-cause classification for a single failed evaluation case."""

    case_id: str
    category: FailureCategory
    explanation: str


class PatchOperation(BaseModel):
    """A single targeted modification to an AgentDefinition field."""

    target_field: str
    old_value: str
    new_value: str
    rationale: str
    targeted_case_ids: list[str] = Field(default_factory=list)


class IterationLog(BaseModel):
    """Audit record for one critic-optimizer iteration."""

    iteration: int
    patch: PatchOperation | None = None
    score_before: float
    score_after: float
    regressions: int
    accepted: bool
    stop_reason: str | None = None

