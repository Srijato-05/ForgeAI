"""Schemas for intake requirements and groundability verification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GroundabilityStatus = Literal["supported", "partial", "unsupported"]


class ClarifyingQuestion(BaseModel):
    """A clarifying question asked during intake, with an unattended fallback."""

    question: str
    rationale: str
    default_answer: str
    user_answer: str | None = None
    assumed: bool = False


class TopicGroundability(BaseModel):
    """Result of probing whether a must-answer topic is supported by the corpus."""

    topic: str
    status: GroundabilityStatus
    evidence_chunks: list[str] = Field(default_factory=list)
    notes: str = ""


class RequirementsDoc(BaseModel):
    """Structured requirements produced by Intake Interviewer and Groundability Guard."""

    brief: str
    scope: str
    audience: str
    tone: str
    must_answer: list[str] = Field(default_factory=list)
    must_refuse: list[str] = Field(default_factory=list)
    known_gaps: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    clarifying_questions: list[ClarifyingQuestion] = Field(default_factory=list)
    groundability_checks: list[TopicGroundability] = Field(default_factory=list)

