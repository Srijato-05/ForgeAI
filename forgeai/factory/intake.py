"""Intake interviewer module (Requirement 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, Field

from forgeai.ingest import DocumentInventory
from forgeai.llm import LLMGateway, call_structured
from forgeai.prompts import load_prompt
from forgeai.schemas import ClarifyingQuestion, RequirementsDoc


class _QuestionsResponse(BaseModel):
    questions: list[ClarifyingQuestion] = Field(default_factory=list)


class _SynthesizedRequirements(BaseModel):
    scope: str
    audience: str
    tone: str
    must_answer: list[str] = Field(default_factory=list)
    must_refuse: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


def save_requirements(req: RequirementsDoc, path: Path) -> None:
    """Persist RequirementsDoc as YAML."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(req.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_requirements(path: Path) -> RequirementsDoc:
    """Load RequirementsDoc from YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RequirementsDoc.model_validate(raw)


def run_intake(
    brief: str,
    inventory: DocumentInventory,
    gateway: LLMGateway,
    ask_user: Callable[[ClarifyingQuestion], str] | None = None,
    unattended: bool = True,
    max_questions: int = 5,
) -> RequirementsDoc:
    """Inspect brief + inventory, ask up to 5 clarifying questions, and write RequirementsDoc."""
    limit = max(1, min(max_questions, 5))
    q_prompt = load_prompt("intake_questions")
    inv_payload = inventory.model_dump(mode="json")

    q_resp = call_structured(
        gateway,
        role="factory",
        messages=[
            {"role": "system", "content": q_prompt},
            {
                "role": "user",
                "content": (
                    f"Brief:\n{brief}\n\n"
                    f"Document Inventory:\n{json.dumps(inv_payload, ensure_ascii=False)}\n\n"
                    f"Return at most {limit} ranked clarifying questions."
                ),
            },
        ],
        schema=_QuestionsResponse,
        node="intake_questions",
    )

    resolved_questions: list[ClarifyingQuestion] = []
    assumed_notes: list[str] = []
    for q in q_resp.questions[:limit]:
        if unattended or ask_user is None:
            resolved = q.model_copy(
                update={"user_answer": q.default_answer, "assumed": True}
            )
            assumed_notes.append(f"Assumed for '{q.question}': {q.default_answer}")
        else:
            ans = (ask_user(q) or "").strip()
            if ans:
                resolved = q.model_copy(update={"user_answer": ans, "assumed": False})
            else:
                resolved = q.model_copy(
                    update={"user_answer": q.default_answer, "assumed": True}
                )
                assumed_notes.append(f"Assumed for '{q.question}': {q.default_answer}")
        resolved_questions.append(resolved)

    req_prompt = load_prompt("intake_requirements")
    qa_summary = [
        {"question": q.question, "answer": q.user_answer, "assumed": q.assumed}
        for q in resolved_questions
    ]
    synth = call_structured(
        gateway,
        role="factory",
        messages=[
            {"role": "system", "content": req_prompt},
            {
                "role": "user",
                "content": (
                    f"Brief:\n{brief}\n\n"
                    f"Document Inventory:\n{json.dumps(inv_payload, ensure_ascii=False)}\n\n"
                    f"Clarifying Q&A:\n{json.dumps(qa_summary, ensure_ascii=False)}"
                ),
            },
        ],
        schema=_SynthesizedRequirements,
        node="intake_requirements",
    )

    data_sources = [d["filename"] for d in inventory.documents] + [
        p.filename for p in inventory.csv_profiles
    ]
    combined_assumptions = list(dict.fromkeys([*synth.assumptions, *assumed_notes]))

    return RequirementsDoc(
        brief=brief,
        scope=synth.scope,
        audience=synth.audience,
        tone=synth.tone,
        must_answer=synth.must_answer,
        must_refuse=synth.must_refuse,
        known_gaps=[],
        assumptions=combined_assumptions,
        data_sources=data_sources,
        clarifying_questions=resolved_questions,
        groundability_checks=[],
    )

