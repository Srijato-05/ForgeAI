"""Prompt tournament stretch goal: generate candidate prompts, evaluate, and pick winner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from forgeai.factory.judge import LayeredJudge, judge_pairwise_swapped
from forgeai.factory.optimizer import evaluate_agent_on_cases
from forgeai.ingest import Chunk, HybridRetriever
from forgeai.llm import LLMGateway, call_structured
from forgeai.prompts import load_prompt
from forgeai.schemas import AgentDefinition, EvalCase, PromptSections, RequirementsDoc
from forgeai.tools import ToolRegistry


class _TournamentVariant(BaseModel):
    strategy: str
    system_prompt: PromptSections


class _TournamentBatch(BaseModel):
    candidates: list[_TournamentVariant] = Field(default_factory=list)


class TournamentCandidate(BaseModel):
    """Evaluated prompt candidate in the tournament."""

    label: str
    strategy: str
    agent_def: AgentDefinition
    dev_pass_rate: float
    pairwise_wins: int = 0


class TournamentResult(BaseModel):
    """Outcome of the prompt tournament and pairwise tie-break comparisons."""

    winner_label: str
    winner_agent: AgentDefinition
    candidates: list[TournamentCandidate] = Field(default_factory=list)
    pairwise_matches: list[dict[str, Any]] = Field(default_factory=list)


def run_prompt_tournament(
    base_agent: AgentDefinition,
    requirements: RequirementsDoc,
    dev_cases: list[EvalCase],
    retriever: HybridRetriever,
    gateway: LLMGateway,
    judge: LayeredJudge,
    chunks_by_id: dict[str, Chunk],
    docs_dir: Path,
    n_candidates: int = 2,
    registry: ToolRegistry | None = None,
) -> TournamentResult:
    """Generate candidate system prompts, score them on dev, and pairwise-compare the top two."""
    prompt = load_prompt("prompt_tournament")
    batch = call_structured(
        gateway,
        role="factory",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Requirements:\n{requirements.model_dump_json(indent=2)}\n\n"
                    f"Baseline System Prompt:\n{base_agent.system_prompt.model_dump_json(indent=2)}\n\n"
                    f"Generate {n_candidates} distinct alternative system_prompt candidates."
                ),
            },
        ],
        schema=_TournamentBatch,
        node="prompt_tournament",
    )

    pool: list[tuple[str, str, AgentDefinition]] = [
        ("baseline", "baseline", base_agent),
    ]
    for idx, var in enumerate(batch.candidates[:n_candidates], start=1):
        cand_agent = base_agent.model_copy(
            update={
                "id": f"{base_agent.id}_t{idx}",
                "system_prompt": var.system_prompt,
            }
        )
        pool.append((f"candidate_{idx}", var.strategy, cand_agent))

    eval_slice = dev_cases[:6]
    evaluated: list[TournamentCandidate] = []
    responses_map: dict[str, dict[str, Any]] = {}

    for label, strategy, agent_def in pool:
        resps, _, pass_rate = evaluate_agent_on_cases(
            agent_def, eval_slice, retriever, gateway, judge, chunks_by_id, docs_dir, registry
        )
        evaluated.append(
            TournamentCandidate(
                label=label,
                strategy=strategy,
                agent_def=agent_def,
                dev_pass_rate=pass_rate,
            )
        )
        responses_map[label] = resps

    evaluated.sort(key=lambda c: c.dev_pass_rate, reverse=True)
    pairwise_matches: list[dict[str, Any]] = []

    if len(evaluated) >= 2 and eval_slice:
        top_a, top_b = evaluated[0], evaluated[1]
        pair_cases = [c for c in eval_slice if not c.should_refuse][:3] or eval_slice[:3]
        for case in pair_cases:
            ans_a = responses_map[top_a.label][case.id].answer
            ans_b = responses_map[top_b.label][case.id].answer
            match = judge_pairwise_swapped(
                gateway, case.question, case.key_facts, ans_a, ans_b
            )
            match["case_id"] = case.id
            match["candidate_a"] = top_a.label
            match["candidate_b"] = top_b.label
            pairwise_matches.append(match)
            if match["winner"] == "A":
                top_a.pairwise_wins += 1
            elif match["winner"] == "B":
                top_b.pairwise_wins += 1

    winner = max(evaluated, key=lambda c: (c.dev_pass_rate, c.pairwise_wins))
    winner_agent = winner.agent_def.model_copy(
        update={"id": base_agent.id, "version": base_agent.version}
    )
    return TournamentResult(
        winner_label=winner.label,
        winner_agent=winner_agent,
        candidates=evaluated,
        pairwise_matches=pairwise_matches,
    )

