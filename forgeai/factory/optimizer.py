"""Targeted critic-optimizer loop with accept/revert gating and stopping rules (Requirement 6)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from forgeai.config import BudgetConfig
from forgeai.factory.classifier import classify_failures
from forgeai.factory.judge import LayeredJudge
from forgeai.ingest import Chunk, HybridRetriever
from forgeai.llm import LLMGateway, TraceLogger, call_structured
from forgeai.prompts import load_prompt
from forgeai.runtime import AgentRuntime
from forgeai.schemas import (
    AgentDefinition,
    AgentResponse,
    CaseVerdict,
    EvalCase,
    FailureClassification,
    IterationLog,
    PatchOperation,
)
from forgeai.tools import ToolRegistry


class _PromptPatchProposal(BaseModel):
    new_value: str
    rationale: str


def evaluate_agent_on_cases(
    agent_def: AgentDefinition,
    cases: list[EvalCase],
    retriever: HybridRetriever,
    gateway: LLMGateway,
    judge: LayeredJudge,
    chunks_by_id: dict[str, Chunk],
    docs_dir: Path,
    registry: ToolRegistry | None = None,
) -> tuple[dict[str, AgentResponse], dict[str, CaseVerdict], float]:
    """Run ``agent_def`` over ``cases`` and score every response with ``judge``."""
    runtime = AgentRuntime(
        agent_def=agent_def,
        retriever=retriever,
        gateway=gateway,
        registry=registry,
        docs_dir=docs_dir,
    )
    responses: dict[str, AgentResponse] = {}
    verdicts: dict[str, CaseVerdict] = {}
    for case in cases:
        resp = runtime.ask(case.question)
        verdict = judge.evaluate(case, resp, chunks_by_id)
        responses[case.id] = resp
        verdicts[case.id] = verdict

    pass_rate = (
        sum(1 for v in verdicts.values() if v.passed) / len(cases) if cases else 1.0
    )
    return responses, verdicts, round(pass_rate, 4)


def apply_patch(agent_def: AgentDefinition, patch: PatchOperation) -> AgentDefinition:
    """Apply a single targeted field patch to ``agent_def`` and increment its version."""
    section, _, field = patch.target_field.partition(".")
    if section == "retrieval":
        current = getattr(agent_def.retrieval, field)
        if isinstance(current, bool):
            typed_val: object = patch.new_value.strip().lower() in {"true", "1", "yes"}
        elif isinstance(current, int):
            typed_val = int(float(patch.new_value))
        elif isinstance(current, float):
            typed_val = float(patch.new_value)
        else:
            typed_val = patch.new_value
        updated_ret = agent_def.retrieval.model_copy(update={field: typed_val})
        return agent_def.model_copy(
            update={
                "version": agent_def.version + 1,
                "parent_version": agent_def.version,
                "retrieval": updated_ret,
            }
        )

    if section == "system_prompt":
        updated_prompt = agent_def.system_prompt.model_copy(update={field: patch.new_value})
        return agent_def.model_copy(
            update={
                "version": agent_def.version + 1,
                "parent_version": agent_def.version,
                "system_prompt": updated_prompt,
            }
        )

    raise ValueError(f"Unsupported patch target_field: {patch.target_field}")


def _propose_prompt_patch(
    agent_def: AgentDefinition,
    field: str,
    cluster_failures: list[FailureClassification],
    cases_by_id: dict[str, EvalCase],
    gateway: LLMGateway,
) -> PatchOperation:
    current_val = str(getattr(agent_def.system_prompt, field))
    target_ids = [f.case_id for f in cluster_failures]
    diag_lines = [
        f"- [{f.case_id}] ({f.category}): Q='{cases_by_id[f.case_id].question}' -> {f.explanation}"
        for f in cluster_failures[:5]
        if f.case_id in cases_by_id
    ]
    prompt = load_prompt("optimizer_patch")
    proposal = call_structured(
        gateway,
        role="factory",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Target Section: system_prompt.{field}\n"
                    f"Current Text:\n{current_val}\n\n"
                    f"Failing Cases:\n" + "\n".join(diag_lines)
                ),
            },
        ],
        schema=_PromptPatchProposal,
        node="optimizer_patch",
    )
    return PatchOperation(
        target_field=f"system_prompt.{field}",
        old_value=current_val,
        new_value=proposal.new_value.strip() or current_val,
        rationale=proposal.rationale,
        targeted_case_ids=target_ids,
    )


def propose_targeted_patch(
    agent_def: AgentDefinition,
    failures: list[FailureClassification],
    cases_by_id: dict[str, EvalCase],
    gateway: LLMGateway,
    applied_fields: set[str],
    verdicts_by_id: dict[str, CaseVerdict] | None = None,
) -> PatchOperation | None:
    """Pick the largest failure cluster and propose one targeted field patch."""
    if not failures:
        return None

    counts = Counter(f.category for f in failures)
    ordered_cats = [cat for cat, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    for category in ordered_cats:
        cluster = [f for f in failures if f.category == category]
        target_ids = [f.case_id for f in cluster]

        if category == "retrieval_miss":
            ret = agent_def.retrieval
            if "retrieval.top_k" not in applied_fields and ret.top_k < 16:
                new_k = min(16, ret.top_k + 4)
                return PatchOperation(
                    target_field="retrieval.top_k",
                    old_value=str(ret.top_k),
                    new_value=str(new_k),
                    rationale="Increase top_k to recover missed source chunks.",
                    targeted_case_ids=target_ids,
                )
            if "retrieval.min_relevance_score" not in applied_fields and ret.min_relevance_score > 0.05:
                new_min = round(max(0.0, ret.min_relevance_score - 0.1), 2)
                return PatchOperation(
                    target_field="retrieval.min_relevance_score",
                    old_value=str(ret.min_relevance_score),
                    new_value=str(new_min),
                    rationale="Lower min_relevance_score threshold to reduce false misses.",
                    targeted_case_ids=target_ids,
                )
            if "retrieval.hybrid_alpha" not in applied_fields and ret.hybrid_alpha > 0.25:
                new_alpha = round(max(0.2, ret.hybrid_alpha - 0.15), 2)
                return PatchOperation(
                    target_field="retrieval.hybrid_alpha",
                    old_value=str(ret.hybrid_alpha),
                    new_value=str(new_alpha),
                    rationale="Shift hybrid_alpha toward lexical BM25 matching.",
                    targeted_case_ids=target_ids,
                )

        if category == "hallucination":
            for field in ("grounding_rules", "refusal_policy"):
                key = f"system_prompt.{field}"
                if key not in applied_fields:
                    return _propose_prompt_patch(agent_def, field, cluster, cases_by_id, gateway)

        if category == "wrong_refusal":
            for field in ("refusal_policy", "scope"):
                key = f"system_prompt.{field}"
                if key not in applied_fields:
                    return _propose_prompt_patch(agent_def, field, cluster, cases_by_id, gateway)

        if category == "prompt_defect":
            has_tool_err = any(
                verdicts_by_id is not None
                and f.case_id in verdicts_by_id
                and verdicts_by_id[f.case_id].l3_tool_ok is False
                for f in cluster
            )
            candidates = (
                ("tool_guidance", "style", "grounding_rules")
                if has_tool_err
                else ("style", "grounding_rules", "tool_guidance")
            )
            for field in candidates:
                key = f"system_prompt.{field}"
                if key not in applied_fields:
                    return _propose_prompt_patch(agent_def, field, cluster, cases_by_id, gateway)

    return None


def _budget_exhausted(tracer: TraceLogger | None, budget: BudgetConfig) -> bool:
    if tracer is None:
        return False
    if tracer.total_tokens >= budget.max_tokens:
        return True
    if (tracer.total_latency_ms / 1000.0) >= budget.max_latency_seconds:
        return True
    return False


def _save_iterations(logs: list[IterationLog], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [entry.model_dump_json() for entry in logs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_optimization_loop(
    agent_def: AgentDefinition,
    dev_cases: list[EvalCase],
    retriever: HybridRetriever,
    gateway: LLMGateway,
    judge: LayeredJudge,
    chunks_by_id: dict[str, Chunk],
    docs_dir: Path,
    run_dir: Path,
    budget: BudgetConfig | None = None,
    tracer: TraceLogger | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[AgentDefinition, list[IterationLog], dict[str, CaseVerdict], dict[str, AgentResponse]]:
    """Run the bounded targeted critic-optimizer loop over ``dev_cases``."""
    cfg = budget or BudgetConfig()
    run_path = Path(run_dir)
    cases_by_id = {c.id: c for c in dev_cases}

    current_agent = agent_def
    cur_responses, cur_verdicts, cur_score = evaluate_agent_on_cases(
        current_agent, dev_cases, retriever, gateway, judge, chunks_by_id, docs_dir, registry
    )

    logs: list[IterationLog] = []
    applied_fields: set[str] = set()

    def _oos_all_pass(verdicts: dict[str, CaseVerdict]) -> bool:
        return all(verdicts[c.id].passed for c in dev_cases if c.should_refuse and c.id in verdicts)

    if cur_score >= cfg.pass_threshold and _oos_all_pass(cur_verdicts):
        logs.append(
            IterationLog(
                iteration=0,
                patch=None,
                score_before=cur_score,
                score_after=cur_score,
                regressions=0,
                accepted=False,
                stop_reason="pass_threshold_reached",
            )
        )
        current_agent.to_yaml(run_path / "agent_final.yaml")
        _save_iterations(logs, run_path / "iterations.jsonl")
        return current_agent, logs, cur_verdicts, cur_responses

    if _budget_exhausted(tracer, cfg):
        logs.append(
            IterationLog(
                iteration=0,
                patch=None,
                score_before=cur_score,
                score_after=cur_score,
                regressions=0,
                accepted=False,
                stop_reason="budget_exhausted",
            )
        )
        current_agent.to_yaml(run_path / "agent_final.yaml")
        _save_iterations(logs, run_path / "iterations.jsonl")
        return current_agent, logs, cur_verdicts, cur_responses

    for iteration in range(1, cfg.max_iterations + 1):
        if _budget_exhausted(tracer, cfg):
            logs.append(
                IterationLog(
                    iteration=iteration,
                    patch=None,
                    score_before=cur_score,
                    score_after=cur_score,
                    regressions=0,
                    accepted=False,
                    stop_reason="budget_exhausted",
                )
            )
            break

        failures = classify_failures(dev_cases, cur_responses, cur_verdicts)
        if not failures:
            logs.append(
                IterationLog(
                    iteration=iteration,
                    patch=None,
                    score_before=cur_score,
                    score_after=cur_score,
                    regressions=0,
                    accepted=False,
                    stop_reason="pass_threshold_reached",
                )
            )
            break

        patch = propose_targeted_patch(
            current_agent, failures, cases_by_id, gateway, applied_fields, cur_verdicts
        )
        if patch is None:
            logs.append(
                IterationLog(
                    iteration=iteration,
                    patch=None,
                    score_before=cur_score,
                    score_after=cur_score,
                    regressions=0,
                    accepted=False,
                    stop_reason="no_patch_available",
                )
            )
            break

        applied_fields.add(patch.target_field)
        candidate = apply_patch(current_agent, patch)
        new_responses, new_verdicts, new_score = evaluate_agent_on_cases(
            candidate, dev_cases, retriever, gateway, judge, chunks_by_id, docs_dir, registry
        )

        regressions = sum(
            1
            for cid, old_v in cur_verdicts.items()
            if old_v.passed and not new_verdicts[cid].passed
        )
        gain = new_score - cur_score
        accepted = gain > 0 and regressions == 0

        stop_reason: str | None = None
        if accepted:
            current_agent = candidate
            cur_responses = new_responses
            cur_verdicts = new_verdicts
            score_before = cur_score
            cur_score = new_score
            current_agent.to_yaml(run_path / f"agent_v{current_agent.version}.yaml")
            if cur_score >= cfg.pass_threshold and _oos_all_pass(cur_verdicts):
                stop_reason = "pass_threshold_reached"
            elif gain < cfg.min_marginal_gain:
                stop_reason = "marginal_gain_plateau"
            elif iteration == cfg.max_iterations:
                stop_reason = "max_iterations_reached"
        else:
            score_before = cur_score
            if iteration == cfg.max_iterations:
                stop_reason = "max_iterations_reached"

        logs.append(
            IterationLog(
                iteration=iteration,
                patch=patch,
                score_before=score_before,
                score_after=new_score,
                regressions=regressions,
                accepted=accepted,
                stop_reason=stop_reason,
            )
        )

        if stop_reason in {"pass_threshold_reached", "marginal_gain_plateau", "budget_exhausted"}:
            break

    current_agent.to_yaml(run_path / "agent_final.yaml")
    _save_iterations(logs, run_path / "iterations.jsonl")
    return current_agent, logs, cur_verdicts, cur_responses

