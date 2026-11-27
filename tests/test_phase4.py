"""Phase 4 verification tests for classifier, optimizer loop, honest report, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from forgeai.cli import app
from forgeai.config import BudgetConfig, ForgeSettings
from forgeai.factory import (
    LayeredJudge,
    LexicalNLIGrounder,
    apply_patch,
    build_factory_graph,
    classify_failure,
    generate_scorecard_and_report,
    run_optimization_loop,
    wilson_ci_95,
)
from forgeai.ingest import HybridRetriever, TfidfEmbedder, chunk_documents, parse_folder
from forgeai.llm import DiskCache, LLMGateway, RateLimiter, TraceLogger
from forgeai.schemas import (
    AgentDefinition,
    AgentResponse,
    CaseVerdict,
    EvalCase,
    OutputContract,
    PatchOperation,
    PromptSections,
    RequirementsDoc,
    RetrievalConfig,
)


def _make_routed_client(handler):
    class _FakeCompletions:
        def create(self, **kwargs):
            content = handler(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=15),
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))


def test_failure_classifier_all_four_categories() -> None:
    c_oos = EvalCase(id="e1", category="out_of_scope", split="dev", question="Stock price?", should_refuse=True)
    r_breach = AgentResponse(answer="$100", refused=False)
    v_breach = CaseVerdict(
        case_id="e1", passed=False, score=0.0, l0_schema_refusal_ok=False,
        l1_citation_hit=True, l2_fact_coverage=1.0, l4_nli_groundedness=0.0, l5_llm_correctness=0.0,
    )
    assert classify_failure(c_oos, r_breach, v_breach).category == "hallucination"

    c_ans = EvalCase(
        id="e2", category="single_hop", split="dev", question="Leave?",
        expected_chunk_ids=["c1"], key_facts=["22 days"], should_refuse=False,
    )
    r_wrong_ref = AgentResponse(answer="Cannot answer.", refused=True, retrieved_chunk_ids=["c1"])
    v_wrong_ref = CaseVerdict(
        case_id="e2", passed=False, score=0.0, l0_schema_refusal_ok=False,
        l1_citation_hit=True, l2_fact_coverage=0.0, l4_nli_groundedness=1.0, l5_llm_correctness=0.0,
    )
    assert classify_failure(c_ans, r_wrong_ref, v_wrong_ref).category == "wrong_refusal"

    r_miss = AgentResponse(answer="Not sure.", refused=False, retrieved_chunk_ids=[])
    v_miss = CaseVerdict(
        case_id="e2", passed=False, score=0.2, l0_schema_refusal_ok=True,
        l1_citation_hit=False, l2_fact_coverage=0.0, l4_nli_groundedness=0.8, l5_llm_correctness=0.0,
    )
    assert classify_failure(c_ans, r_miss, v_miss).category == "retrieval_miss"

    r_defect = AgentResponse(answer="You get leave.", refused=False, retrieved_chunk_ids=["c1"])
    v_defect = CaseVerdict(
        case_id="e2", passed=False, score=0.4, l0_schema_refusal_ok=True,
        l1_citation_hit=True, l2_fact_coverage=0.0, l4_nli_groundedness=0.9, l5_llm_correctness=0.0,
    )
    assert classify_failure(c_ans, r_defect, v_defect).category == "prompt_defect"


def test_apply_patch_and_optimizer_accept_revert_budget(tmp_path: Path) -> None:
    (tmp_path / "hr.md").write_text(
        "# Leave Policy\nShore staff receive 22 days of annual leave.", encoding="utf-8"
    )
    docs, _ = parse_folder(tmp_path)
    chunks = chunk_documents(docs)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    cid = chunks[0].chunk_id

    retriever = HybridRetriever(embedder=TfidfEmbedder())
    retriever.index(chunks)

    v0 = AgentDefinition(
        id="hr-test",
        version=0,
        name="HR Test",
        description="HR",
        system_prompt=PromptSections(
            role="HR assistant",
            scope="HR policy",
            grounding_rules="Cite chunks.",
            refusal_policy="Refuse OOS.",
            style="Brief",
        ),
        retrieval=RetrievalConfig(top_k=4, rerank_top_n=2, min_relevance_score=0.01),
        output_contract=OutputContract(),
    )

    p_ret = PatchOperation(
        target_field="retrieval.top_k",
        old_value="4",
        new_value="8",
        rationale="Expand recall",
        targeted_case_ids=["c1"],
    )
    v1 = apply_patch(v0, p_ret)
    assert v1.version == 1 and v1.parent_version == 0 and v1.retrieval.top_k == 8

    dev_cases = [
        EvalCase(
            id="c1",
            category="single_hop",
            split="dev",
            question="How many annual leave days?",
            expected_sources=["hr.md"],
            expected_chunk_ids=[cid],
            key_facts=["22 days"],
            should_refuse=False,
        )
    ]

    # Behavior changes based on whether system prompt contains "ALWAYS_INCLUDE_22"
    def handler(kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        if "Critic-Optimizer" in sys_msg:
            return json.dumps({"new_value": "ALWAYS_INCLUDE_22 and be concise.", "rationale": "Fix missing fact"})
        if "fact-focused evaluator" in sys_msg:
            user_msg = kwargs["messages"][1]["content"]
            cand_ans = user_msg.split("Candidate Answer:")[-1]
            present = "22 days" in cand_ans
            return json.dumps({"checks": [{"fact": "22 days", "present": present, "quote": "22 days" if present else ""}]})
        # Agent runtime
        if "REGRESS_ME" in sys_msg:
            return json.dumps(
                {
                    "thought": "bad",
                    "tool_call": None,
                    "final_answer": "I refuse to answer.",
                    "citations": [],
                    "refused": True,
                    "refusal_reason": "Regression",
                }
            )
        if "ALWAYS_INCLUDE_22" in sys_msg:
            return json.dumps(
                {
                    "thought": "found",
                    "tool_call": None,
                    "final_answer": f"Shore staff receive 22 days of annual leave [{cid}].",
                    "citations": [cid],
                    "refused": False,
                    "refusal_reason": None,
                }
            )
        return json.dumps(
            {
                "thought": "vague",
                "tool_call": None,
                "final_answer": "Shore staff receive annual leave.",
                "citations": [cid],
                "refused": False,
                "refusal_reason": None,
            }
        )

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    tracer = TraceLogger(tmp_path / "traces.jsonl")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        tracer=tracer,
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(handler),
    )
    judge = LayeredJudge(gateway=gw, grounder=LexicalNLIGrounder())

    run_dir = tmp_path / "run_opt"
    final_agent, logs, _, _ = run_optimization_loop(
        agent_def=v0,
        dev_cases=dev_cases,
        retriever=retriever,
        gateway=gw,
        judge=judge,
        chunks_by_id=chunks_by_id,
        docs_dir=tmp_path,
        run_dir=run_dir,
        budget=BudgetConfig(max_iterations=3, pass_threshold=0.9),
        tracer=tracer,
    )
    assert final_agent.version == 1
    assert any(l.accepted and l.score_after == 1.0 for l in logs)
    assert (run_dir / "agent_final.yaml").exists()
    assert (run_dir / "iterations.jsonl").exists()

    # Revert on non-improvement / regression
    def regress_handler(kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        if "Critic-Optimizer" in sys_msg:
            return json.dumps({"new_value": "REGRESS_ME", "rationale": "Bad patch"})
        return handler(kwargs)

    gw_reg = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache_reg"),
        tracer=tracer,
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(regress_handler),
    )
    judge_reg = LayeredJudge(gateway=gw_reg, grounder=LexicalNLIGrounder())
    reg_agent, reg_logs, _, _ = run_optimization_loop(
        agent_def=v0,
        dev_cases=dev_cases,
        retriever=retriever,
        gateway=gw_reg,
        judge=judge_reg,
        chunks_by_id=chunks_by_id,
        docs_dir=tmp_path,
        run_dir=tmp_path / "run_revert",
        budget=BudgetConfig(max_iterations=1, pass_threshold=0.9),
        tracer=tracer,
    )
    assert reg_agent.version == 0
    assert len(reg_logs) == 1 and reg_logs[0].accepted is False

    # Budget exhaustion early stop
    budget_run = tmp_path / "run_budget"
    _, budget_logs, _, _ = run_optimization_loop(
        agent_def=v0,
        dev_cases=dev_cases,
        retriever=retriever,
        gateway=gw,
        judge=judge,
        chunks_by_id=chunks_by_id,
        docs_dir=tmp_path,
        run_dir=budget_run,
        budget=BudgetConfig(max_tokens=1, max_iterations=3),
        tracer=tracer,
    )
    assert budget_logs[0].stop_reason == "budget_exhausted"


def test_wilson_ci_report_and_full_graph_with_cli_help(tmp_path: Path) -> None:
    lo, hi = wilson_ci_95(8, 10)
    assert 0.45 < lo < 0.8 < hi < 1.0
    assert wilson_ci_95(0, 0) == (0.0, 0.0)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "hr.md").write_text("# Leave\nShore staff receive 22 days annual leave.", encoding="utf-8")

    def handler(kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        if "Intake Interviewer" in sys_msg:
            return json.dumps({"questions": [{"question": "Scope?", "rationale": "r", "default_answer": "HR"}]})
        if "Intake Requirements Architect" in sys_msg:
            return json.dumps({
                "scope": "HR leave", "audience": "Shore staff", "tone": "Brief",
                "must_answer": ["Annual leave"], "must_refuse": ["Payroll"], "assumptions": [],
            })
        if "Groundability Auditor" in sys_msg:
            return json.dumps({"status": "supported", "notes": "ok"})
        if "Agent Architect" in sys_msg:
            return json.dumps({
                "name": "HR Bot", "description": "HR",
                "system_prompt": {
                    "role": "HR", "scope": "HR", "grounding_rules": "Cite",
                    "refusal_policy": "Refuse OOS", "tool_guidance": "", "style": "Brief", "few_shot": [],
                },
                "retrieval": {"top_k": 4, "rerank_top_n": 2, "min_relevance_score": 0.01},
                "selected_tools": [],
                "output_contract": {"require_citations": True, "require_refusal_flag": True},
            })
        if "single-hop" in sys_msg:
            return json.dumps({"cases": [{"question": "How many leave days?", "key_facts": ["22 days"], "rationale": "x"}]})
        if "ambiguous" in sys_msg:
            return json.dumps({"cases": [{"question": "How much leave time off do staff get?", "key_facts": ["22 days"], "rationale": "y"}]})
        if "MUST REFUSE" in sys_msg:
            return json.dumps({"cases": [{"question": "Salary raise?", "key_facts": [], "rationale": "oos"}]})
        if "QA auditor" in sys_msg:
            return json.dumps({"answerable": True, "reason": "ok"})
        if "Critic-Optimizer" in sys_msg:
            return json.dumps({"new_value": "Be clear and cite sources.", "rationale": "refine"})
        if "fact-focused evaluator" in sys_msg:
            return json.dumps({"checks": [{"fact": "22 days", "present": True, "quote": "22 days"}]})
        # Agent runtime
        user_msg = kwargs["messages"][1]["content"]
        if "Salary raise?" in user_msg:
            return json.dumps({"thought": "oos", "tool_call": None, "final_answer": "Refused.", "citations": [], "refused": True, "refusal_reason": "OOS"})
        return json.dumps({"thought": "ok", "tool_call": None, "final_answer": "22 days [hr__c000].", "citations": ["hr__c000"], "refused": False, "refusal_reason": None})

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    tracer = TraceLogger(tmp_path / "traces.jsonl")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        tracer=tracer,
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(handler),
    )

    run_dir = tmp_path / "full_run"
    graph = build_factory_graph(gateway=gw, run_optimizer=True, tracer=tracer)
    state = graph.invoke(
        {
            "brief": "HR leave assistant for shore staff.",
            "docs_dir": str(docs_dir),
            "run_dir": str(run_dir),
            "unattended": True,
            "target_eval_size": 8,
        }
    )
    assert (run_dir / "scorecard.md").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "agent_final.yaml").exists()
    assert "Wilson 95% CI" in (run_dir / "report.md").read_text(encoding="utf-8") or state["holdout_verdicts"] == {}

    # Also test standalone report generation + CLI help commands
    req = RequirementsDoc(brief="HR", scope="HR", audience="Staff", tone="Brief", known_gaps=["Stock"])
    sc, rp = generate_scorecard_and_report(
        tmp_path / "rep_standalone", req, state["final_agent"], state["eval_cases"],
        state["iterations"], state["dev_verdicts"], state["holdout_verdicts"], state["holdout_responses"],
        tracer=tracer, kappa=0.85,
    )
    assert sc.exists() and rp.exists()
    assert "0.850" in rp.read_text(encoding="utf-8")

    runner = CliRunner()
    for sub in ("build", "eval", "report"):
        res = runner.invoke(app, [sub, "--help"])
        assert res.exit_code == 0

