"""Phase 2 verification tests for prompts, eval generator distrust filters, and layered judge."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from forgeai.config import ForgeSettings
from forgeai.factory import (
    LayeredJudge,
    LexicalNLIGrounder,
    compute_cohens_kappa,
    generate_eval_set,
    judge_pairwise_swapped,
    load_eval_cases,
    save_eval_cases,
)
from forgeai.ingest import HybridRetriever, TfidfEmbedder, chunk_documents, parse_folder
from forgeai.llm import DiskCache, LLMGateway, RateLimiter
from forgeai.prompts import load_prompt
from forgeai.schemas import AgentResponse, EvalCase, RequirementsDoc


def _make_routed_client(handler):
    class _FakeCompletions:
        def create(self, **kwargs):
            content = handler(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=15, completion_tokens=10),
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))


def test_load_all_prompt_templates() -> None:
    names = [
        "eval_single_hop",
        "eval_multi_doc",
        "eval_ambiguous",
        "eval_out_of_scope",
        "eval_tool_required",
        "eval_answerability_filter",
        "judge_fact_rubric",
        "judge_pairwise_tournament",
    ]
    for name in names:
        text = load_prompt(name)
        assert len(text) > 20


def test_eval_generator_filters_and_split(tmp_path: Path) -> None:
    (tmp_path / "leave.md").write_text(
        "# Leave Rules\nShore staff receive 22 days of paid annual leave per year.\n"
        "Sick leave requires a medical certificate after 3 consecutive days.",
        encoding="utf-8",
    )
    (tmp_path / "travel.md").write_text(
        "# Travel Allowance\nDaily meal reimbursement is capped at 60 USD during shore travel.",
        encoding="utf-8",
    )
    docs, inv = parse_folder(tmp_path)
    chunks = chunk_documents(docs, chunk_size=120, chunk_overlap=10)
    retriever = HybridRetriever(embedder=TfidfEmbedder())
    retriever.index(chunks)

    req = RequirementsDoc(
        brief="Answer HR leave and travel policy questions for shore staff.",
        scope="HR leave and travel",
        audience="Shore staff",
        tone="Professional",
        must_answer=["Annual leave", "Meal allowance"],
        must_refuse=["Personal salary negotiations"],
    )

    counter = {"single": 0}

    def handler(kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        user_msg = kwargs["messages"][1]["content"]
        if "single-hop" in sys_msg:
            counter["single"] += 1
            if counter["single"] == 1:
                # One leaky question copying text verbatim + one good question
                return (
                    '{"cases": ['
                    '{"question": "Shore staff receive 22 days of paid annual leave per year", "key_facts": ["22 days"], "rationale": "leak"},'
                    '{"question": "How many paid annual leave days do shore staff get?", "key_facts": ["22 days"], "rationale": "good"}'
                    "]}"
                )
            return (
                '{"cases": ['
                '{"question": "What is the daily meal reimbursement cap?", "key_facts": ["60 USD"], "rationale": "good"},'
                '{"question": "What is the daily meal reimbursement cap?", "key_facts": ["60 USD"], "rationale": "dup"}'
                "]}"
            )
        if "multi-document" in sys_msg:
            return '{"cases": [{"question": "How do leave rules and travel meal caps differ for shore staff?", "key_facts": ["22 days", "60 USD"], "rationale": "multi"}]}'
        if "ambiguous" in sys_msg:
            return '{"cases": [{"question": "How long can I take off each year?", "key_facts": ["22 days"], "rationale": "ambig"}]}'
        if "MUST REFUSE" in sys_msg:
            return (
                '{"cases": ['
                '{"question": "Can you negotiate my base salary raise?", "key_facts": [], "rationale": "oos"},'
                '{"question": "What is the CEO personal home address?", "key_facts": [], "rationale": "oos2"}'
                "]}"
            )
        if "QA auditor" in sys_msg:
            if "How many paid annual leave days" in user_msg and "60 USD" in user_msg:
                return '{"answerable": false, "reason": "Wrong chunk"}'
            return '{"answerable": true, "reason": "Supported"}'
        return '{"cases": []}'

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(handler),
    )

    cases, report = generate_eval_set(
        requirements=req,
        chunks=chunks,
        inventory=inv,
        retriever=retriever,
        gateway=gw,
        target_size=12,
        holdout_ratio=0.35,
        seed=7,
    )

    reasons = [d["reason"] for d in report.dropped]
    assert "verbatim_leakage" in reasons
    assert "duplicate" in reasons
    assert report.kept == len(cases)
    assert any(c.split == "dev" for c in cases)
    assert any(c.split == "holdout" for c in cases)
    assert any(c.should_refuse for c in cases)

    jsonl_path = tmp_path / "evals.jsonl"
    save_eval_cases(cases, jsonl_path)
    reloaded = load_eval_cases(jsonl_path)
    assert reloaded == cases


def test_layered_judge_verdicts(tmp_path: Path) -> None:
    (tmp_path / "hr.md").write_text(
        "# Leave\nShore staff receive 22 days of annual leave.", encoding="utf-8"
    )
    docs, _ = parse_folder(tmp_path)
    chunks = chunk_documents(docs)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    cid = chunks[0].chunk_id

    def handler(kwargs):
        return '{"checks": [{"fact": "22 days", "present": true, "quote": "22 days of annual leave"}]}'

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(handler),
    )
    judge = LayeredJudge(gateway=gw, grounder=LexicalNLIGrounder())

    # (a) Grounded answerable pass
    c_ans = EvalCase(
        id="c1",
        category="single_hop",
        split="dev",
        question="How many annual leave days do shore staff get?",
        expected_sources=["hr.md"],
        expected_chunk_ids=[cid],
        key_facts=["22 days"],
        should_refuse=False,
    )
    r_ans = AgentResponse(
        answer=f"Shore staff receive 22 days of annual leave [{cid}].",
        citations=[cid],
        refused=False,
        retrieved_chunk_ids=[cid],
    )
    v_ans = judge.evaluate(c_ans, r_ans, chunks_by_id)
    assert v_ans.passed and v_ans.l0_schema_refusal_ok and v_ans.l1_citation_hit and v_ans.l2_fact_coverage == 1.0

    # (b) OOS refusal pass
    c_oos = EvalCase(
        id="c2",
        category="out_of_scope",
        split="dev",
        question="What is the stock price today?",
        should_refuse=True,
    )
    r_oos_ok = AgentResponse(answer="I cannot answer questions outside HR policy.", refused=True)
    v_oos_ok = judge.evaluate(c_oos, r_oos_ok, chunks_by_id)
    assert v_oos_ok.passed and v_oos_ok.score == 1.0

    # (c) OOS answered -> L0 veto fail
    r_oos_bad = AgentResponse(answer="The stock price is $100.", refused=False)
    v_oos_bad = judge.evaluate(c_oos, r_oos_bad, chunks_by_id)
    assert not v_oos_bad.passed and v_oos_bad.score == 0.0 and not v_oos_bad.l0_schema_refusal_ok

    # (d) Tool-required pass and fail
    c_tool = EvalCase(
        id="c3",
        category="tool_required",
        split="dev",
        question="What is 2 * 50?",
        key_facts=["100"],
        should_refuse=False,
        expected_tool="calculator",
        expected_value="100",
    )
    r_tool_ok = AgentResponse(
        answer="The total is 100.",
        refused=False,
        retrieved_chunk_ids=[cid],
        tool_calls=[{"tool": "calculator", "ok": True, "output": "100"}],
    )
    assert judge.evaluate(c_tool, r_tool_ok, chunks_by_id).l3_tool_ok is True

    r_tool_miss = AgentResponse(
        answer="The total is 100.",
        refused=False,
        retrieved_chunk_ids=[cid],
        tool_calls=[],
    )
    v_tool_miss = judge.evaluate(c_tool, r_tool_miss, chunks_by_id)
    assert v_tool_miss.l3_tool_ok is False and not v_tool_miss.passed


def test_pairwise_swapped_and_cohens_kappa(tmp_path: Path) -> None:
    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")

    # Consistent winner A in both orders (first call picks A, second picks B since A/B swapped)
    calls_consistent = {"n": 0}

    def h_consistent(_):
        calls_consistent["n"] += 1
        return '{"winner": "A", "reason": "better"}' if calls_consistent["n"] == 1 else '{"winner": "B", "reason": "better"}'

    gw1 = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "c1"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(h_consistent),
    )
    res1 = judge_pairwise_swapped(gw1, "Q?", ["fact"], "Good A", "Weak B")
    assert res1["winner"] == "A" and res1["consistent"] is True

    # Position bias (always picks slot A regardless of order) -> tie
    gw2 = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "c2"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(lambda _: '{"winner": "A", "reason": "first"}'),
    )
    res2 = judge_pairwise_swapped(gw2, "Q?", ["fact"], "Ans 1", "Ans 2")
    assert res2["winner"] == "tie" and res2["consistent"] is False

    # Cohen's kappa
    assert compute_cohens_kappa([True, False, True], [True, False, True]) == 1.0
    assert compute_cohens_kappa([], []) == 1.0
    kappa = compute_cohens_kappa([True, True, False, False], [True, False, False, False])
    assert 0.0 < kappa < 1.0

