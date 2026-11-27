"""Phase 6 verification tests for stretch goals: tournament, regression, baseline, and Streamlit UI."""

from __future__ import annotations

import json
import py_compile
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from forgeai.cli import app
from forgeai.config import ForgeSettings
from forgeai.factory import (
    LayeredJudge,
    LexicalNLIGrounder,
    compare_factory_vs_human,
    generate_scorecard_and_report,
    run_prompt_tournament,
    run_regression,
    save_eval_cases,
    save_requirements,
)
from forgeai.ingest import HybridRetriever, TfidfEmbedder, chunk_documents, parse_folder
from forgeai.llm import DiskCache, LLMGateway, RateLimiter
from forgeai.schemas import (
    AgentDefinition,
    EvalCase,
    OutputContract,
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


def test_prompt_tournament_and_report_rendering(tmp_path: Path) -> None:
    (tmp_path / "hr.md").write_text("# Leave\nShore staff receive 22 days annual leave.", encoding="utf-8")
    docs, _ = parse_folder(tmp_path)
    chunks = chunk_documents(docs)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    cid = chunks[0].chunk_id

    retriever = HybridRetriever(embedder=TfidfEmbedder())
    retriever.index(chunks)

    base_agent = AgentDefinition(
        id="hr-base",
        version=0,
        name="HR Base",
        description="HR",
        system_prompt=PromptSections(
            role="HR assistant",
            scope="HR",
            grounding_rules="Cite chunks.",
            refusal_policy="Refuse OOS.",
            style="Plain",
        ),
        retrieval=RetrievalConfig(min_relevance_score=0.01),
        output_contract=OutputContract(),
    )
    req = RequirementsDoc(brief="HR", scope="HR", audience="Staff", tone="Brief")
    cases = [
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

    def handler(kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        if "Prompt Strategist" in sys_msg:
            return json.dumps(
                {
                    "candidates": [
                        {
                            "strategy": "strict_terse",
                            "system_prompt": {
                                "role": "HR assistant",
                                "scope": "HR",
                                "grounding_rules": "STRICT_WIN cite exact numbers.",
                                "refusal_policy": "Refuse OOS.",
                                "tool_guidance": "",
                                "style": "Terse",
                                "few_shot": [],
                            },
                        },
                        {
                            "strategy": "verbose_mode",
                            "system_prompt": {
                                "role": "HR assistant",
                                "scope": "HR",
                                "grounding_rules": "Cite.",
                                "refusal_policy": "Refuse OOS.",
                                "tool_guidance": "",
                                "style": "Verbose",
                                "few_shot": [],
                            },
                        },
                    ]
                }
            )
        if "pairwise evaluator" in sys_msg:
            user_msg = kwargs["messages"][1]["content"]
            resp_a = user_msg.split("Response A:")[1].split("Response B:")[0]
            return json.dumps({"winner": "A" if "22 days" in resp_a else "B", "reason": "exact"})
        if "fact-focused evaluator" in sys_msg:
            cand = kwargs["messages"][1]["content"].split("Candidate Answer:")[-1]
            present = "22 days" in cand
            return json.dumps({"checks": [{"fact": "22 days", "present": present, "quote": "22 days" if present else ""}]})
        # Agent runtime
        if "STRICT_WIN" in sys_msg:
            return json.dumps(
                {
                    "thought": "exact",
                    "tool_call": None,
                    "final_answer": f"Shore staff get 22 days of annual leave [{cid}].",
                    "citations": [cid],
                    "refused": False,
                    "refusal_reason": None,
                }
            )
        return json.dumps(
            {
                "thought": "vague",
                "tool_call": None,
                "final_answer": "Check HR leave policy.",
                "citations": [cid],
                "refused": False,
                "refusal_reason": None,
            }
        )

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(handler),
    )
    judge = LayeredJudge(gateway=gw, grounder=LexicalNLIGrounder())

    result = run_prompt_tournament(base_agent, req, cases, retriever, gw, judge, chunks_by_id, tmp_path)
    assert result.winner_label == "candidate_1"
    assert "STRICT_WIN" in result.winner_agent.system_prompt.grounding_rules

    _, rp = generate_scorecard_and_report(
        tmp_path / "rep",
        req,
        result.winner_agent,
        cases,
        [],
        {},
        {},
        {},
        tournament=result,
    )
    assert "Prompt Tournament Comparison" in rp.read_text(encoding="utf-8")


def test_regression_mode_detects_doc_changes_and_broken_case(tmp_path: Path) -> None:
    old_docs = tmp_path / "old_docs"
    old_docs.mkdir()
    (old_docs / "leave.md").write_text("# Leave\nShore staff receive 22 days annual leave.", encoding="utf-8")

    new_docs = tmp_path / "new_docs"
    new_docs.mkdir()
    # Leave policy removed/replaced with unrelated policy
    (new_docs / "parking.md").write_text("# Parking\nVisitor parking is on level B1.", encoding="utf-8")

    docs, _ = parse_folder(old_docs)
    chunks = chunk_documents(docs)
    cid = chunks[0].chunk_id
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    retriever = HybridRetriever(embedder=TfidfEmbedder())
    retriever.index(chunks)
    retriever.save(run_dir / "index")

    req = RequirementsDoc(
        brief="HR",
        scope="HR",
        audience="Staff",
        tone="Brief",
        data_sources=["leave.md"],
    )
    save_requirements(req, run_dir / "requirements.yaml")
    save_eval_cases(
        [
            EvalCase(
                id="c1",
                category="single_hop",
                split="holdout",
                question="How many annual leave days?",
                expected_sources=["leave.md"],
                expected_chunk_ids=[cid],
                key_facts=["22 days"],
                should_refuse=False,
            )
        ],
        run_dir / "evals.jsonl",
    )
    agent = AgentDefinition(
        id="hr-0",
        version=0,
        name="HR Agent",
        description="HR",
        system_prompt=PromptSections(
            role="HR", scope="HR", grounding_rules="Cite", refusal_policy="Refuse"
        ),
        retrieval=RetrievalConfig(min_relevance_score=0.01),
    )
    agent.to_yaml(run_dir / "agent_final.yaml")

    def handler(kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        user_msg = kwargs["messages"][1]["content"]
        if "fact-focused evaluator" in sys_msg:
            cand = user_msg.split("Candidate Answer:")[-1]
            present = "22 days" in cand
            return json.dumps({"checks": [{"fact": "22 days", "present": present, "quote": "22 days" if present else ""}]})
        if "22 days" in user_msg:
            return json.dumps(
                {
                    "thought": "found",
                    "tool_call": None,
                    "final_answer": f"22 days [{cid}].",
                    "citations": [cid],
                    "refused": False,
                    "refusal_reason": None,
                }
            )
        return json.dumps(
            {
                "thought": "missing",
                "tool_call": None,
                "final_answer": "Not found in documents.",
                "citations": [],
                "refused": True,
                "refusal_reason": "Missing",
            }
        )

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(handler),
    )

    rep, out_path = run_regression(run_dir, new_docs, gw)
    assert rep.old_pass_rate == 1.0
    assert rep.new_pass_rate == 0.0
    assert len(rep.newly_broken) == 1
    assert rep.doc_changes["added"] == ["parking.md"]
    assert rep.doc_changes["removed"] == ["leave.md"]
    assert out_path.exists()


def test_factory_vs_human_baseline_ui_syntax_and_cli_help(tmp_path: Path) -> None:
    (tmp_path / "hr.md").write_text("# Leave\nShore staff receive 22 days annual leave.", encoding="utf-8")
    docs, _ = parse_folder(tmp_path)
    chunks = chunk_documents(docs)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    cid = chunks[0].chunk_id
    retriever = HybridRetriever(embedder=TfidfEmbedder())
    retriever.index(chunks)

    factory_agent = AgentDefinition(
        id="factory",
        name="Factory Agent",
        description="F",
        system_prompt=PromptSections(
            role="FACTORY_WIN", scope="HR", grounding_rules="Cite", refusal_policy="Refuse"
        ),
        retrieval=RetrievalConfig(min_relevance_score=0.01),
    )
    human_agent = AgentDefinition(
        id="human",
        name="Human Baseline",
        description="H",
        system_prompt=PromptSections(
            role="HUMAN_ROLE", scope="HR", grounding_rules="Cite", refusal_policy="Refuse"
        ),
        retrieval=RetrievalConfig(min_relevance_score=0.01),
    )
    cases = [
        EvalCase(
            id="c1",
            category="single_hop",
            split="holdout",
            question="How many annual leave days?",
            expected_sources=["hr.md"],
            expected_chunk_ids=[cid],
            key_facts=["22 days"],
            should_refuse=False,
        )
    ]

    def handler(kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        if "fact-focused evaluator" in sys_msg:
            cand = kwargs["messages"][1]["content"].split("Candidate Answer:")[-1]
            present = "22 days" in cand
            return json.dumps({"checks": [{"fact": "22 days", "present": present, "quote": "22 days" if present else ""}]})
        if "FACTORY_WIN" in sys_msg:
            return json.dumps(
                {
                    "thought": "ok",
                    "tool_call": None,
                    "final_answer": f"22 days [{cid}].",
                    "citations": [cid],
                    "refused": False,
                    "refusal_reason": None,
                }
            )
        return json.dumps(
            {
                "thought": "vague",
                "tool_call": None,
                "final_answer": "See HR handbook.",
                "citations": [cid],
                "refused": False,
                "refusal_reason": None,
            }
        )

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(handler),
    )
    judge = LayeredJudge(gateway=gw, grounder=LexicalNLIGrounder())

    out_md = tmp_path / "baseline_comparison.md"
    comp = compare_factory_vs_human(
        factory_agent, human_agent, cases, retriever, gw, judge, chunks_by_id, tmp_path, out_md
    )
    assert comp.factory_pass_rate == 1.0
    assert comp.human_pass_rate == 0.0
    assert comp.factory_wins == ["c1"]
    assert out_md.exists()

    # Streamlit UI syntax check & CLI help checks
    py_compile.compile("ui/app.py", doraise=True)
    runner = CliRunner()
    for sub in ("regress", "compare"):
        res = runner.invoke(app, [sub, "--help"])
        assert res.exit_code == 0

