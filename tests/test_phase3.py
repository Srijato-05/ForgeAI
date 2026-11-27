"""Phase 3 verification tests for intake, groundability guard, generator, and LangGraph wiring."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver

from forgeai.config import ForgeSettings
from forgeai.factory import (
    build_factory_graph,
    generate_agent_definition,
    load_requirements,
    run_groundability_guard,
    run_intake,
    save_requirements,
)
from forgeai.ingest import HybridRetriever, TfidfEmbedder, chunk_documents, parse_folder
from forgeai.llm import DiskCache, LLMGateway, RateLimiter
from forgeai.prompts import load_prompt
from forgeai.schemas import RequirementsDoc


def _make_routed_client(handler):
    class _FakeCompletions:
        def create(self, **kwargs):
            content = handler(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=15),
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))


def _factory_mock_handler(kwargs):
    sys_msg = kwargs["messages"][0]["content"]
    user_msg = kwargs["messages"][1]["content"]

    if "Intake Interviewer" in sys_msg:
        # Return 6 questions to verify clamping to <= 5
        qs = [
            {
                "question": f"Clarifying question {i}?",
                "rationale": "Clarifies scope",
                "default_answer": f"Default answer {i}",
            }
            for i in range(1, 7)
        ]
        return json.dumps({"questions": qs})

    if "Intake Requirements Architect" in sys_msg:
        return json.dumps(
            {
                "scope": "HR policy questions for shore staff",
                "audience": "Shore staff",
                "tone": "Professional and concise",
                "must_answer": ["Annual leave entitlement", "Crypto stock options"],
                "must_refuse": ["Personal salary negotiations"],
                "assumptions": ["Shore staff only"],
            }
        )

    if "Groundability Auditor" in sys_msg:
        if "Crypto stock options" in user_msg:
            return json.dumps({"status": "unsupported", "notes": "Not mentioned in docs."})
        return json.dumps({"status": "supported", "notes": "Explicitly covered."})

    if "Agent Architect" in sys_msg:
        has_csv = "csv_lookup" in user_msg
        tools = (
            [{"name": "csv_lookup", "justification": "catalogue.csv is present", "config": {}}]
            if has_csv
            else []
        )
        # Also include weather_api even if not eligible to prove deterministic gate drops it
        if not has_csv:
            tools = [{"name": "weather_api", "justification": "hallucinated tool"}]
        return json.dumps(
            {
                "name": "Generated Assistant",
                "description": "Answers grounded queries.",
                "system_prompt": {
                    "role": "You are a helpful assistant.",
                    "scope": "Answer strictly within scope.",
                    "grounding_rules": "Cite chunk IDs.",
                    "refusal_policy": "Refuse out-of-scope requests.",
                    "tool_guidance": "Use attached tools only when needed.",
                    "style": "Concise",
                    "few_shot": [],
                },
                "retrieval": {"top_k": 6, "rerank_top_n": 3, "min_relevance_score": 0.1},
                "selected_tools": tools,
                "output_contract": {"require_citations": True, "require_refusal_flag": True},
            }
        )

    if "single-hop" in sys_msg:
        return json.dumps(
            {
                "cases": [
                    {
                        "question": "How many days of paid leave do shore staff receive?",
                        "key_facts": ["22 days"],
                        "rationale": "leave",
                    }
                ]
            }
        )
    if "multi-document" in sys_msg:
        return json.dumps(
            {
                "cases": [
                    {
                        "question": "Compare leave days and daily meal caps for shore staff.",
                        "key_facts": ["22 days", "60 USD"],
                        "rationale": "multi",
                    }
                ]
            }
        )
    if "ambiguous" in sys_msg:
        return json.dumps(
            {
                "cases": [
                    {
                        "question": "How much time off do I get?",
                        "key_facts": ["22 days"],
                        "rationale": "ambig",
                    }
                ]
            }
        )
    if "MUST REFUSE" in sys_msg:
        return json.dumps(
            {
                "cases": [
                    {
                        "question": "Do shore staff get crypto stock options?",
                        "key_facts": [],
                        "rationale": "known gap",
                    },
                    {
                        "question": "Can you negotiate my salary?",
                        "key_facts": [],
                        "rationale": "must refuse",
                    },
                ]
            }
        )
    if "QA auditor" in sys_msg:
        return json.dumps({"answerable": True, "reason": "Supported"})
    return json.dumps({"cases": []})


def test_prompts_loaded_and_intake_modes(tmp_path: Path) -> None:
    for name in ("intake_questions", "intake_requirements", "groundability_probe", "agent_generator"):
        assert len(load_prompt(name)) > 20

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "leave.md").write_text("# Leave\nShore staff get 22 days annual leave.", encoding="utf-8")
    _, inv = parse_folder(docs_dir)

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(_factory_mock_handler),
    )

    # Unattended mode
    req_auto = run_intake("HR assistant for shore staff", inv, gw, unattended=True)
    assert len(req_auto.clarifying_questions) == 5
    assert all(q.assumed for q in req_auto.clarifying_questions)

    req_path = tmp_path / "requirements.yaml"
    save_requirements(req_auto, req_path)
    assert load_requirements(req_path) == req_auto

    # Interactive mode overriding Q1 and leaving Q2 blank
    answers = iter(["Include permanent shore staff only", ""])
    req_interactive = run_intake(
        "HR assistant for shore staff",
        inv,
        gw,
        ask_user=lambda _: next(answers, ""),
        unattended=False,
    )
    assert req_interactive.clarifying_questions[0].assumed is False
    assert req_interactive.clarifying_questions[0].user_answer == "Include permanent shore staff only"
    assert req_interactive.clarifying_questions[1].assumed is True


def test_groundability_guard_and_selective_tools(tmp_path: Path) -> None:
    hr_dir = tmp_path / "hr"
    hr_dir.mkdir()
    (hr_dir / "leave.md").write_text("# Leave\nShore staff receive 22 days annual leave.", encoding="utf-8")
    docs, hr_inv = parse_folder(hr_dir)
    chunks = chunk_documents(docs)
    retriever = HybridRetriever(embedder=TfidfEmbedder())
    retriever.index(chunks)

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(_factory_mock_handler),
    )

    req = RequirementsDoc(
        brief="Answer HR policy questions for shore staff.",
        scope="HR policy",
        audience="Shore staff",
        tone="Professional",
        must_answer=["Annual leave entitlement", "Crypto stock options"],
        must_refuse=["Salary negotiations"],
    )
    checked = run_groundability_guard(req, retriever, gw)
    assert checked.must_answer == ["Annual leave entitlement"]
    assert "Crypto stock options" in checked.known_gaps
    assert "Crypto stock options" in checked.must_refuse

    # HR agent must attach 0 tools and reject hallucinated weather_api
    hr_agent = generate_agent_definition(checked, hr_inv, gw)
    assert hr_agent.tools == []
    assert any(r.name == "weather_api" for r in hr_agent.tools_rejected)
    assert "Crypto stock options" in hr_agent.system_prompt.refusal_policy

    # Spares folder with CSV attaches csv_lookup and rejects weather_api
    spares_dir = tmp_path / "spares"
    spares_dir.mkdir()
    (spares_dir / "catalogue.csv").write_text("part,price\nP1,50\n", encoding="utf-8")
    _, spares_inv = parse_folder(spares_dir)
    spares_req = checked.model_copy(update={"brief": "Ship spares procurement lookup and pricing."})
    spares_agent = generate_agent_definition(spares_req, spares_inv, gw)
    assert [t.name for t in spares_agent.tools] == ["csv_lookup"]
    assert any(r.name == "weather_api" for r in spares_agent.tools_rejected)


def test_langgraph_factory_end_to_end_and_human_interrupt(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "leave.md").write_text("# Leave\nShore staff receive 22 days of annual leave.", encoding="utf-8")
    (docs_dir / "travel.md").write_text("# Travel\nDaily meal cap is 60 USD.", encoding="utf-8")

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(_factory_mock_handler),
    )

    # 1. Unattended end-to-end graph execution
    run_dir = tmp_path / "run_auto"
    graph = build_factory_graph(gateway=gw)
    final_state = graph.invoke(
        {
            "brief": "Need an HR assistant for shore staff.",
            "docs_dir": str(docs_dir),
            "run_dir": str(run_dir),
            "unattended": True,
            "target_eval_size": 10,
        }
    )

    assert (run_dir / "requirements.yaml").exists()
    assert (run_dir / "agent_v0.yaml").exists()
    assert (run_dir / "evals.jsonl").exists()
    assert (run_dir / "index" / "chunks.json").exists()
    assert final_state["agent_def"].tools == []
    assert len(final_state["eval_cases"]) >= 2

    # 2. Human checkpoint interrupt + edit requirements + resume
    run_human = tmp_path / "run_human"
    saver = MemorySaver()
    human_graph = build_factory_graph(gateway=gw, checkpointer=saver, interrupt_for_human=True)
    cfg = {"configurable": {"thread_id": "human-1"}}
    human_graph.invoke(
        {
            "brief": "Need an HR assistant for shore staff.",
            "docs_dir": str(docs_dir),
            "run_dir": str(run_human),
            "unattended": True,
            "target_eval_size": 8,
        },
        config=cfg,
    )
    snapshot = human_graph.get_state(cfg)
    assert snapshot.next == ("human_checkpoint",)
    assert not (run_human / "agent_v0.yaml").exists()

    # Simulate human editing requirements.yaml before resuming
    edited = load_requirements(run_human / "requirements.yaml").model_copy(
        update={"tone": "Friendly and ultra-concise"}
    )
    save_requirements(edited, run_human / "requirements.yaml")

    resumed = human_graph.invoke(None, config=cfg)
    assert (run_human / "agent_v0.yaml").exists()
    assert resumed["requirements"].tone == "Friendly and ultra-concise"

