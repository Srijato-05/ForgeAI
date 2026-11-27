"""Phase 0 verification tests for schemas, cache, tracer, rate limiter, structured LLM, and CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from forgeai.cli import app
from forgeai.config import ForgeSettings
from forgeai.llm import DiskCache, LLMGateway, RateLimiter, StructuredOutputError, TraceLogger, call_structured
from forgeai.schemas import (
    AgentDefinition,
    AgentResponse,
    CaseVerdict,
    ClarifyingQuestion,
    EvalCase,
    IterationLog,
    OutputContract,
    PatchOperation,
    PromptSections,
    RejectedTool,
    RequirementsDoc,
    RetrievalConfig,
    ToolAttachment,
    TopicGroundability,
)


class _SamplePayload(BaseModel):
    topic: str
    count: int


def _make_fake_client(responses: list[str]):
    call_log: list[dict] = []

    class _FakeCompletions:
        def create(self, **kwargs):
            idx = len(call_log)
            call_log.append(kwargs)
            content = responses[min(idx, len(responses) - 1)]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    return client, call_log


def test_agent_definition_yaml_round_trip_and_render(tmp_path: Path) -> None:
    agent = AgentDefinition(
        id="hr-001",
        version=0,
        name="HR Policy Assistant",
        description="Answers shore staff HR policy questions.",
        system_prompt=PromptSections(
            role="You are an HR policy assistant.",
            scope="Only answer questions about company HR policies.",
            grounding_rules="Cite chunk IDs for every factual claim.",
            refusal_policy="Refuse questions outside HR policy documents.",
            tool_guidance="",
            few_shot=["Q: Leave balance? -> A: Check policy [chunk_1]."],
        ),
        retrieval=RetrievalConfig(top_k=6),
        tools=[ToolAttachment(name="current_date", justification="Policy effective date check")],
        tools_rejected=[RejectedTool(name="weather_api", reason="Not needed for HR policy")],
        output_contract=OutputContract(require_citations=True, require_refusal_flag=True),
    )

    out_path = tmp_path / "agent.yaml"
    agent.to_yaml(out_path)
    loaded = AgentDefinition.from_yaml(out_path)

    assert loaded == agent
    rendered = loaded.render_system_prompt()
    assert "## Role\nYou are an HR policy assistant." in rendered
    assert "## Examples\nQ: Leave balance?" in rendered
    assert "## Tool Guidance" not in rendered


def test_other_schemas_instantiate() -> None:
    req = RequirementsDoc(
        brief="Need an HR assistant",
        scope="HR policies",
        audience="Shore staff",
        tone="Professional",
        must_answer=["Annual leave"],
        must_refuse=["Payroll disputes"],
        clarifying_questions=[
            ClarifyingQuestion(
                question="Include contractors?",
                rationale="Changes scope",
                default_answer="No, shore staff only",
                assumed=True,
            )
        ],
        groundability_checks=[
            TopicGroundability(topic="Annual leave", status="supported", evidence_chunks=["c1"])
        ],
    )
    case = EvalCase(
        id="e1",
        category="single_hop",
        split="dev",
        question="How many days of annual leave?",
        expected_sources=["hr.pdf"],
        expected_chunk_ids=["c1"],
        key_facts=["20 days"],
        should_refuse=False,
    )
    resp = AgentResponse(answer="20 days [c1]", citations=["c1"], refused=False)
    verdict = CaseVerdict(
        case_id="e1",
        passed=True,
        score=1.0,
        l0_schema_refusal_ok=True,
        l1_citation_hit=True,
        l2_fact_coverage=1.0,
        l4_nli_groundedness=1.0,
        l5_llm_correctness=1.0,
    )
    log = IterationLog(
        iteration=1,
        patch=PatchOperation(
            target_field="retrieval.top_k",
            old_value="4",
            new_value="8",
            rationale="Improve recall",
            targeted_case_ids=["e1"],
        ),
        score_before=0.75,
        score_after=0.90,
        regressions=0,
        accepted=True,
    )
    assert req.brief == "Need an HR assistant"
    assert case.id == resp.citations[0].replace("c1", "e1")
    assert verdict.passed and log.accepted


def test_disk_cache_and_trace_logger(tmp_path: Path) -> None:
    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    tracer = TraceLogger(tmp_path / "traces.jsonl")
    fake_client, call_log = _make_fake_client(['{"topic": "leave", "count": 5}'])

    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(settings.cache_dir),
        tracer=tracer,
        rate_limiter=RateLimiter(rpm=0, tpm=0),
        client_factory=lambda *_: fake_client,
    )

    msgs = [{"role": "user", "content": "test"}]
    first = gw.complete("factory", msgs, json_mode=True, node="intake")
    second = gw.complete("factory", msgs, json_mode=True, node="intake")

    assert first.cached is False
    assert second.cached is True
    assert len(call_log) == 1
    assert tracer.call_count == 2
    assert tracer.total_tokens == 40
    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_rate_limiter_acquire() -> None:
    limiter = RateLimiter(rpm=60, tpm=1000)
    limiter.acquire(estimated_tokens=100)
    limiter.record_actual_tokens(50)


def test_call_structured_happy_repair_and_exhaustion(tmp_path: Path) -> None:
    settings = ForgeSettings(cache_dir=tmp_path / "c1", forge_provider="ollama")

    # Happy path with markdown fences
    happy_client, _ = _make_fake_client(['```json\n{"topic": "safety", "count": 3}\n```'])
    gw_happy = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "c1"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: happy_client,
    )
    parsed = call_structured(gw_happy, "factory", [{"role": "user", "content": "hi"}], _SamplePayload)
    assert parsed == _SamplePayload(topic="safety", count=3)

    # Forced malformed JSON -> repair retry succeeds on attempt 2
    repair_client, repair_calls = _make_fake_client(
        [
            '{"topic": "safety", "count": "not_an_int"}',
            '{"topic": "safety", "count": 7}',
        ]
    )
    gw_repair = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "c2"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: repair_client,
    )
    repaired = call_structured(
        gw_repair, "factory", [{"role": "user", "content": "hi"}], _SamplePayload, max_retries=3
    )
    assert repaired.count == 7
    assert len(repair_calls) == 2

    # Exhaustion path
    bad_client, bad_calls = _make_fake_client(["not json at all", "still bad", "nope"])
    gw_bad = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "c3"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: bad_client,
    )
    with pytest.raises(StructuredOutputError):
        call_structured(gw_bad, "factory", [{"role": "user", "content": "hi"}], _SamplePayload, max_retries=3)
    assert len(bad_calls) == 3


def test_cli_doctor() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ForgeAI Environment Diagnostics" in result.stdout
    assert "Active Provider" in result.stdout

