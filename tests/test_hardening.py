"""Verification tests for pre-Phase-5 hardening improvements."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel
from typer.testing import CliRunner

from forgeai.cli import app
from forgeai.config import ForgeSettings
from forgeai.ingest import HybridRetriever, TfidfEmbedder, chunk_documents, parse_folder
from forgeai.llm import DiskCache, LLMGateway, RateLimiter, call_structured
from forgeai.runtime import AgentRuntime
from forgeai.schemas import AgentDefinition, OutputContract, PromptSections, RetrievalConfig


class _Sample(BaseModel):
    answer: str
    score: int


def _make_routed_client(handler):
    class _FakeCompletions:
        def create(self, **kwargs):
            content = handler(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))


def test_call_structured_strips_think_tags_and_prose(tmp_path: Path) -> None:
    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    raw = '<think>\nLet me analyze carefully.\n</think>\nHere is the JSON:\n{"answer": "ok", "score": 5}\nDone.'
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(lambda _: raw),
    )
    parsed = call_structured(gw, "factory", [{"role": "user", "content": "hi"}], _Sample)
    assert parsed == _Sample(answer="ok", score=5)


def test_runtime_query_rewrite(tmp_path: Path) -> None:
    (tmp_path / "hr.md").write_text("# Leave\nShore staff get 22 days annual leave.", encoding="utf-8")
    docs, _ = parse_folder(tmp_path)
    chunks = chunk_documents(docs)
    cid = chunks[0].chunk_id
    retriever = HybridRetriever(embedder=TfidfEmbedder())
    retriever.index(chunks)

    agent = AgentDefinition(
        id="hr-rw",
        name="HR",
        description="HR",
        system_prompt=PromptSections(
            role="HR", scope="HR", grounding_rules="Cite", refusal_policy="Refuse"
        ),
        retrieval=RetrievalConfig(min_relevance_score=0.01, query_rewrite=True),
        output_contract=OutputContract(),
    )

    seen_nodes: list[str] = []

    def handler(kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        if "search query" in sys_msg:
            seen_nodes.append("rewrite")
            return "shore staff annual leave days"
        seen_nodes.append("answer")
        return (
            f'{{"thought": "ok", "tool_call": null, "final_answer": "22 days [{cid}].", '
            f'"citations": ["{cid}"], "refused": false, "refusal_reason": null}}'
        )

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: _make_routed_client(handler),
    )
    rt = AgentRuntime(agent, retriever, gw, docs_dir=tmp_path)
    resp = rt.ask("How much vacation?")
    assert not resp.refused and "22 days" in resp.answer
    assert seen_nodes == ["rewrite", "answer"]


def test_cli_hardening_flags() -> None:
    runner = CliRunner()
    res_doc = runner.invoke(app, ["doctor", "--help"])
    assert res_doc.exit_code == 0 and "--ping" in res_doc.stdout

    res_build = runner.invoke(app, ["build", "--help"], env={"COLUMNS": "140"})
    assert res_build.exit_code == 0 and "--human-checkpoint" in res_build.stdout


def test_gateway_json_mode_400_fallback(tmp_path: Path) -> None:
    calls: list[dict] = []

    class _Bad400(Exception):
        status_code = 400

    class _FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise _Bad400("output_parse_failed")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer": "ok", "score": 9}'))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=8),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: client,
    )
    parsed = call_structured(gw, "agent", [{"role": "user", "content": "hi"}], _Sample)
    assert parsed == _Sample(answer="ok", score=9)
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_gateway_tool_use_failed_recovery(tmp_path: Path) -> None:
    import json

    class _ToolUse400(Exception):
        status_code = 400
        body = {
            "error": {
                "message": "Tool choice is none, but model called a tool",
                "code": "tool_use_failed",
                "failed_generation": '{"name": "csv_lookup", "arguments": {"filename": "spares.csv", "query_value": "NM-101"}}',
            }
        }

    class _FakeCompletions:
        def create(self, **kwargs):
            raise _ToolUse400("tool_use_failed")

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")
    gw = LLMGateway(
        settings=settings,
        cache=DiskCache(tmp_path / "cache"),
        rate_limiter=RateLimiter(0, 0),
        client_factory=lambda *_: client,
    )
    res = gw.complete("agent", [{"role": "user", "content": "Check NM-101"}], json_mode=True, node="agent_runtime")
    payload = json.loads(res.content)
    assert payload["tool_call"] == {
        "name": "csv_lookup",
        "arguments": {"filename": "spares.csv", "query_value": "NM-101"},
    }
    assert payload["final_answer"] is None

