"""Phase 1 verification tests for ingestion, chunking, hybrid retrieval, tools, runtime, and chat CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
from typer.testing import CliRunner

from forgeai.cli import app
from forgeai.config import ForgeSettings
from forgeai.ingest import HybridRetriever, TfidfEmbedder, chunk_documents, parse_folder
from forgeai.llm import DiskCache, LLMGateway, RateLimiter
from forgeai.runtime import AgentRuntime
from forgeai.schemas import (
    AgentDefinition,
    OutputContract,
    PromptSections,
    RetrievalConfig,
    ToolAttachment,
)
from forgeai.tools import CalculatorTool, CsvLookupTool, CurrentDateTool, ToolRegistry, WeatherApiTool


def _make_fake_client(responses: list[str]):
    call_log: list[dict] = []

    class _FakeCompletions:
        def create(self, **kwargs):
            idx = len(call_log)
            call_log.append(kwargs)
            content = responses[min(idx, len(responses) - 1)]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=15, completion_tokens=12),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    return client, call_log


def test_parse_folder_and_chunking(tmp_path: Path) -> None:
    (tmp_path / "policy.md").write_text(
        "# Leave Policy\nShore staff receive 22 days of annual leave per calendar year.\n\n"
        "## Carryover\nUp to 5 unused days may carry over into Q1.",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("Emergency leave requires manager sign-off.", encoding="utf-8")
    (tmp_path / "parts.csv").write_text("sku,name,price\nP-100,Valve,45.50\nP-200,Gasket,12.00\n", encoding="utf-8")

    docs, inventory = parse_folder(tmp_path)
    filenames = {d.filename for d in docs}
    assert filenames == {"policy.md", "notes.txt"}
    assert len(inventory.csv_profiles) == 1
    assert inventory.csv_profiles[0].filename == "parts.csv"
    assert inventory.csv_profiles[0].row_count == 2

    chunks = chunk_documents(docs, chunk_size=128, chunk_overlap=16)
    assert len(chunks) >= 3
    assert any("Leave Policy" in c.section_title for c in chunks)
    assert all(c.contextual_text.startswith("[Document:") for c in chunks)


def test_hybrid_retriever_search_and_save_load(tmp_path: Path) -> None:
    (tmp_path / "hr.md").write_text(
        "# Annual Leave\nShore staff get 22 days annual leave.\n\n"
        "# Travel Reimbursement\nMeals are reimbursed up to 60 USD per day.",
        encoding="utf-8",
    )
    docs, _ = parse_folder(tmp_path)
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    retriever = HybridRetriever(embedder=TfidfEmbedder())
    retriever.index(chunks)

    cfg = RetrievalConfig(top_k=5, rerank_top_n=2, min_relevance_score=0.05)
    hits = retriever.search("annual leave days for shore staff", cfg)
    assert len(hits) >= 1
    assert "22 days" in hits[0].chunk.raw_text

    index_dir = tmp_path / "idx"
    retriever.save(index_dir)
    loaded = HybridRetriever.load(index_dir)
    hits_loaded = loaded.search("annual leave days", cfg)
    assert len(hits_loaded) >= 1
    assert hits_loaded[0].chunk.chunk_id == hits[0].chunk.chunk_id


def test_tools_and_eligibility(tmp_path: Path) -> None:
    (tmp_path / "catalogue.csv").write_text(
        "part_no,desc,unit_price\nSP-01,Impeller,120.0\nSP-02,Seal Kit,35.0\nSP-03,Impeller Pro,200.0\n",
        encoding="utf-8",
    )
    (tmp_path / "hr.md").write_text("# Leave\nStandard leave policy.", encoding="utf-8")

    # CSV lookup filter + sum
    csv_tool = CsvLookupTool()
    res_filter = csv_tool.run(
        {"filename": "catalogue.csv", "query_column": "part_no", "query_value": "SP-01"},
        {"docs_dir": tmp_path},
    )
    assert res_filter.ok and res_filter.data[0]["unit_price"] == 120.0

    res_sum = csv_tool.run(
        {
            "filename": "catalogue.csv",
            "query_column": "desc",
            "query_value": "impeller",
            "operation": "sum",
            "target_column": "unit_price",
        },
        {"docs_dir": tmp_path},
    )
    assert res_sum.ok and res_sum.data["sum"] == 320.0

    # Calculator safe math + rejection
    calc = CalculatorTool()
    assert calc.run({"expression": "round(3 * 45.5 + 10, 2)"}, {}).data["value"] == 146.5
    unsafe = calc.run({"expression": "__import__('os').system('echo hi')"}, {})
    assert not unsafe.ok

    # Current date
    date_res = CurrentDateTool().run({}, {})
    assert date_res.ok and "date" in date_res.data

    # Weather API with mocked HTTP
    def fake_get(url, params=None, timeout=10.0):
        req = httpx.Request("GET", url, params=params)
        return httpx.Response(
            200,
            json={"current_weather": {"temperature": 21.5, "windspeed": 14.2}},
            request=req,
        )

    weather = WeatherApiTool(http_get=fake_get)
    w_res = weather.run({"latitude": 1.35, "longitude": 103.8}, {})
    assert w_res.ok and w_res.data["temperature"] == 21.5

    # Eligibility check: HR brief without CSV must reject weather_api and csv_lookup
    hr_dir = tmp_path / "hr_only"
    hr_dir.mkdir()
    (hr_dir / "policy.md").write_text("# HR Policy\nShore staff leave policy.", encoding="utf-8")
    _, hr_inv = parse_folder(hr_dir)
    registry = ToolRegistry()
    eligible, ineligible = registry.check_eligibility(hr_inv, "Answer HR policy questions for shore staff.")
    assert {t.name for t, _ in eligible} == set()
    assert {t.name for t, _ in ineligible} == {"csv_lookup", "calculator", "weather_api", "current_date"}


def test_agent_runtime_direct_tool_and_refusal(tmp_path: Path) -> None:
    (tmp_path / "spares.md").write_text(
        "# Ordering\nMinimum order quantity for Impeller SP-01 is 2 units.",
        encoding="utf-8",
    )
    (tmp_path / "catalogue.csv").write_text("part_no,price\nSP-01,50.0\n", encoding="utf-8")

    docs, _ = parse_folder(tmp_path)
    chunks = chunk_documents(docs)
    retriever = HybridRetriever(embedder=TfidfEmbedder())
    retriever.index(chunks)

    agent_def = AgentDefinition(
        id="spares-v0",
        name="Spares Assistant",
        description="Helps with ship spares.",
        system_prompt=PromptSections(
            role="You are a procurement assistant.",
            scope="Ship spares only.",
            grounding_rules="Cite chunk IDs.",
            refusal_policy="Refuse out-of-scope requests.",
            tool_guidance="Use calculator for totals.",
        ),
        retrieval=RetrievalConfig(min_relevance_score=0.01),
        tools=[ToolAttachment(name="calculator", justification="Compute order totals")],
        output_contract=OutputContract(),
    )

    settings = ForgeSettings(cache_dir=tmp_path / "cache", forge_provider="ollama")

    # (a) Direct grounded answer
    c1, _ = _make_fake_client(
        [
            '{"thought": "found chunk", "tool_call": null, "final_answer": "MOQ is 2 units.", "citations": ["spares__c000"], "refused": false, "refusal_reason": null}'
        ]
    )
    rt1 = AgentRuntime(
        agent_def,
        retriever,
        LLMGateway(settings, DiskCache(tmp_path / "c1"), rate_limiter=RateLimiter(0, 0), client_factory=lambda *_: c1),
        docs_dir=tmp_path,
    )
    r1 = rt1.ask("What is the MOQ for SP-01?")
    assert not r1.refused and "2 units" in r1.answer and "spares__c000" in r1.citations

    # (b) Tool-calling turn -> final answer
    c2, _ = _make_fake_client(
        [
            '{"thought": "compute 2 * 50", "tool_call": {"name": "calculator", "arguments": {"expression": "2 * 50"}}, "final_answer": null, "citations": [], "refused": false, "refusal_reason": null}',
            '{"thought": "done", "tool_call": null, "final_answer": "Total cost is 100.", "citations": ["spares__c000"], "refused": false, "refusal_reason": null}',
        ]
    )
    rt2 = AgentRuntime(
        agent_def,
        retriever,
        LLMGateway(settings, DiskCache(tmp_path / "c2"), rate_limiter=RateLimiter(0, 0), client_factory=lambda *_: c2),
        docs_dir=tmp_path,
    )
    r2 = rt2.ask("Total cost for 2 units at 50 each?")
    assert not r2.refused and "100" in r2.answer
    assert len(r2.tool_calls) == 1 and r2.tool_calls[0]["tool"] == "calculator"

    # (c) Refusal
    c3, _ = _make_fake_client(
        [
            '{"thought": "out of scope", "tool_call": null, "final_answer": "I cannot answer weather questions.", "citations": [], "refused": true, "refusal_reason": "Out of scope"}'
        ]
    )
    rt3 = AgentRuntime(
        agent_def,
        retriever,
        LLMGateway(settings, DiskCache(tmp_path / "c3"), rate_limiter=RateLimiter(0, 0), client_factory=lambda *_: c3),
        docs_dir=tmp_path,
    )
    r3 = rt3.ask("What is the weather in Oslo?")
    assert r3.refused and r3.refusal_reason == "Out of scope"


def test_cli_chat_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--docs" in result.stdout
    assert "--embedder" in result.stdout

