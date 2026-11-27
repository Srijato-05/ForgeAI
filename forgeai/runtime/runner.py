"""Generic agent runtime shared by `forge chat` and the automated evaluation harness."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from forgeai.ingest.retriever import HybridRetriever
from forgeai.llm import LLMGateway, StructuredOutputError, call_structured
from forgeai.schemas import AgentDefinition, AgentResponse, ToolAttachment
from forgeai.tools import ToolRegistry


class _ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class _AgentStepOutput(BaseModel):
    thought: str = ""
    tool_call: _ToolCallRequest | None = None
    final_answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None


class AgentRuntime:
    """Executes any declarative AgentDefinition over its retriever and attached tools."""

    def __init__(
        self,
        agent_def: AgentDefinition,
        retriever: HybridRetriever,
        gateway: LLMGateway,
        registry: ToolRegistry | None = None,
        docs_dir: Path | None = None,
        max_tool_turns: int = 3,
    ) -> None:
        self.agent_def = agent_def
        self.retriever = retriever
        self.gateway = gateway
        self.registry = registry or ToolRegistry()
        self.docs_dir = Path(docs_dir) if docs_dir is not None else Path(".")
        self.max_tool_turns = max_tool_turns
        self._attached: dict[str, ToolAttachment] = {t.name: t for t in agent_def.tools}

    def _format_tools_block(self) -> str:
        if not self._attached:
            return "No external tools are attached. Answer strictly from retrieved context or refuse."
        lines = ["Attached tools (call ONLY if needed):"]
        for name in self._attached:
            tool = self.registry.get(name)
            if tool is None:
                continue
            lines.append(
                f"- {tool.name}: {tool.description} | params={json.dumps(tool.parameters_schema)}"
            )
        return "\n".join(lines)

    def _format_context_block(self, retrieved: list[Any]) -> str:
        if not retrieved:
            return "Retrieved Context: (no relevant document chunks met the relevance threshold)"
        blocks = ["Retrieved Context:"]
        for item in retrieved:
            c = item.chunk
            blocks.append(f"[{c.chunk_id}] (source={c.source_doc}, score={item.score:.3f}):\n{c.raw_text}")
        return "\n\n".join(blocks)

    def _rewrite_query(self, question: str) -> str:
        if not self.agent_def.retrieval.query_rewrite:
            return question
        try:
            res = self.gateway.complete(
                role="agent",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user's question into a concise keyword-rich search query. "
                            "Return ONLY the search query."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                json_mode=False,
                node="query_rewrite",
            )
            rewritten = res.content.strip()
            return rewritten or question
        except Exception:
            return question

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> AgentResponse:
        """Answer a single user question following the agent's retrieval, tool, and output contract."""
        t0 = time.perf_counter()
        search_query = self._rewrite_query(question)
        retrieved = self.retriever.search(search_query, self.agent_def.retrieval)
        retrieved_ids = [r.chunk.chunk_id for r in retrieved]

        system_prompt = (
            f"{self.agent_def.render_system_prompt()}\n\n"
            f"{self._format_tools_block()}\n\n"
            "To call a tool, set `tool_call` and leave `final_answer` as null. "
            "Otherwise, set `tool_call` to null and provide `final_answer`, `citations` (list of chunk_ids), "
            "`refused` (boolean), and `refusal_reason` (if refused)."
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": f"{self._format_context_block(retrieved)}\n\nUser Question: {question}",
            }
        )

        tool_trace: list[dict[str, Any]] = []
        last_step: _AgentStepOutput | None = None

        for _ in range(self.max_tool_turns + 1):
            try:
                step = call_structured(
                    self.gateway,
                    role="agent",
                    messages=messages,
                    schema=_AgentStepOutput,
                    node="agent_runtime",
                )
            except StructuredOutputError as exc:
                return AgentResponse(
                    answer="I could not produce a properly formatted grounded answer.",
                    citations=[],
                    refused=True,
                    refusal_reason=f"Structured output error: {exc}",
                    retrieved_chunk_ids=retrieved_ids,
                    tool_calls=tool_trace,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            last_step = step

            if step.tool_call is not None and step.final_answer is None:
                req = step.tool_call
                if len(tool_trace) >= self.max_tool_turns:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Tool limit reached. Provide `final_answer` now without further tool calls.",
                        }
                    )
                    continue

                if req.name not in self._attached:
                    obs = f"Tool '{req.name}' is not attached to this agent."
                    tool_trace.append({"tool": req.name, "args": req.arguments, "ok": False, "output": obs})
                else:
                    tool = self.registry.get(req.name)
                    if tool is None:
                        obs = f"Tool '{req.name}' is not registered."
                        tool_trace.append({"tool": req.name, "args": req.arguments, "ok": False, "output": obs})
                    else:
                        ctx = {
                            "docs_dir": self.docs_dir,
                            "tool_config": self._attached[req.name].config,
                        }
                        res = tool.run(req.arguments, ctx)
                        obs = res.output
                        tool_trace.append(
                            {
                                "tool": req.name,
                                "args": req.arguments,
                                "ok": res.ok,
                                "output": res.output,
                                "data": res.data,
                            }
                        )

                messages.append({"role": "assistant", "content": step.model_dump_json()})
                messages.append({"role": "user", "content": f"Tool Observation ({req.name}): {obs}"})
                continue

            break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        answer = (last_step.final_answer if last_step else None) or ""
        citations = last_step.citations if last_step else []
        refused = last_step.refused if last_step else True
        refusal_reason = last_step.refusal_reason if last_step else "No final answer produced."

        if not answer and not refused:
            refused = True
            refusal_reason = refusal_reason or "Unable to produce a grounded answer."
            answer = f"I cannot answer this based on the available documents. ({refusal_reason})"

        if self.agent_def.output_contract.max_words and answer:
            words = answer.split()
            if len(words) > self.agent_def.output_contract.max_words:
                answer = " ".join(words[: self.agent_def.output_contract.max_words])

        return AgentResponse(
            answer=answer,
            citations=citations,
            refused=refused,
            refusal_reason=refusal_reason,
            retrieved_chunk_ids=retrieved_ids,
            tool_calls=tool_trace,
            latency_ms=latency_ms,
        )

