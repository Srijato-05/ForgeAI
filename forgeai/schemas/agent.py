"""Declarative, modular agent definition schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PromptSections(BaseModel):
    """Independently patchable sections of the generated agent's system prompt."""

    role: str
    scope: str
    grounding_rules: str
    refusal_policy: str
    tool_guidance: str = ""
    style: str = "Concise, accurate, and grounded in cited sources."
    few_shot: list[str] = Field(default_factory=list)


class RetrievalConfig(BaseModel):
    """Chunking and hybrid retrieval parameters."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 8
    rerank_top_n: int = 4
    hybrid_alpha: float = 0.5
    min_relevance_score: float = 0.25
    query_rewrite: bool = False


class ToolAttachment(BaseModel):
    """A tool attached to the agent with explicit evidence-based justification."""

    name: str
    justification: str
    config: dict[str, Any] = Field(default_factory=dict)


class RejectedTool(BaseModel):
    """A tool explicitly evaluated and rejected for this agent."""

    name: str
    reason: str


class OutputContract(BaseModel):
    """Response format guarantees enforced by the runtime."""

    require_citations: bool = True
    require_refusal_flag: bool = True
    max_words: int | None = None


class AgentDefinition(BaseModel):
    """Complete declarative specification of a generated agent."""

    id: str
    version: int = 0
    parent_version: int | None = None
    name: str
    description: str
    system_prompt: PromptSections
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: list[ToolAttachment] = Field(default_factory=list)
    tools_rejected: list[RejectedTool] = Field(default_factory=list)
    output_contract: OutputContract = Field(default_factory=OutputContract)

    def render_system_prompt(self) -> str:
        """Assemble the modular prompt sections into a single system prompt string."""
        parts = [
            f"## Role\n{self.system_prompt.role}",
            f"## Scope\n{self.system_prompt.scope}",
            f"## Grounding Rules\n{self.system_prompt.grounding_rules}",
            f"## Refusal Policy\n{self.system_prompt.refusal_policy}",
        ]
        if self.system_prompt.tool_guidance.strip():
            parts.append(f"## Tool Guidance\n{self.system_prompt.tool_guidance}")
        if self.system_prompt.style.strip():
            parts.append(f"## Style\n{self.system_prompt.style}")
        if self.system_prompt.few_shot:
            examples = "\n\n".join(self.system_prompt.few_shot)
            parts.append(f"## Examples\n{examples}")
        return "\n\n".join(parts)

    def to_yaml(self, path: Path) -> None:
        """Serialize this agent definition to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: Path) -> AgentDefinition:
        """Load an AgentDefinition from a YAML file."""
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

