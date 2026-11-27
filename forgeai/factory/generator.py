"""Declarative agent generator with 2-step selective tool attachment (Requirement 3)."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from forgeai.ingest import DocumentInventory
from forgeai.llm import LLMGateway, call_structured
from forgeai.prompts import load_prompt
from forgeai.schemas import (
    AgentDefinition,
    OutputContract,
    PromptSections,
    RejectedTool,
    RequirementsDoc,
    RetrievalConfig,
    ToolAttachment,
)
from forgeai.tools import ToolRegistry


class _GeneratedAgentPlan(BaseModel):
    name: str
    description: str
    system_prompt: PromptSections
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    selected_tools: list[ToolAttachment] = Field(default_factory=list)
    output_contract: OutputContract = Field(default_factory=OutputContract)


def generate_agent_definition(
    requirements: RequirementsDoc,
    inventory: DocumentInventory,
    gateway: LLMGateway,
    registry: ToolRegistry | None = None,
    agent_id: str = "agent_v0",
) -> AgentDefinition:
    """Produce a modular AgentDefinition with evidence-based selective tool attachment."""
    reg = registry or ToolRegistry()
    eligible, ineligible = reg.check_eligibility(inventory, requirements.brief)

    eligible_map = {tool.name: (tool, reason) for tool, reason in eligible}
    rejected: list[RejectedTool] = [
        RejectedTool(name=tool.name, reason=reason) for tool, reason in ineligible
    ]

    eligible_payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "eligibility_signal": reason,
        }
        for tool, reason in eligible
    ]

    gen_prompt = load_prompt("agent_generator")
    plan = call_structured(
        gateway,
        role="factory",
        messages=[
            {"role": "system", "content": gen_prompt},
            {
                "role": "user",
                "content": (
                    f"Requirements:\n{requirements.model_dump_json(indent=2)}\n\n"
                    f"Document Inventory:\n{json.dumps(inventory.model_dump(mode='json'), ensure_ascii=False)}\n\n"
                    f"Eligible Tools:\n{json.dumps(eligible_payload, ensure_ascii=False)}"
                ),
            },
        ],
        schema=_GeneratedAgentPlan,
        node="agent_generator",
    )

    attached: list[ToolAttachment] = []
    attached_names: set[str] = set()
    for candidate in plan.selected_tools:
        if candidate.name not in eligible_map:
            rejected.append(
                RejectedTool(
                    name=candidate.name,
                    reason="Selected by generator but failed deterministic eligibility check.",
                )
            )
            continue
        if not candidate.justification.strip():
            rejected.append(
                RejectedTool(
                    name=candidate.name,
                    reason="Dropped because generator provided no evidence-based justification.",
                )
            )
            continue
        if candidate.name not in attached_names:
            attached.append(candidate)
            attached_names.add(candidate.name)

    for name in eligible_map:
        if name not in attached_names:
            rejected.append(
                RejectedTool(
                    name=name,
                    reason="Eligible by heuristic but deemed unnecessary for the brief.",
                )
            )

    prompt_sections = plan.system_prompt
    if requirements.known_gaps:
        gaps_text = "; ".join(requirements.known_gaps)
        if gaps_text not in prompt_sections.refusal_policy:
            prompt_sections = prompt_sections.model_copy(
                update={
                    "refusal_policy": (
                        f"{prompt_sections.refusal_policy.strip()}\n"
                        f"Known unsupported gaps (MUST refuse rather than guess): {gaps_text}."
                    )
                }
            )

    return AgentDefinition(
        id=agent_id,
        version=0,
        parent_version=None,
        name=plan.name,
        description=plan.description,
        system_prompt=prompt_sections,
        retrieval=plan.retrieval,
        tools=attached,
        tools_rejected=rejected,
        output_contract=plan.output_contract,
    )

