"""Base tool interface and result model."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from forgeai.ingest.parsers import DocumentInventory


class ToolResult(BaseModel):
    """Standardized output returned by any tool execution."""

    ok: bool
    output: str
    data: Any = None


class BaseTool(Protocol):
    """Protocol implemented by every tool in ForgeAI's tool registry."""

    name: str
    description: str
    parameters_schema: dict[str, Any]

    def is_eligible(self, inventory: DocumentInventory, brief: str) -> tuple[bool, str]:
        """Determine whether this tool is justified by the brief or documents."""
        ...

    def run(self, args: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        """Execute the tool with validated arguments and runtime context."""
        ...

