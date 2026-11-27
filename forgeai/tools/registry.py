"""Central registry for generic tools and eligibility filtering."""

from __future__ import annotations

from forgeai.ingest.parsers import DocumentInventory
from forgeai.tools.base import BaseTool
from forgeai.tools.calculator import CalculatorTool
from forgeai.tools.csv_lookup import CsvLookupTool
from forgeai.tools.current_date import CurrentDateTool
from forgeai.tools.weather_api import WeatherApiTool


class ToolRegistry:
    """Holds all available generic tools and checks domain eligibility."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        default_tools: list[BaseTool] = [
            CsvLookupTool(),
            CalculatorTool(),
            WeatherApiTool(),
            CurrentDateTool(),
        ]
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or default_tools)}

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def check_eligibility(
        self, inventory: DocumentInventory, brief: str
    ) -> tuple[list[tuple[BaseTool, str]], list[tuple[BaseTool, str]]]:
        """Partition tools into (eligible_with_reason, ineligible_with_reason)."""
        eligible: list[tuple[BaseTool, str]] = []
        ineligible: list[tuple[BaseTool, str]] = []
        for tool in self._tools.values():
            ok, reason = tool.is_eligible(inventory, brief)
            if ok:
                eligible.append((tool, reason))
            else:
                ineligible.append((tool, reason))
        return eligible, ineligible

