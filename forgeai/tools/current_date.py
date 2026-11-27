"""Current UTC date and weekday tool."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from forgeai.ingest.parsers import DocumentInventory
from forgeai.tools.base import ToolResult

_DATE_RE = re.compile(
    r"\b(today|current date|expir\w*|deadline|lead time|valid until|as of|schedule|due date)\b",
    re.IGNORECASE,
)


class CurrentDateTool:
    """Return the current UTC date and day of week."""

    name = "current_date"
    description = "Return today's UTC date (YYYY-MM-DD) and day of week."
    parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def is_eligible(self, inventory: DocumentInventory, brief: str) -> tuple[bool, str]:
        combined = brief + " " + " ".join(d.get("preview", "") for d in inventory.documents)
        if _DATE_RE.search(combined):
            return True, "Brief or documents reference current dates, deadlines, expiry, or lead times."
        return False, "No temporal/date-relative reasoning detected in brief or documents."

    def run(self, args: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        now = datetime.now(timezone.utc)
        iso_date = now.strftime("%Y-%m-%d")
        weekday = now.strftime("%A")
        return ToolResult(
            ok=True,
            output=f"Current UTC date: {iso_date} ({weekday})",
            data={"date": iso_date, "weekday": weekday},
        )

