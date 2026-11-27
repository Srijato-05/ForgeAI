"""Deterministic CSV lookup and aggregation tool powered by pandas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from forgeai.ingest.parsers import DocumentInventory
from forgeai.tools.base import ToolResult


class CsvLookupTool:
    """Filter, search, and aggregate rows in CSV files."""

    name = "csv_lookup"
    description = (
        "Query and aggregate CSV tables in the document folder. "
        "Supports operations: filter, count, sum, min, max, head."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "CSV filename (optional if only one CSV exists)."},
            "query_column": {"type": "string", "description": "Column to filter on."},
            "query_value": {"type": "string", "description": "Value or substring to match (case-insensitive)."},
            "operation": {
                "type": "string",
                "enum": ["filter", "count", "sum", "min", "max", "head"],
                "default": "filter",
            },
            "target_column": {"type": "string", "description": "Numeric column for sum/min/max."},
            "match_mode": {"type": "string", "enum": ["auto", "exact", "contains"], "default": "auto"},
            "limit": {"type": "integer", "default": 10},
        },
    }

    def is_eligible(self, inventory: DocumentInventory, brief: str) -> tuple[bool, str]:
        if inventory.csv_profiles:
            names = ", ".join(p.filename for p in inventory.csv_profiles)
            return True, f"CSV file(s) present in folder: {names}"
        return False, "No CSV files found in the document folder."

    def _resolve_csv(self, filename: str | None, context: dict[str, Any]) -> Path:
        docs_dir = Path(context.get("docs_dir", "."))
        cfg_file = context.get("tool_config", {}).get("file")
        target = filename or cfg_file
        if target:
            direct = docs_dir / target
            if direct.exists():
                return direct
            matches = list(docs_dir.rglob(Path(target).name))
            if matches:
                return matches[0]
            raise FileNotFoundError(f"CSV file not found: {target}")

        csvs = sorted(docs_dir.rglob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]
        if not csvs:
            raise FileNotFoundError("No CSV files found in docs_dir.")
        raise ValueError(f"Multiple CSV files present; specify 'filename' among {[p.name for p in csvs]}.")

    def run(self, args: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            csv_path = self._resolve_csv(args.get("filename"), context)
            df = pd.read_csv(csv_path)
        except Exception as exc:
            return ToolResult(ok=False, output=f"CSV load error: {exc}")

        query_col = args.get("query_column")
        query_val = args.get("query_value")
        operation = str(args.get("operation") or "filter").lower()
        target_col = args.get("target_column")
        match_mode = str(args.get("match_mode") or "auto").lower()
        limit = int(args.get("limit") or 10)

        filtered = df
        if query_col and query_val is not None:
            if query_col not in df.columns:
                return ToolResult(ok=False, output=f"Unknown column '{query_col}'. Available: {list(df.columns)}")
            needle = str(query_val).strip().lower()
            series = df[query_col].astype(str).str.lower()
            if match_mode == "exact":
                filtered = df[series == needle]
            elif match_mode == "contains":
                filtered = df[series.str.contains(needle, regex=False, na=False)]
            elif operation in {"sum", "count", "min", "max"}:
                filtered = df[series.str.contains(needle, regex=False, na=False)]
            else:
                exact = df[series == needle]
                filtered = exact if not exact.empty else df[series.str.contains(needle, regex=False, na=False)]

        if operation == "count":
            count = int(len(filtered))
            return ToolResult(ok=True, output=f"count={count}", data={"count": count})

        if operation in {"sum", "min", "max"}:
            if not target_col or target_col not in filtered.columns:
                return ToolResult(
                    ok=False,
                    output=f"target_column '{target_col}' invalid. Available: {list(filtered.columns)}",
                )
            numeric = pd.to_numeric(filtered[target_col], errors="coerce").dropna()
            if numeric.empty:
                return ToolResult(ok=False, output=f"No numeric values in '{target_col}' after filtering.")
            val = float(getattr(numeric, operation)())
            return ToolResult(ok=True, output=f"{operation}({target_col})={val}", data={operation: val})

        rows = filtered.head(limit).fillna("").to_dict(orient="records")
        return ToolResult(ok=True, output=f"Matched {len(filtered)} row(s): {rows}", data=rows)

