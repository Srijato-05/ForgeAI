"""Safe AST-evaluated arithmetic calculator tool."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

from forgeai.ingest.parsers import DocumentInventory
from forgeai.tools.base import ToolResult

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

_ELIGIBLE_RE = re.compile(
    r"\b(calculat\w*|math|arithmetic|cost|price|pricing|total|sum|quantity|budget|margin|discount|tax|percent\w*)\b",
    re.IGNORECASE,
)


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        if isinstance(node.op, ast.Pow):
            exp = _eval_node(node.right)
            if abs(exp) > 100:
                raise ValueError("Exponent too large.")
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")
        args = [_eval_node(arg) for arg in node.args]
        return _SAFE_FUNCS[node.func.id](*args)
    raise ValueError(f"Disallowed expression node: {type(node).__name__}")


class CalculatorTool:
    """Safely evaluate arithmetic expressions using a strict Python AST whitelist."""

    name = "calculator"
    description = "Evaluate safe arithmetic expressions (+, -, *, /, //, %, **, abs, round, min, max)."
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression, e.g. '3 * 49.95 + 12.0'"},
        },
        "required": ["expression"],
    }

    def is_eligible(self, inventory: DocumentInventory, brief: str) -> tuple[bool, str]:
        if _ELIGIBLE_RE.search(brief) or inventory.csv_profiles:
            return True, "Brief or tabular data involves quantities, pricing, or arithmetic."
        return False, "Brief does not indicate arithmetic or pricing calculations."

    def run(self, args: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return ToolResult(ok=False, output="Missing 'expression' argument.")
        try:
            tree = ast.parse(expr, mode="eval")
            value = _eval_node(tree)
            return ToolResult(ok=True, output=f"{value}", data={"value": value})
        except Exception as exc:
            return ToolResult(ok=False, output=f"Invalid arithmetic expression: {exc}")

