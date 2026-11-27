"""Prompt template loader for ForgeAI's committed prompt files."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template from ``forgeai/prompts/<name>.txt``."""
    stem = name[:-4] if name.endswith(".txt") else name
    path = PROMPTS_DIR / f"{stem}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()

