"""Factory vs. human-crafted baseline comparison stretch goal."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from forgeai.factory.judge import LayeredJudge
from forgeai.factory.optimizer import evaluate_agent_on_cases
from forgeai.ingest import Chunk, HybridRetriever
from forgeai.llm import LLMGateway
from forgeai.schemas import AgentDefinition, EvalCase
from forgeai.tools import ToolRegistry


class BaselineComparison(BaseModel):
    """Side-by-side comparison between a ForgeAI-built agent and a hand-written agent."""

    factory_pass_rate: float
    human_pass_rate: float
    factory_wins: list[str] = Field(default_factory=list)
    human_wins: list[str] = Field(default_factory=list)
    ties: list[str] = Field(default_factory=list)
    markdown: str = ""


def compare_factory_vs_human(
    factory_agent: AgentDefinition,
    human_agent: AgentDefinition,
    eval_cases: list[EvalCase],
    retriever: HybridRetriever,
    gateway: LLMGateway,
    judge: LayeredJudge,
    chunks_by_id: dict[str, Chunk],
    docs_dir: Path,
    output_path: Path | None = None,
    registry: ToolRegistry | None = None,
) -> BaselineComparison:
    """Evaluate both agents on ``eval_cases`` and summarize where each wins or loses."""
    _, f_verdicts, f_rate = evaluate_agent_on_cases(
        factory_agent, eval_cases, retriever, gateway, judge, chunks_by_id, docs_dir, registry
    )
    _, h_verdicts, h_rate = evaluate_agent_on_cases(
        human_agent, eval_cases, retriever, gateway, judge, chunks_by_id, docs_dir, registry
    )

    factory_wins: list[str] = []
    human_wins: list[str] = []
    ties: list[str] = []

    rows: list[str] = []
    for case in eval_cases:
        fv = f_verdicts[case.id]
        hv = h_verdicts[case.id]
        if fv.passed and not hv.passed:
            factory_wins.append(case.id)
            winner = "Factory"
        elif hv.passed and not fv.passed:
            human_wins.append(case.id)
            winner = "Human"
        else:
            ties.append(case.id)
            winner = "Tie (Pass)" if fv.passed else "Tie (Fail)"
        rows.append(
            f"| `{case.id}` | {case.category} | {fv.score:.2f} ({'pass' if fv.passed else 'fail'}) | "
            f"{hv.score:.2f} ({'pass' if hv.passed else 'fail'}) | {winner} |"
        )

    md_lines = [
        "# Factory vs. Human Baseline Comparison",
        "",
        f"- **ForgeAI Agent (`{factory_agent.name}` v{factory_agent.version}):** {f_rate:.1%}",
        f"- **Human Baseline (`{human_agent.name}` v{human_agent.version}):** {h_rate:.1%}",
        f"- **Factory Wins:** {len(factory_wins)} ({factory_wins or 'None'})",
        f"- **Human Wins:** {len(human_wins)} ({human_wins or 'None'})",
        f"- **Ties:** {len(ties)}",
        "",
        "| Case ID | Category | ForgeAI | Human | Outcome |",
        "|---|---|---:|---:|---|",
        *rows,
    ]
    markdown = "\n".join(md_lines) + "\n"

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")

    return BaselineComparison(
        factory_pass_rate=f_rate,
        human_pass_rate=h_rate,
        factory_wins=factory_wins,
        human_wins=human_wins,
        ties=ties,
        markdown=markdown,
    )

