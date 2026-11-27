"""Regression mode stretch goal: re-run saved evals when documents change."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from forgeai.factory.evalgen import load_eval_cases
from forgeai.factory.generator import generate_agent_definition
from forgeai.factory.intake import load_requirements
from forgeai.factory.judge import LayeredJudge, NLIGrounder
from forgeai.factory.optimizer import evaluate_agent_on_cases
from forgeai.ingest import (
    Embedder,
    HybridRetriever,
    Reranker,
    TfidfEmbedder,
    chunk_documents,
    parse_folder,
)
from forgeai.llm import LLMGateway
from forgeai.schemas import AgentDefinition


class RegressionReport(BaseModel):
    """Comparison between an existing run and an updated document folder."""

    old_pass_rate: float
    new_pass_rate: float
    newly_broken: list[dict[str, str]] = Field(default_factory=list)
    newly_fixed: list[dict[str, str]] = Field(default_factory=list)
    unchanged_pass: int = 0
    unchanged_fail: int = 0
    doc_changes: dict[str, list[str]] = Field(default_factory=dict)


def run_regression(
    run_dir: Path,
    updated_docs_dir: Path,
    gateway: LLMGateway,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    grounder: NLIGrounder | None = None,
    regenerate_agent: bool = False,
) -> tuple[RegressionReport, Path]:
    """Re-index updated documents, optionally regenerate agent, and report what broke."""
    rdir = Path(run_dir)
    udir = Path(updated_docs_dir)

    requirements = load_requirements(rdir / "requirements.yaml")
    eval_cases = load_eval_cases(rdir / "evals.jsonl")
    agent_path = rdir / "agent_final.yaml"
    if not agent_path.exists():
        agent_path = rdir / "agent_v0.yaml"
    agent_def = AgentDefinition.from_yaml(agent_path)

    parsed_new, new_inv = parse_folder(udir)
    old_sources = set(requirements.data_sources)
    new_sources = {d["filename"] for d in new_inv.documents} | {p.filename for p in new_inv.csv_profiles}
    doc_changes = {
        "added": sorted(new_sources - old_sources),
        "removed": sorted(old_sources - new_sources),
        "retained": sorted(old_sources & new_sources),
    }

    judge = LayeredJudge(gateway=gateway, grounder=grounder)

    # Evaluate on old index if available; otherwise use new index for baseline
    new_chunks = chunk_documents(
        parsed_new,
        chunk_size=agent_def.retrieval.chunk_size,
        chunk_overlap=agent_def.retrieval.chunk_overlap,
    )
    new_retriever = HybridRetriever(
        embedder=embedder if embedder is not None else TfidfEmbedder(),
        reranker=reranker,
    )
    new_retriever.index(new_chunks)
    new_chunks_by_id = {c.chunk_id: c for c in new_chunks}

    old_index_dir = rdir / "index"
    if (old_index_dir / "chunks.json").exists():
        old_retriever = HybridRetriever.load(old_index_dir, embedder=embedder, reranker=reranker)
        old_chunks_by_id = {c.chunk_id: c for c in old_retriever.chunks}
        _, old_verdicts, old_rate = evaluate_agent_on_cases(
            agent_def, eval_cases, old_retriever, gateway, judge, old_chunks_by_id, udir
        )
    else:
        _, old_verdicts, old_rate = evaluate_agent_on_cases(
            agent_def, eval_cases, new_retriever, gateway, judge, new_chunks_by_id, udir
        )

    target_agent = (
        generate_agent_definition(requirements, new_inv, gateway, agent_id=f"{agent_def.id}_regen")
        if regenerate_agent
        else agent_def
    )
    _, new_verdicts, new_rate = evaluate_agent_on_cases(
        target_agent, eval_cases, new_retriever, gateway, judge, new_chunks_by_id, udir
    )

    newly_broken: list[dict[str, str]] = []
    newly_fixed: list[dict[str, str]] = []
    unchanged_pass = 0
    unchanged_fail = 0

    for case in eval_cases:
        old_ok = old_verdicts[case.id].passed
        new_ok = new_verdicts[case.id].passed
        if old_ok and not new_ok:
            newly_broken.append(
                {"case_id": case.id, "question": case.question, "notes": new_verdicts[case.id].notes}
            )
        elif not old_ok and new_ok:
            newly_fixed.append({"case_id": case.id, "question": case.question})
        elif old_ok and new_ok:
            unchanged_pass += 1
        else:
            unchanged_fail += 1

    report = RegressionReport(
        old_pass_rate=old_rate,
        new_pass_rate=new_rate,
        newly_broken=newly_broken,
        newly_fixed=newly_fixed,
        unchanged_pass=unchanged_pass,
        unchanged_fail=unchanged_fail,
        doc_changes=doc_changes,
    )

    lines = [
        f"# Regression Report — {target_agent.name}",
        "",
        f"- **Previous Pass Rate:** {old_rate:.1%}",
        f"- **Updated Corpus Pass Rate:** {new_rate:.1%} (Δ {(new_rate - old_rate):+.1%})",
        f"- **Added Docs:** {doc_changes['added'] or 'None'}",
        f"- **Removed Docs:** {doc_changes['removed'] or 'None'}",
        f"- **Unchanged Pass / Fail:** {unchanged_pass} / {unchanged_fail}",
        "",
        f"## Newly Broken Cases ({len(newly_broken)})",
    ]
    if newly_broken:
        for b in newly_broken:
            lines.append(f"- **{b['case_id']}**: *{b['question']}* — {b['notes']}")
    else:
        lines.append("- None.")

    lines.extend(["", f"## Newly Fixed Cases ({len(newly_fixed)})"])
    if newly_fixed:
        for fx in newly_fixed:
            lines.append(f"- **{fx['case_id']}**: *{fx['question']}*")
    else:
        lines.append("- None.")

    out_path = rdir / "regression_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, out_path

