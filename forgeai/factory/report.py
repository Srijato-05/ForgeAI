"""Honest scorecard and quality report generator (Requirement 7)."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from forgeai.factory.classifier import classify_failures
from forgeai.factory.evalgen import EvalGenerationReport

if TYPE_CHECKING:
    from forgeai.factory.tournament import TournamentResult
from forgeai.llm import TraceLogger
from forgeai.schemas import (
    AgentDefinition,
    AgentResponse,
    CaseVerdict,
    EvalCase,
    IterationLog,
    RequirementsDoc,
)


def wilson_ci_95(successes: int, n: int) -> tuple[float, float]:
    """Compute the 95% Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959964
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def _category_breakdown(
    cases: list[EvalCase], verdicts: dict[str, CaseVerdict]
) -> dict[str, tuple[int, int]]:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c in cases:
        v = verdicts.get(c.id)
        if v is None:
            continue
        stats[c.category][1] += 1
        if v.passed:
            stats[c.category][0] += 1
    return {k: (v[0], v[1]) for k, v in sorted(stats.items())}


def generate_scorecard_and_report(
    run_dir: Path,
    requirements: RequirementsDoc,
    final_agent: AgentDefinition,
    eval_cases: list[EvalCase],
    iterations: list[IterationLog],
    dev_verdicts: dict[str, CaseVerdict],
    holdout_verdicts: dict[str, CaseVerdict],
    holdout_responses: dict[str, AgentResponse],
    eval_report: EvalGenerationReport | None = None,
    tracer: TraceLogger | None = None,
    kappa: float | None = None,
    tournament: TournamentResult | None = None,
) -> tuple[Path, Path]:
    """Write ``scorecard.md`` and ``report.md`` into ``run_dir``."""
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_cases = [c for c in eval_cases if c.split == "dev"]
    holdout_cases = [c for c in eval_cases if c.split == "holdout"]

    dev_pass = sum(1 for c in dev_cases if dev_verdicts.get(c.id) and dev_verdicts[c.id].passed)
    hold_pass = sum(
        1 for c in holdout_cases if holdout_verdicts.get(c.id) and holdout_verdicts[c.id].passed
    )
    hold_rate = (hold_pass / len(holdout_cases)) if holdout_cases else 0.0
    ci_low, ci_high = wilson_ci_95(hold_pass, len(holdout_cases))

    # 1. Scorecard
    sc_lines = [
        f"# ForgeAI Scorecard — {final_agent.name} (v{final_agent.version})",
        "",
        "## Optimization Iterations (Dev Split)",
        "",
        "| Iter | Target Field | Score Before | Score After | Regressions | Accepted | Stop Reason |",
        "|---:|---|---:|---:|---:|:---:|---|",
    ]
    for it in iterations:
        field = it.patch.target_field if it.patch else "—"
        sc_lines.append(
            f"| {it.iteration} | `{field}` | {it.score_before:.1%} | {it.score_after:.1%} | "
            f"{it.regressions} | {'yes' if it.accepted else 'no'} | {it.stop_reason or ''} |"
        )

    sc_lines.extend(
        [
            "",
            "## Category Breakdown",
            "",
            "| Category | Dev Pass | Holdout Pass |",
            "|---|---|---|",
        ]
    )
    dev_cats = _category_breakdown(dev_cases, dev_verdicts)
    hold_cats = _category_breakdown(holdout_cases, holdout_verdicts)
    all_cats = sorted(set(dev_cats) | set(hold_cats))
    for cat in all_cats:
        dp, dn = dev_cats.get(cat, (0, 0))
        hp, hn = hold_cats.get(cat, (0, 0))
        d_str = f"{dp}/{dn} ({dp / dn:.0%})" if dn else "—"
        h_str = f"{hp}/{hn} ({hp / hn:.0%})" if hn else "—"
        sc_lines.append(f"| {cat} | {d_str} | {h_str} |")

    scorecard_path = out_dir / "scorecard.md"
    scorecard_path.write_text("\n".join(sc_lines) + "\n", encoding="utf-8")

    # 2. Honest Quality Report
    all_cases_by_id = {c.id: c for c in eval_cases}
    failures = classify_failures(holdout_cases, holdout_responses, holdout_verdicts)
    grouped_failures: dict[str, list] = defaultdict(list)
    for f in failures:
        grouped_failures[f.category].append(f)

    rp_lines = [
        f"# ForgeAI Honest Quality Report — {final_agent.name}",
        "",
        "## Headline Summary",
        f"- **Final Agent Version:** `v{final_agent.version}` (`{final_agent.id}`)",
        f"- **Dev Pass Rate:** {dev_pass}/{len(dev_cases)} ({(dev_pass / len(dev_cases)):.1%})" if dev_cases else "- **Dev Pass Rate:** N/A",
        (
            f"- **Holdout Pass Rate:** {hold_pass}/{len(holdout_cases)} ({hold_rate:.1%}) "
            f"— **Wilson 95% CI:** [{ci_low:.1%}, {ci_high:.1%}]"
            if holdout_cases
            else "- **Holdout Pass Rate:** No holdout cases generated."
        ),
        "",
        "## What Still Fails and Why",
    ]
    if not failures:
        rp_lines.append("- No holdout failures observed in this run (note sample size and CI width above).")
    else:
        for cat, items in sorted(grouped_failures.items()):
            rp_lines.append(f"### `{cat}` ({len(items)} case(s))")
            for item in items:
                q_text = all_cases_by_id[item.case_id].question if item.case_id in all_cases_by_id else ""
                rp_lines.append(f"- **{item.case_id}** — *{q_text}*: {item.explanation}")

    rp_lines.extend(["", "## Groundability Gaps & Unattended Assumptions"])
    if requirements.known_gaps:
        for gap in requirements.known_gaps:
            rp_lines.append(f"- **Unsupported Topic (moved to refuse):** {gap}")
    else:
        rp_lines.append("- None detected.")
    if requirements.assumptions:
        for assumption in requirements.assumptions:
            rp_lines.append(f"- **Assumption:** {assumption}")

    rp_lines.extend(["", "## Tool Selection Audit"])
    if final_agent.tools:
        for t in final_agent.tools:
            rp_lines.append(f"- **Attached `{t.name}`:** {t.justification}")
    else:
        rp_lines.append("- **Attached:** None")
    for r in final_agent.tools_rejected:
        rp_lines.append(f"- **Rejected `{r.name}`:** {r.reason}")

    if eval_report is not None:
        rp_lines.extend(
            [
                "",
                "## Eval Synthesis & Distrust Filter Audit",
                f"- Candidates generated: {eval_report.candidates_generated}, Kept: {eval_report.kept}, Dropped: {len(eval_report.dropped)}",
            ]
        )

    if tournament is not None:
        rp_lines.extend(
            [
                "",
                "## Prompt Tournament Comparison",
                f"- **Winner:** `{tournament.winner_label}`",
                "",
                "| Candidate | Strategy | Dev Slice Pass Rate | Pairwise Wins |",
                "|---|---|---:|---:|",
            ]
        )
        for cand in tournament.candidates:
            rp_lines.append(
                f"| `{cand.label}` | {cand.strategy} | {cand.dev_pass_rate:.1%} | {cand.pairwise_wins} |"
            )

    rp_lines.extend(["", "## Judge Calibration & Budget Usage"])
    rp_lines.append(
        f"- **Judge-Human Agreement (Cohen's κ):** {kappa:.3f}"
        if kappa is not None
        else "- **Judge-Human Agreement (Cohen's κ):** Not calibrated in this unattended run."
    )
    if tracer is not None:
        rp_lines.append(
            f"- **LLM Calls:** {tracer.call_count} | **Total Tokens:** {tracer.total_tokens} | "
            f"**Cumulative Call Latency:** {tracer.total_latency_ms / 1000.0:.2f}s"
        )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(rp_lines) + "\n", encoding="utf-8")
    return scorecard_path, report_path

