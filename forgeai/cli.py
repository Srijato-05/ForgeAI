"""ForgeAI command-line interface."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgeai.config import BudgetConfig, ForgeSettings, ModelRole, ProviderName
from forgeai.factory import (
    CrossEncoderNLIGrounder,
    LayeredJudge,
    LexicalNLIGrounder,
    build_factory_graph,
    compare_factory_vs_human,
    evaluate_agent_on_cases,
    load_eval_cases,
    run_regression,
)
from forgeai.ingest import (
    CrossEncoderReranker,
    HybridRetriever,
    SentenceTransformerEmbedder,
    TfidfEmbedder,
    chunk_documents,
    parse_folder,
)
from forgeai.llm import LLMGateway, TraceLogger
from forgeai.runtime import AgentRuntime
from forgeai.schemas import AgentDefinition, ClarifyingQuestion

app = typer.Typer(name="forge", help="ForgeAI — Autonomous Agent Factory CLI", no_args_is_help=True)
console = Console()


def _mask_secret(secret: str) -> str:
    if not secret:
        return "[red]missing[/red]"
    if secret == "ollama":
        return "[green]not required[/green]"
    if len(secret) <= 8:
        return "[green]set (****)[/green]"
    return f"[green]{secret[:4]}...{secret[-4:]}[/green]"


@app.callback()
def main() -> None:
    """ForgeAI CLI root command."""


@app.command()
def doctor(
    ping: bool = typer.Option(
        False, "--ping/--no-ping", help="Verify live endpoint connectivity and configured model IDs via /models."
    ),
) -> None:
    """Display active provider, configured models per role, API key status, and cache path."""
    settings = ForgeSettings()
    provider = settings.forge_provider

    table = Table(title="ForgeAI Environment Diagnostics")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Active Provider", provider)
    table.add_row("Base URL", settings.get_base_url(provider))
    table.add_row("API Key", _mask_secret(settings.get_api_key(provider)))

    roles: tuple[ModelRole, ...] = ("factory", "agent", "judge")
    for role in roles:
        table.add_row(f"Model ({role})", settings.get_model(role, provider))

    table.add_row("Cache Directory", str(settings.cache_dir.resolve()))
    table.add_row("Runs Directory", str(settings.runs_dir.resolve()))
    table.add_row("Rate Limits (RPM / TPM)", f"{settings.rpm_limit} / {settings.tpm_limit}")

    if ping:
        base_url = settings.get_base_url(provider).rstrip("/")
        api_key = settings.get_api_key(provider)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key and api_key != "ollama" else {}
        try:
            resp = httpx.get(f"{base_url}/models", headers=headers, timeout=10.0)
            resp.raise_for_status()
            available = {m.get("id", "") for m in resp.json().get("data", [])}
            table.add_row("Endpoint Ping", f"[green]OK ({len(available)} models)[/green]")
            for role in roles:
                model_id = settings.get_model(role, provider)
                status = "[green]available[/green]" if model_id in available else "[yellow]not listed[/yellow]"
                table.add_row(f"Ping ({role})", f"{model_id}: {status}")
        except Exception as exc:
            table.add_row("Endpoint Ping", f"[red]FAILED: {exc}[/red]")

    console.print(table)


@app.command()
def chat(
    agent_yaml: Path = typer.Argument(..., help="Path to a generated agent.yaml file."),
    docs: Path = typer.Option(..., "--docs", "-d", help="Path to the source documents folder."),
    index: Optional[Path] = typer.Option(None, "--index", "-i", help="Optional persisted index directory."),
    provider: Optional[ProviderName] = typer.Option(None, "--provider", "-p", help="Override LLM provider."),
    embedder: Literal["tfidf", "st"] = typer.Option(
        "tfidf", "--embedder", help="Embedder backend ('tfidf' offline or 'st' BAAI/bge-small-en-v1.5)."
    ),
) -> None:
    """Interactive CLI chat runner that loads and executes any produced AgentDefinition."""
    settings = ForgeSettings()
    if provider is not None:
        settings.forge_provider = provider

    agent_def = AgentDefinition.from_yaml(agent_yaml)
    emb = TfidfEmbedder() if embedder == "tfidf" else SentenceTransformerEmbedder()
    reranker = None if embedder == "tfidf" else CrossEncoderReranker()

    if index is not None and (index / "chunks.json").exists():
        retriever = HybridRetriever.load(index, embedder=emb, reranker=reranker)
    else:
        parsed_docs, _ = parse_folder(docs)
        chunks = chunk_documents(
            parsed_docs,
            chunk_size=agent_def.retrieval.chunk_size,
            chunk_overlap=agent_def.retrieval.chunk_overlap,
        )
        retriever = HybridRetriever(embedder=emb, reranker=reranker)
        retriever.index(chunks)
        if index is not None:
            retriever.save(index)

    gateway = LLMGateway(settings=settings)
    runtime = AgentRuntime(agent_def=agent_def, retriever=retriever, gateway=gateway, docs_dir=docs)

    console.print(
        Panel(
            f"[bold]{agent_def.name}[/bold] (v{agent_def.version}) | Provider: {settings.forge_provider}\n"
            f"Tools: {[t.name for t in agent_def.tools] or ['none']}\n"
            "Type 'exit' or 'quit' to stop.",
            title="ForgeAI Chat Runner",
        )
    )

    history: list[dict[str, str]] = []
    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break

        resp = runtime.ask(question, history=history)
        if resp.refused:
            console.print(f"[yellow]Refused:[/yellow] {resp.answer}")
            if resp.refusal_reason:
                console.print(f"[dim]Reason: {resp.refusal_reason}[/dim]")
        else:
            console.print(f"[bold green]Agent:[/bold green] {resp.answer}")
        if resp.citations:
            console.print(f"[dim]Citations: {', '.join(resp.citations)}[/dim]")
        if resp.tool_calls:
            tools_used = [f"{tc['tool']}(ok={tc['ok']})" for tc in resp.tool_calls]
            console.print(f"[dim]Tools: {', '.join(tools_used)}[/dim]")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": resp.answer})


@app.command()
def build(
    brief: str = typer.Option(..., "--brief", "-b", help="One-paragraph brief or path to a brief text file."),
    docs: Path = typer.Option(..., "--docs", "-d", help="Path to the source documents folder."),
    run_dir: Optional[Path] = typer.Option(None, "--run-dir", "-o", help="Output directory for run artifacts."),
    provider: Optional[ProviderName] = typer.Option(None, "--provider", "-p", help="Override LLM provider."),
    embedder: Literal["tfidf", "st"] = typer.Option(
        "tfidf", "--embedder", help="Embedder backend ('tfidf' offline or 'st' BAAI/bge-small-en-v1.5)."
    ),
    unattended: bool = typer.Option(True, "--unattended/--interactive", help="Auto-answer intake questions."),
    max_iterations: int = typer.Option(3, "--max-iterations", min=1, max=5, help="Max optimizer iterations."),
    eval_size: int = typer.Option(24, "--eval-size", min=6, max=60, help="Target eval set size."),
    tournament: bool = typer.Option(False, "--tournament/--no-tournament", help="Run prompt tournament before optimizer."),
    human_checkpoint: bool = typer.Option(
        False, "--human-checkpoint/--no-human-checkpoint", help="Pause after Groundability Guard to review requirements.yaml."
    ),
) -> None:
    """Run the full ForgeAI agent factory end-to-end on a brief and document folder."""
    settings = ForgeSettings()
    if provider is not None:
        settings.forge_provider = provider

    brief_path = Path(brief)
    brief_text = brief_path.read_text(encoding="utf-8").strip() if brief_path.is_file() else brief.strip()

    target_run = run_dir or (
        settings.runs_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    target_run.mkdir(parents=True, exist_ok=True)

    tracer = TraceLogger(target_run / "traces.jsonl")
    gateway = LLMGateway(settings=settings, tracer=tracer)
    emb = TfidfEmbedder() if embedder == "tfidf" else SentenceTransformerEmbedder()
    reranker = None if embedder == "tfidf" else CrossEncoderReranker()
    grounder = LexicalNLIGrounder() if embedder == "tfidf" else CrossEncoderNLIGrounder()
    budget = settings.budget.model_copy(update={"max_iterations": max_iterations})

    def _ask_cli(q: ClarifyingQuestion) -> str:
        console.print(f"\n[bold cyan]Clarifying Question:[/bold cyan] {q.question}")
        console.print(f"[dim]Rationale: {q.rationale} | Default: {q.default_answer}[/dim]")
        return console.input("Your answer (Enter for default): ").strip()

    graph = build_factory_graph(
        gateway=gateway,
        embedder=emb,
        reranker=reranker,
        grounder=grounder,
        ask_user=None if unattended else _ask_cli,
        interrupt_for_human=human_checkpoint,
        run_optimizer=True,
        enable_tournament=tournament,
        budget=budget,
        tracer=tracer,
    )
    init_state = {
        "brief": brief_text,
        "docs_dir": str(docs),
        "run_dir": str(target_run),
        "unattended": unattended,
        "target_eval_size": eval_size,
    }
    if human_checkpoint:
        cfg = {"configurable": {"thread_id": f"build-{target_run.name}"}}
        graph.invoke(init_state, config=cfg)
        console.print(
            Panel(
                f"Paused at Human Checkpoint.\nReview or edit [bold]{target_run / 'requirements.yaml'}[/bold] "
                "and press Enter to continue...",
                title="Human Checkpoint",
            )
        )
        console.input("")
        final_state = graph.invoke(None, config=cfg)
    else:
        final_state = graph.invoke(init_state)

    console.print(
        Panel(
            f"[bold green]Build complete![/bold green]\n"
            f"Run Dir: {target_run}\n"
            f"Final Agent: {target_run / 'agent_final.yaml'}\n"
            f"Scorecard: {final_state.get('scorecard_path')}\n"
            f"Report: {final_state.get('report_path')}",
            title="ForgeAI Build Summary",
        )
    )


@app.command(name="eval")
def eval_cmd(
    agent_yaml: Path = typer.Argument(..., help="Path to agent YAML definition."),
    evals: Path = typer.Option(..., "--evals", "-e", help="Path to evals.jsonl."),
    docs: Path = typer.Option(..., "--docs", "-d", help="Path to source documents folder."),
    index: Optional[Path] = typer.Option(None, "--index", "-i", help="Optional persisted index directory."),
    provider: Optional[ProviderName] = typer.Option(None, "--provider", "-p", help="Override LLM provider."),
    embedder: Literal["tfidf", "st"] = typer.Option("tfidf", "--embedder"),
) -> None:
    """Evaluate any AgentDefinition against an evals.jsonl file."""
    settings = ForgeSettings()
    if provider is not None:
        settings.forge_provider = provider

    agent_def = AgentDefinition.from_yaml(agent_yaml)
    cases = load_eval_cases(evals)
    emb = TfidfEmbedder() if embedder == "tfidf" else SentenceTransformerEmbedder()
    reranker = None if embedder == "tfidf" else CrossEncoderReranker()
    grounder = LexicalNLIGrounder() if embedder == "tfidf" else CrossEncoderNLIGrounder()

    if index is not None and (index / "chunks.json").exists():
        retriever = HybridRetriever.load(index, embedder=emb, reranker=reranker)
    else:
        parsed_docs, _ = parse_folder(docs)
        chunks = chunk_documents(parsed_docs, agent_def.retrieval.chunk_size, agent_def.retrieval.chunk_overlap)
        retriever = HybridRetriever(embedder=emb, reranker=reranker)
        retriever.index(chunks)

    gateway = LLMGateway(settings=settings)
    judge = LayeredJudge(gateway=gateway, grounder=grounder)
    chunks_by_id = {c.chunk_id: c for c in retriever.chunks}
    _, verdicts, pass_rate = evaluate_agent_on_cases(
        agent_def, cases, retriever, gateway, judge, chunks_by_id, docs
    )
    passed = sum(1 for v in verdicts.values() if v.passed)
    console.print(f"[bold]Eval Result:[/bold] {passed}/{len(cases)} passed ({pass_rate:.1%})")


@app.command()
def report(
    run_dir: Path = typer.Argument(..., help="Path to a completed ForgeAI run directory."),
) -> None:
    """Display the scorecard and quality report for a completed run."""
    for fname in ("scorecard.md", "report.md"):
        fpath = run_dir / fname
        if fpath.exists():
            console.print(Panel(fpath.read_text(encoding="utf-8"), title=fname))
        else:
            console.print(f"[yellow]Missing {fpath}[/yellow]")


@app.command()
def regress(
    run_dir: Path = typer.Argument(..., help="Path to an existing completed run directory."),
    docs: Path = typer.Option(..., "--docs", "-d", help="Path to the updated documents folder."),
    regenerate: bool = typer.Option(False, "--regenerate", help="Regenerate agent definition against updated docs."),
    provider: Optional[ProviderName] = typer.Option(None, "--provider", "-p", help="Override LLM provider."),
    embedder: Literal["tfidf", "st"] = typer.Option("tfidf", "--embedder"),
) -> None:
    """Re-run saved evals against updated documents and report what broke."""
    settings = ForgeSettings()
    if provider is not None:
        settings.forge_provider = provider
    emb = TfidfEmbedder() if embedder == "tfidf" else SentenceTransformerEmbedder()
    reranker = None if embedder == "tfidf" else CrossEncoderReranker()
    grounder = LexicalNLIGrounder() if embedder == "tfidf" else CrossEncoderNLIGrounder()
    gateway = LLMGateway(settings=settings)

    rep, out_path = run_regression(
        run_dir=run_dir,
        updated_docs_dir=docs,
        gateway=gateway,
        embedder=emb,
        reranker=reranker,
        grounder=grounder,
        regenerate_agent=regenerate,
    )
    console.print(
        Panel(
            f"Old: {rep.old_pass_rate:.1%} -> New: {rep.new_pass_rate:.1%}\n"
            f"Newly Broken: {len(rep.newly_broken)} | Newly Fixed: {len(rep.newly_fixed)}\n"
            f"Report saved to: {out_path}",
            title="ForgeAI Regression Summary",
        )
    )


@app.command()
def compare(
    factory_yaml: Path = typer.Argument(..., help="Path to ForgeAI-generated agent YAML."),
    human_yaml: Path = typer.Argument(..., help="Path to hand-written baseline agent YAML."),
    evals: Path = typer.Option(..., "--evals", "-e", help="Path to evals.jsonl."),
    docs: Path = typer.Option(..., "--docs", "-d", help="Path to source documents folder."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Optional Markdown output path."),
    provider: Optional[ProviderName] = typer.Option(None, "--provider", "-p", help="Override LLM provider."),
    embedder: Literal["tfidf", "st"] = typer.Option("tfidf", "--embedder"),
) -> None:
    """Compare a ForgeAI agent against a hand-written human baseline on the same evals."""
    settings = ForgeSettings()
    if provider is not None:
        settings.forge_provider = provider
    factory_agent = AgentDefinition.from_yaml(factory_yaml)
    human_agent = AgentDefinition.from_yaml(human_yaml)
    cases = load_eval_cases(evals)

    emb = TfidfEmbedder() if embedder == "tfidf" else SentenceTransformerEmbedder()
    reranker = None if embedder == "tfidf" else CrossEncoderReranker()
    grounder = LexicalNLIGrounder() if embedder == "tfidf" else CrossEncoderNLIGrounder()
    parsed_docs, _ = parse_folder(docs)
    chunks = chunk_documents(parsed_docs)
    retriever = HybridRetriever(embedder=emb, reranker=reranker)
    retriever.index(chunks)
    chunks_by_id = {c.chunk_id: c for c in chunks}

    gateway = LLMGateway(settings=settings)
    judge = LayeredJudge(gateway=gateway, grounder=grounder)
    comp = compare_factory_vs_human(
        factory_agent=factory_agent,
        human_agent=human_agent,
        eval_cases=cases,
        retriever=retriever,
        gateway=gateway,
        judge=judge,
        chunks_by_id=chunks_by_id,
        docs_dir=docs,
        output_path=out,
    )
    console.print(Panel(comp.markdown, title="Factory vs. Human Baseline"))


if __name__ == "__main__":
    app()

