"""LangGraph state machine wiring the ForgeAI core factory stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from forgeai.config import BudgetConfig
from forgeai.factory.evalgen import EvalGenerationReport, generate_eval_set, save_eval_cases
from forgeai.factory.generator import generate_agent_definition
from forgeai.factory.guard import run_groundability_guard
from forgeai.factory.intake import load_requirements, run_intake, save_requirements
from forgeai.factory.judge import LayeredJudge, NLIGrounder
from forgeai.factory.optimizer import evaluate_agent_on_cases, run_optimization_loop
from forgeai.factory.report import generate_scorecard_and_report
from forgeai.factory.tournament import TournamentResult, run_prompt_tournament
from forgeai.ingest import (
    Chunk,
    DocumentInventory,
    Embedder,
    HybridRetriever,
    Reranker,
    TfidfEmbedder,
    chunk_documents,
    parse_folder,
)
from forgeai.llm import LLMGateway, TraceLogger
from forgeai.schemas import (
    AgentDefinition,
    AgentResponse,
    CaseVerdict,
    ClarifyingQuestion,
    EvalCase,
    IterationLog,
    RequirementsDoc,
)
from forgeai.tools import ToolRegistry


class FactoryState(TypedDict, total=False):
    """Typed state flowing through the ForgeAI LangGraph pipeline."""

    brief: str
    docs_dir: str
    run_dir: str
    unattended: bool
    target_eval_size: int
    inventory: DocumentInventory
    chunks: list[Chunk]
    retriever_index_dir: str
    requirements: RequirementsDoc
    agent_def: AgentDefinition
    final_agent: AgentDefinition
    eval_cases: list[EvalCase]
    eval_report: EvalGenerationReport
    tournament: TournamentResult
    iterations: list[IterationLog]
    dev_verdicts: dict[str, CaseVerdict]
    holdout_verdicts: dict[str, CaseVerdict]
    holdout_responses: dict[str, AgentResponse]
    scorecard_path: str
    report_path: str


def build_factory_graph(
    gateway: LLMGateway,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    grounder: NLIGrounder | None = None,
    ask_user: Callable[[ClarifyingQuestion], str] | None = None,
    registry: ToolRegistry | None = None,
    checkpointer: Any = None,
    interrupt_for_human: bool = False,
    run_optimizer: bool = False,
    enable_tournament: bool = False,
    budget: BudgetConfig | None = None,
    tracer: TraceLogger | None = None,
):
    """Compile the core LangGraph factory graph up to v0 generation and eval synthesis."""
    reg = registry or ToolRegistry()

    def _make_embedder() -> Embedder:
        return embedder if embedder is not None else TfidfEmbedder()

    def ingest_node(state: FactoryState) -> FactoryState:
        docs_dir = Path(state["docs_dir"])
        run_dir = Path(state["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        parsed_docs, inventory = parse_folder(docs_dir)
        chunks = chunk_documents(parsed_docs)

        index_dir = run_dir / "index"
        retriever = HybridRetriever(embedder=_make_embedder(), reranker=reranker)
        retriever.index(chunks)
        retriever.save(index_dir)

        return {
            "inventory": inventory,
            "chunks": chunks,
            "retriever_index_dir": str(index_dir),
        }

    def intake_node(state: FactoryState) -> FactoryState:
        run_dir = Path(state["run_dir"])
        req = run_intake(
            brief=state["brief"],
            inventory=state["inventory"],
            gateway=gateway,
            ask_user=ask_user,
            unattended=state.get("unattended", True),
        )
        save_requirements(req, run_dir / "requirements.yaml")
        return {"requirements": req}

    def guard_node(state: FactoryState) -> FactoryState:
        run_dir = Path(state["run_dir"])
        retriever = HybridRetriever.load(
            Path(state["retriever_index_dir"]),
            embedder=_make_embedder() if embedder is not None else None,
            reranker=reranker,
        )
        checked = run_groundability_guard(
            requirements=state["requirements"],
            retriever=retriever,
            gateway=gateway,
            grounder=grounder,
        )
        save_requirements(checked, run_dir / "requirements.yaml")
        return {"requirements": checked}

    def human_checkpoint_node(state: FactoryState) -> FactoryState:
        req_path = Path(state["run_dir"]) / "requirements.yaml"
        if req_path.exists():
            return {"requirements": load_requirements(req_path)}
        return {}

    def generate_agent_node(state: FactoryState) -> FactoryState:
        run_dir = Path(state["run_dir"])
        agent_def = generate_agent_definition(
            requirements=state["requirements"],
            inventory=state["inventory"],
            gateway=gateway,
            registry=reg,
            agent_id=f"{run_dir.name}_v0",
        )
        agent_def.to_yaml(run_dir / "agent_v0.yaml")
        return {"agent_def": agent_def}

    def generate_evals_node(state: FactoryState) -> FactoryState:
        run_dir = Path(state["run_dir"])
        retriever = HybridRetriever.load(
            Path(state["retriever_index_dir"]),
            embedder=_make_embedder() if embedder is not None else None,
            reranker=reranker,
        )
        cases, report = generate_eval_set(
            requirements=state["requirements"],
            chunks=state["chunks"],
            inventory=state["inventory"],
            retriever=retriever,
            gateway=gateway,
            target_size=state.get("target_eval_size", 24),
        )
        save_eval_cases(cases, run_dir / "evals.jsonl")
        return {"eval_cases": cases, "eval_report": report}

    def tournament_node(state: FactoryState) -> FactoryState:
        retriever = HybridRetriever.load(
            Path(state["retriever_index_dir"]),
            embedder=_make_embedder() if embedder is not None else None,
            reranker=reranker,
        )
        judge = LayeredJudge(gateway=gateway, grounder=grounder)
        chunks_by_id = {c.chunk_id: c for c in state["chunks"]}
        eval_cases = state["eval_cases"]
        dev_cases = [c for c in eval_cases if c.split == "dev"] or eval_cases

        result = run_prompt_tournament(
            base_agent=state["agent_def"],
            requirements=state["requirements"],
            dev_cases=dev_cases,
            retriever=retriever,
            gateway=gateway,
            judge=judge,
            chunks_by_id=chunks_by_id,
            docs_dir=Path(state["docs_dir"]),
            registry=reg,
        )
        result.winner_agent.to_yaml(Path(state["run_dir"]) / "agent_v0.yaml")
        return {"agent_def": result.winner_agent, "tournament": result}

    def optimize_node(state: FactoryState) -> FactoryState:
        run_dir = Path(state["run_dir"])
        docs_dir = Path(state["docs_dir"])
        retriever = HybridRetriever.load(
            Path(state["retriever_index_dir"]),
            embedder=_make_embedder() if embedder is not None else None,
            reranker=reranker,
        )
        judge = LayeredJudge(gateway=gateway, grounder=grounder)
        chunks_by_id = {c.chunk_id: c for c in state["chunks"]}
        eval_cases = state["eval_cases"]
        dev_cases = [c for c in eval_cases if c.split == "dev"] or eval_cases

        final_agent, iterations, dev_verdicts, _ = run_optimization_loop(
            agent_def=state["agent_def"],
            dev_cases=dev_cases,
            retriever=retriever,
            gateway=gateway,
            judge=judge,
            chunks_by_id=chunks_by_id,
            docs_dir=docs_dir,
            run_dir=run_dir,
            budget=budget,
            tracer=tracer,
            registry=reg,
        )
        holdout_cases = [c for c in eval_cases if c.split == "holdout"]
        if holdout_cases:
            hold_resps, hold_verdicts, _ = evaluate_agent_on_cases(
                final_agent, holdout_cases, retriever, gateway, judge, chunks_by_id, docs_dir, reg
            )
        else:
            hold_resps, hold_verdicts = {}, {}

        return {
            "final_agent": final_agent,
            "iterations": iterations,
            "dev_verdicts": dev_verdicts,
            "holdout_verdicts": hold_verdicts,
            "holdout_responses": hold_resps,
        }

    def report_node(state: FactoryState) -> FactoryState:
        sc_path, rp_path = generate_scorecard_and_report(
            run_dir=Path(state["run_dir"]),
            requirements=state["requirements"],
            final_agent=state["final_agent"],
            eval_cases=state["eval_cases"],
            iterations=state["iterations"],
            dev_verdicts=state["dev_verdicts"],
            holdout_verdicts=state["holdout_verdicts"],
            holdout_responses=state["holdout_responses"],
            eval_report=state.get("eval_report"),
            tracer=tracer,
            tournament=state.get("tournament"),
        )
        return {"scorecard_path": str(sc_path), "report_path": str(rp_path)}

    graph = StateGraph(FactoryState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("intake", intake_node)
    graph.add_node("groundability_guard", guard_node)
    graph.add_node("human_checkpoint", human_checkpoint_node)
    graph.add_node("generate_agent", generate_agent_node)
    graph.add_node("generate_evals", generate_evals_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "intake")
    graph.add_edge("intake", "groundability_guard")
    graph.add_edge("groundability_guard", "human_checkpoint")
    graph.add_edge("human_checkpoint", "generate_agent")
    graph.add_edge("generate_agent", "generate_evals")

    if run_optimizer:
        graph.add_node("optimize", optimize_node)
        graph.add_node("report", report_node)
        if enable_tournament:
            graph.add_node("tournament", tournament_node)
            graph.add_edge("generate_evals", "tournament")
            graph.add_edge("tournament", "optimize")
        else:
            graph.add_edge("generate_evals", "optimize")
        graph.add_edge("optimize", "report")
        graph.add_edge("report", END)
    elif enable_tournament:
        graph.add_node("tournament", tournament_node)
        graph.add_edge("generate_evals", "tournament")
        graph.add_edge("tournament", END)
    else:
        graph.add_edge("generate_evals", END)

    if interrupt_for_human:
        saver = checkpointer if checkpointer is not None else MemorySaver()
        return graph.compile(checkpointer=saver, interrupt_before=["human_checkpoint"])
    return graph.compile(checkpointer=checkpointer)

