"""Streamlit web UI for ForgeAI (Factory Runner + Agent Chat)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import streamlit as st

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from forgeai.config import ForgeSettings, ProviderName
from forgeai.factory import build_factory_graph
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
from forgeai.schemas import AgentDefinition

st.set_page_config(page_title="ForgeAI — Agent Factory", layout="wide")
st.title("ForgeAI — Autonomous Agent Factory")

settings = ForgeSettings()

with st.sidebar:
    st.header("Model Configuration")
    provider_options: list[ProviderName] = ["groq", "openrouter", "sambanova", "ollama"]
    provider: ProviderName = st.selectbox(
        "Provider",
        provider_options,
        index=provider_options.index(settings.forge_provider),
    )
    api_key = st.text_input(
        f"{provider.upper()} API Key (optional if set in env)",
        type="password",
    )
    embedder_choice = st.selectbox("Embedder Backend", ["tfidf", "st"])
    max_iters = st.slider("Max Optimizer Iterations", min_value=1, max_value=3, value=3)
    eval_size = st.slider("Eval Set Size", min_value=8, max_value=30, value=20)
    enable_tournament = st.checkbox("Enable Prompt Tournament", value=False)

tab_factory, tab_chat = st.tabs(["Factory Runner", "Agent Chat"])

with tab_factory:
    st.subheader("Build a Tested Agent from a Brief + Documents")
    brief = st.text_area(
        "One-paragraph brief",
        placeholder="I need an assistant that answers HR policy questions for shore staff...",
        height=110,
    )
    uploaded_files = st.file_uploader(
        "Upload source documents",
        type=["pdf", "md", "txt", "docx", "csv"],
        accept_multiple_files=True,
    )
    run_name = st.text_input(
        "Run name",
        value=f"ui_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )

    if st.button("Run ForgeAI Factory", type="primary", disabled=not (brief.strip() and uploaded_files)):
        run_dir = settings.runs_dir / run_name
        docs_dir = run_dir / "uploaded_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            (docs_dir / uf.name).write_bytes(uf.getvalue())

        run_settings = settings.model_copy(update={"forge_provider": provider})
        if api_key.strip():
            key_field = f"{provider}_api_key"
            if hasattr(run_settings, key_field):
                run_settings = run_settings.model_copy(update={key_field: api_key.strip()})

        tracer = TraceLogger(run_dir / "traces.jsonl")
        gateway = LLMGateway(settings=run_settings, tracer=tracer)
        emb = TfidfEmbedder() if embedder_choice == "tfidf" else SentenceTransformerEmbedder()
        reranker = None if embedder_choice == "tfidf" else CrossEncoderReranker()
        budget = run_settings.budget.model_copy(update={"max_iterations": max_iters})

        with st.spinner("Running ForgeAI LangGraph pipeline..."):
            graph = build_factory_graph(
                gateway=gateway,
                embedder=emb,
                reranker=reranker,
                run_optimizer=True,
                enable_tournament=enable_tournament,
                budget=budget,
                tracer=tracer,
            )
            graph.invoke(
                {
                    "brief": brief.strip(),
                    "docs_dir": str(docs_dir),
                    "run_dir": str(run_dir),
                    "unattended": True,
                    "target_eval_size": eval_size,
                }
            )

        st.success(f"Build complete! Artifacts saved to `{run_dir}`")
        col1, col2 = st.columns(2)
        with col1:
            if (run_dir / "scorecard.md").exists():
                st.markdown((run_dir / "scorecard.md").read_text(encoding="utf-8"))
        with col2:
            if (run_dir / "report.md").exists():
                st.markdown((run_dir / "report.md").read_text(encoding="utf-8"))

with tab_chat:
    st.subheader("Chat with Any Produced Agent")
    runs_root = settings.runs_dir
    available_runs = sorted(
        [p for p in runs_root.iterdir() if p.is_dir() and (p / "agent_final.yaml").exists()]
    ) if runs_root.exists() else []

    if not available_runs:
        st.info("No completed runs found under `runs/` yet. Build an agent first or run a demo.")
    else:
        selected_run = st.selectbox("Select Run", available_runs, format_func=lambda p: p.name)
        docs_override = st.text_input(
            "Docs Folder",
            value=str(selected_run / "uploaded_docs")
            if (selected_run / "uploaded_docs").exists()
            else str(Path("demos") / selected_run.name / "docs"),
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_q := st.chat_input("Ask the generated agent a question..."):
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.write(user_q)

            agent_def = AgentDefinition.from_yaml(selected_run / "agent_final.yaml")
            emb = TfidfEmbedder() if embedder_choice == "tfidf" else SentenceTransformerEmbedder()
            reranker = None if embedder_choice == "tfidf" else CrossEncoderReranker()
            idx_dir = selected_run / "index"
            if (idx_dir / "chunks.json").exists():
                retriever = HybridRetriever.load(idx_dir, embedder=emb, reranker=reranker)
            else:
                parsed, _ = parse_folder(Path(docs_override))
                retriever = HybridRetriever(embedder=emb, reranker=reranker)
                retriever.index(chunk_documents(parsed))

            run_settings = settings.model_copy(update={"forge_provider": provider})
            if api_key.strip():
                key_field = f"{provider}_api_key"
                if hasattr(run_settings, key_field):
                    run_settings = run_settings.model_copy(update={key_field: api_key.strip()})

            runtime = AgentRuntime(
                agent_def=agent_def,
                retriever=retriever,
                gateway=LLMGateway(settings=run_settings),
                docs_dir=Path(docs_override),
            )
            resp = runtime.ask(user_q)
            with st.chat_message("assistant"):
                st.write(resp.answer)
                if resp.citations:
                    st.caption(f"Citations: {', '.join(resp.citations)}")
                if resp.tool_calls:
                    st.caption(f"Tool calls: {resp.tool_calls}")
                if resp.refused and resp.refusal_reason:
                    st.warning(f"Refusal reason: {resp.refusal_reason}")
            st.session_state.chat_history.append({"role": "assistant", "content": resp.answer})

