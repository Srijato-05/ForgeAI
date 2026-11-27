"""Groundability guard verifying that must_answer topics are backed by documents (Requirement 2)."""

from __future__ import annotations

from pydantic import BaseModel

from forgeai.factory.judge import LexicalNLIGrounder, NLIGrounder
from forgeai.ingest import HybridRetriever
from forgeai.llm import LLMGateway, call_structured
from forgeai.prompts import load_prompt
from forgeai.schemas import (
    GroundabilityStatus,
    RequirementsDoc,
    RetrievalConfig,
    TopicGroundability,
)


class _ProbeVerdict(BaseModel):
    status: GroundabilityStatus
    notes: str = ""


def run_groundability_guard(
    requirements: RequirementsDoc,
    retriever: HybridRetriever,
    gateway: LLMGateway,
    grounder: NLIGrounder | None = None,
) -> RequirementsDoc:
    """Probe every must_answer topic against the corpus and flag/move unsupported gaps."""
    probe_prompt = load_prompt("groundability_probe")
    nli = grounder or LexicalNLIGrounder()

    supported_topics: list[str] = []
    must_refuse = list(requirements.must_refuse)
    known_gaps = list(requirements.known_gaps)
    checks: list[TopicGroundability] = []

    probe_cfg = RetrievalConfig(top_k=5, rerank_top_n=3, min_relevance_score=0.05)

    for topic in requirements.must_answer:
        hits = retriever.search(topic, probe_cfg)
        if not hits:
            checks.append(
                TopicGroundability(
                    topic=topic,
                    status="unsupported",
                    evidence_chunks=[],
                    notes="No relevant chunks retrieved from corpus.",
                )
            )
            if topic not in known_gaps:
                known_gaps.append(topic)
            if topic not in must_refuse:
                must_refuse.append(topic)
            continue

        evidence_ids = [h.chunk.chunk_id for h in hits]
        context_block = "\n\n".join(
            f"[{h.chunk.chunk_id}] ({h.chunk.source_doc}): {h.chunk.raw_text}" for h in hits
        )
        nli_score = nli.score_groundedness(context_block, topic)

        verdict = call_structured(
            gateway,
            role="judge",
            messages=[
                {"role": "system", "content": probe_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\n"
                        f"Lexical/NLI pre-score: {nli_score:.2f}\n\n"
                        f"Retrieved Chunks:\n{context_block}"
                    ),
                },
            ],
            schema=_ProbeVerdict,
            node="groundability_guard",
        )

        checks.append(
            TopicGroundability(
                topic=topic,
                status=verdict.status,
                evidence_chunks=evidence_ids,
                notes=verdict.notes,
            )
        )

        if verdict.status == "unsupported":
            if topic not in known_gaps:
                known_gaps.append(topic)
            if topic not in must_refuse:
                must_refuse.append(topic)
        else:
            supported_topics.append(topic)

    return requirements.model_copy(
        update={
            "must_answer": supported_topics,
            "must_refuse": must_refuse,
            "known_gaps": known_gaps,
            "groundability_checks": checks,
        }
    )

