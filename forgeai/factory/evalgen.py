"""Automatic evaluation set generator with distrust filters and stratified dev/holdout split."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from forgeai.ingest import Chunk, DocumentInventory, HybridRetriever
from forgeai.llm import LLMGateway, StructuredOutputError, call_structured
from forgeai.prompts import load_prompt
from forgeai.schemas import EvalCase, EvalCategory, RequirementsDoc, RetrievalConfig
from forgeai.tools import ToolRegistry


class _GeneratedQuestionItem(BaseModel):
    question: str
    key_facts: list[str] = Field(default_factory=list)
    rationale: str = ""
    expected_tool: str | None = None
    expected_value: str | None = None


class _GeneratedQuestionBatch(BaseModel):
    cases: list[_GeneratedQuestionItem] = Field(default_factory=list)


class _AnswerabilityVerdict(BaseModel):
    answerable: bool
    reason: str = ""


class EvalGenerationReport(BaseModel):
    """Audit summary of candidate generation, filtering, and split counts."""

    candidates_generated: int
    kept: int
    dropped: list[dict[str, str]] = Field(default_factory=list)
    category_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)


def save_eval_cases(cases: list[EvalCase], path: Path) -> None:
    """Persist evaluation cases as JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [c.model_dump_json() for c in cases]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Load evaluation cases from a JSONL file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]


def _has_verbatim_leakage(question: str, chunk_text: str) -> bool:
    q_clean = " ".join(question.lower().split())
    c_clean = " ".join(chunk_text.lower().split())
    if not q_clean or not c_clean:
        return False
    if fuzz.ratio(q_clean, c_clean) >= 85:
        return True
    q_words = q_clean.split()
    if len(q_words) >= 8:
        for i in range(len(q_words) - 7):
            ngram = " ".join(q_words[i : i + 8])
            if ngram in c_clean:
                return True
    return False


def _stratified_split(cases: list[EvalCase], holdout_ratio: float, seed: int) -> list[EvalCase]:
    rng = random.Random(seed)
    by_cat: dict[EvalCategory, list[EvalCase]] = defaultdict(list)
    for case in cases:
        by_cat[case.category].append(case)

    split_cases: list[EvalCase] = []
    for cat in sorted(by_cat.keys()):
        group = list(by_cat[cat])
        rng.shuffle(group)
        if len(group) >= 2 and holdout_ratio > 0:
            n_holdout = max(1, min(len(group) - 1, round(len(group) * holdout_ratio)))
        else:
            n_holdout = 0
        for idx, case in enumerate(group):
            split_name = "holdout" if idx < n_holdout else "dev"
            split_cases.append(case.model_copy(update={"split": split_name}))
    return split_cases


def generate_eval_set(
    requirements: RequirementsDoc,
    chunks: list[Chunk],
    inventory: DocumentInventory,
    retriever: HybridRetriever,
    gateway: LLMGateway,
    target_size: int = 24,
    holdout_ratio: float = 0.35,
    seed: int = 42,
    enable_answerability_filter: bool = True,
) -> tuple[list[EvalCase], EvalGenerationReport]:
    """Generate, filter, and split a document-grounded evaluation set."""
    rng = random.Random(seed)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.source_doc].append(chunk)

    candidates: list[EvalCase] = []
    dropped: list[dict[str, str]] = []
    idx_counter = 1

    def _next_id() -> str:
        nonlocal idx_counter
        val = f"eval_{idx_counter:03d}"
        idx_counter += 1
        return val

    # 1. Single-hop candidates (stratified across documents)
    single_prompt = load_prompt("eval_single_hop")
    selected_chunks: list[Chunk] = []
    doc_names = sorted(by_doc.keys())
    if doc_names:
        per_doc_pools = {d: list(by_doc[d]) for d in doc_names}
        for d in per_doc_pools:
            rng.shuffle(per_doc_pools[d])
        want_chunks = min(len(chunks), max(4, target_size // 2))
        while len(selected_chunks) < want_chunks and any(per_doc_pools.values()):
            for d in doc_names:
                if per_doc_pools[d] and len(selected_chunks) < want_chunks:
                    selected_chunks.append(per_doc_pools[d].pop())

    single_parents: list[tuple[EvalCase, Chunk]] = []
    per_chunk_q = max(2, min(4, math.ceil(target_size / max(1, len(selected_chunks)))))
    for chunk in selected_chunks:
        batch = call_structured(
            gateway,
            role="factory",
            messages=[
                {"role": "system", "content": single_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Requirements Scope: {requirements.scope}\n"
                        f"Must-Answer Topics: {requirements.must_answer}\n\n"
                        f"Source Chunk [{chunk.chunk_id}] ({chunk.source_doc} / {chunk.section_title}):\n"
                        f"{chunk.raw_text}\n\nGenerate {per_chunk_q} distinct single-hop questions."
                    ),
                },
            ],
            schema=_GeneratedQuestionBatch,
            node="evalgen_single_hop",
        )
        for item in batch.cases[:per_chunk_q]:
            case = EvalCase(
                id=_next_id(),
                category="single_hop",
                split="dev",
                question=item.question.strip(),
                expected_sources=[chunk.source_doc],
                expected_chunk_ids=[chunk.chunk_id],
                key_facts=item.key_facts,
                should_refuse=False,
                rationale=item.rationale,
            )
            candidates.append(case)
            single_parents.append((case, chunk))

    # 2. Multi-doc synthesis candidates
    if len(doc_names) >= 2:
        multi_prompt = load_prompt("eval_multi_doc")
        n_pairs = min(4, max(2, target_size // 8))
        for pair_i in range(n_pairs):
            d1 = doc_names[pair_i % len(doc_names)]
            d2 = doc_names[(pair_i + 1) % len(doc_names)]
            c1 = rng.choice(by_doc[d1])
            c2 = rng.choice(by_doc[d2])
            try:
                batch = call_structured(
                    gateway,
                    role="factory",
                    messages=[
                        {"role": "system", "content": multi_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Chunk A [{c1.chunk_id}] ({c1.source_doc}):\n{c1.raw_text}\n\n"
                                f"Chunk B [{c2.chunk_id}] ({c2.source_doc}):\n{c2.raw_text}\n\n"
                                "Generate 1 multi-doc synthesis question."
                            ),
                        },
                    ],
                    schema=_GeneratedQuestionBatch,
                    node="evalgen_multi_doc",
                )
            except StructuredOutputError:
                continue
            for item in batch.cases[:1]:
                candidates.append(
                    EvalCase(
                        id=_next_id(),
                        category="multi_doc",
                        split="dev",
                        question=item.question.strip(),
                        expected_sources=[c1.source_doc, c2.source_doc],
                        expected_chunk_ids=[c1.chunk_id, c2.chunk_id],
                        key_facts=item.key_facts,
                        should_refuse=False,
                        rationale=item.rationale,
                    )
                )

    # 3. Ambiguous / colloquial phrasing edge cases
    ambig_prompt = load_prompt("eval_ambiguous")
    for parent_case, parent_chunk in single_parents[: max(2, target_size // 8)]:
        try:
            batch = call_structured(
                gateway,
                role="factory",
                messages=[
                    {"role": "system", "content": ambig_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Original Question: {parent_case.question}\n"
                            f"Key Facts: {parent_case.key_facts}\n"
                            f"Source Chunk: {parent_chunk.raw_text}\n\n"
                            "Generate 1 colloquial or ambiguously phrased version."
                        ),
                    },
                ],
                schema=_GeneratedQuestionBatch,
                node="evalgen_ambiguous",
            )
        except StructuredOutputError:
            continue
        for item in batch.cases[:1]:
            candidates.append(
                EvalCase(
                    id=_next_id(),
                    category="ambiguous",
                    split="dev",
                    question=item.question.strip(),
                    expected_sources=parent_case.expected_sources,
                    expected_chunk_ids=parent_case.expected_chunk_ids,
                    key_facts=item.key_facts or parent_case.key_facts,
                    should_refuse=False,
                    rationale=item.rationale or f"Ambiguous paraphrase of {parent_case.id}",
                )
            )

    # 4. Out-of-scope / must-refuse candidates
    oos_prompt = load_prompt("eval_out_of_scope")
    oos_count = max(4, target_size // 4)
    oos_batch = call_structured(
        gateway,
        role="factory",
        messages=[
            {"role": "system", "content": oos_prompt},
            {
                "role": "user",
                "content": (
                    f"Scope: {requirements.scope}\n"
                    f"Must Refuse Topics: {requirements.must_refuse}\n"
                    f"Known Gaps: {requirements.known_gaps}\n"
                    f"Document Inventory: {json.dumps(inventory.documents, ensure_ascii=False)}\n\n"
                    f"Generate {oos_count} out-of-scope questions that must be refused."
                ),
            },
        ],
        schema=_GeneratedQuestionBatch,
        node="evalgen_oos",
    )
    for item in oos_batch.cases[:oos_count]:
        candidates.append(
            EvalCase(
                id=_next_id(),
                category="out_of_scope",
                split="dev",
                question=item.question.strip(),
                expected_sources=[],
                expected_chunk_ids=[],
                key_facts=[],
                should_refuse=True,
                rationale=item.rationale or "Out-of-scope test",
            )
        )

    # 5. Tool-required candidates (when eligible tools exist)
    eligible_tools, _ = ToolRegistry().check_eligibility(inventory, requirements.brief)
    eligible_names = [t.name for t, _ in eligible_tools if t.name != "weather_api"]
    if eligible_names:
        tool_prompt = load_prompt("eval_tool_required")
        csv_info = [p.model_dump(mode="json") for p in inventory.csv_profiles]
        tool_batch = call_structured(
            gateway,
            role="factory",
            messages=[
                {"role": "system", "content": tool_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Eligible Tools: {eligible_names}\n"
                        f"CSV Profiles: {json.dumps(csv_info, ensure_ascii=False)}\n"
                        "Generate 3 tool-required questions."
                    ),
                },
            ],
            schema=_GeneratedQuestionBatch,
            node="evalgen_tool",
        )
        for item in tool_batch.cases[:3]:
            candidates.append(
                EvalCase(
                    id=_next_id(),
                    category="tool_required",
                    split="dev",
                    question=item.question.strip(),
                    expected_sources=[p.filename for p in inventory.csv_profiles],
                    expected_chunk_ids=[],
                    key_facts=item.key_facts,
                    should_refuse=False,
                    expected_tool=item.expected_tool or eligible_names[0],
                    expected_value=item.expected_value,
                    rationale=item.rationale,
                )
            )

    # Distrust Filters
    answerability_prompt = load_prompt("eval_answerability_filter")
    kept: list[EvalCase] = []
    seen_questions: list[str] = []

    for case in candidates:
        q = case.question
        if not q:
            dropped.append({"id": case.id, "question": q, "reason": "empty_question"})
            continue

        # Filter 1: Verbatim leakage check on answerable questions
        if not case.should_refuse and case.expected_chunk_ids:
            source_text = " ".join(
                chunks_by_id[cid].raw_text for cid in case.expected_chunk_ids if cid in chunks_by_id
            )
            if _has_verbatim_leakage(q, source_text):
                dropped.append({"id": case.id, "question": q, "reason": "verbatim_leakage"})
                continue

        # Filter 2: Deduplication
        if any(fuzz.token_set_ratio(q.lower(), prev.lower()) >= 85 for prev in seen_questions):
            dropped.append({"id": case.id, "question": q, "reason": "duplicate"})
            continue

        # Filter 3: Independent answerability verification (via judge role)
        if enable_answerability_filter and not case.should_refuse and case.expected_chunk_ids:
            chunk_texts = "\n\n".join(
                f"[{cid}]: {chunks_by_id[cid].raw_text}"
                for cid in case.expected_chunk_ids
                if cid in chunks_by_id
            )
            if chunk_texts:
                verdict = call_structured(
                    gateway,
                    role="judge",
                    messages=[
                        {"role": "system", "content": answerability_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Question: {q}\n"
                                f"Key Facts: {case.key_facts}\n\n"
                                f"Source Chunk(s):\n{chunk_texts}"
                            ),
                        },
                    ],
                    schema=_AnswerabilityVerdict,
                    node="evalgen_answerability_filter",
                )
                if not verdict.answerable:
                    dropped.append(
                        {"id": case.id, "question": q, "reason": f"unanswerable: {verdict.reason}"}
                    )
                    continue

        # Filter 4: Out-of-scope verification against retriever
        if case.should_refuse and chunks:
            hits = retriever.search(
                q, RetrievalConfig(top_k=3, rerank_top_n=1, min_relevance_score=0.0)
            )
            if hits:
                top_hit = hits[0]
                overlap = fuzz.token_set_ratio(q.lower(), top_hit.chunk.raw_text.lower())
                if top_hit.score >= 0.80 and overlap >= 75:
                    dropped.append(
                        {"id": case.id, "question": q, "reason": "oos_actually_covered_in_corpus"}
                    )
                    continue

        seen_questions.append(q)
        kept.append(case)

    kept = kept[:target_size]
    final_cases = _stratified_split(kept, holdout_ratio=holdout_ratio, seed=seed)

    cat_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    for c in final_cases:
        cat_counts[c.category] += 1
        split_counts[c.split] += 1

    report = EvalGenerationReport(
        candidates_generated=len(candidates),
        kept=len(final_cases),
        dropped=dropped,
        category_counts=dict(cat_counts),
        split_counts=dict(split_counts),
    )
    return final_cases, report

