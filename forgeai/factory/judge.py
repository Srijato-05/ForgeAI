"""Layered judge (L0-L3 deterministic, L4 NLI grounding, L5 cross-family LLM) and calibration."""

from __future__ import annotations

import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from forgeai.ingest import Chunk
from forgeai.llm import LLMGateway, call_structured
from forgeai.prompts import load_prompt
from forgeai.schemas import AgentResponse, CaseVerdict, EvalCase

_CITATION_TAG_RE = re.compile(r"\[[a-zA-Z0-9_-]+__c\d+\]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


class NLIGrounder(Protocol):
    """Protocol for sentence-level claim groundedness scorers."""

    def score_groundedness(self, context: str, answer: str) -> float: ...


class LexicalNLIGrounder:
    """Fast offline sentence-level lexical/fuzzy entailment checker."""

    def score_groundedness(self, context: str, answer: str) -> float:
        clean_ans = _CITATION_TAG_RE.sub("", answer).strip()
        if not clean_ans:
            return 0.0
        if not context.strip():
            return 0.0

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_ans) if len(s.strip()) >= 8]
        if not sentences:
            sentences = [clean_ans]

        ctx_lower = context.lower()
        ctx_words = set(_WORD_RE.findall(ctx_lower))
        supported = 0
        for sent in sentences:
            sent_words = [w for w in _WORD_RE.findall(sent.lower()) if len(w) > 2]
            overlap = (
                sum(1 for w in sent_words if w in ctx_words) / len(sent_words)
                if sent_words
                else 0.0
            )
            fuzzy = fuzz.partial_ratio(sent.lower(), ctx_lower)
            if overlap >= 0.45 or fuzzy >= 70:
                supported += 1

        return supported / len(sentences)


class CrossEncoderNLIGrounder:
    """Lazy-loaded cross-encoder NLI entailment scorer (nli-deberta-v3-small)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score_groundedness(self, context: str, answer: str) -> float:
        clean_ans = _CITATION_TAG_RE.sub("", answer).strip()
        if not clean_ans or not context.strip():
            return 0.0
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_ans) if len(s.strip()) >= 8]
        if not sentences:
            sentences = [clean_ans]
        model = self._get_model()
        pairs = [(context[:1800], s) for s in sentences]
        logits = model.predict(pairs)
        # DeBERTa-v3 NLI label mapping typically: 0=contradiction, 1=entailment, 2=neutral
        entailed = 0
        for row in logits:
            if hasattr(row, "__len__") and len(row) >= 3:
                if int(row.argmax()) == 1:
                    entailed += 1
            elif float(row) > 0:
                entailed += 1
        return entailed / len(sentences)


class _FactCheckItem(BaseModel):
    fact: str
    present: bool
    quote: str = ""


class _FactRubricVerdict(BaseModel):
    checks: list[_FactCheckItem] = Field(default_factory=list)


class _PairwiseVerdict(BaseModel):
    winner: Literal["A", "B", "tie"]
    reason: str = ""


class LayeredJudge:
    """Evaluates an AgentResponse against an EvalCase using L0-L5 checks."""

    def __init__(self, gateway: LLMGateway, grounder: NLIGrounder | None = None) -> None:
        self.gateway = gateway
        self.grounder: NLIGrounder = grounder or LexicalNLIGrounder()
        self._rubric_prompt = load_prompt("judge_fact_rubric")

    def _check_l1_citation(
        self, case: EvalCase, response: AgentResponse, chunks_by_id: dict[str, Chunk]
    ) -> bool:
        if case.should_refuse:
            return True
        if not case.expected_chunk_ids and not case.expected_sources:
            return True

        seen_ids = set(response.citations) | set(response.retrieved_chunk_ids)
        if any(cid in seen_ids for cid in case.expected_chunk_ids):
            return True

        seen_sources = {chunks_by_id[cid].source_doc for cid in seen_ids if cid in chunks_by_id}
        if any(src in seen_sources for src in case.expected_sources):
            return True
        return False

    def _check_l2_facts(self, case: EvalCase, response: AgentResponse) -> float:
        if case.should_refuse or not case.key_facts:
            return 1.0
        ans_lower = " ".join(response.answer.lower().split())
        hits = 0
        for fact in case.key_facts:
            f_lower = " ".join(fact.lower().split())
            if f_lower in ans_lower or fuzz.partial_ratio(f_lower, ans_lower) >= 80:
                hits += 1
        return hits / len(case.key_facts)

    def _check_l3_tool(self, case: EvalCase, response: AgentResponse) -> bool | None:
        if not case.expected_tool:
            return None
        matching = [
            tc for tc in response.tool_calls if tc.get("tool") == case.expected_tool and tc.get("ok")
        ]
        if not matching:
            return False
        if case.expected_value:
            val = case.expected_value.strip().lower()
            in_answer = val in response.answer.lower()
            in_tool = any(val in str(tc.get("output", "")).lower() for tc in matching)
            return in_answer or in_tool
        return True

    def evaluate(
        self, case: EvalCase, response: AgentResponse, chunks_by_id: dict[str, Chunk]
    ) -> CaseVerdict:
        """Run L0-L5 scoring on a single evaluation case."""
        l0_ok = bool(response.answer.strip()) and (response.refused == case.should_refuse)
        l1_hit = self._check_l1_citation(case, response, chunks_by_id)
        l2_cov = self._check_l2_facts(case, response)
        l3_ok = self._check_l3_tool(case, response)

        # L4: Claim-level groundedness against retrieved chunks + tool outputs
        if case.should_refuse and response.refused:
            l4_nli = 1.0
        else:
            ctx_parts = [
                chunks_by_id[cid].raw_text
                for cid in response.retrieved_chunk_ids
                if cid in chunks_by_id
            ]
            for tc in response.tool_calls:
                if tc.get("ok") and tc.get("output"):
                    ctx_parts.append(str(tc["output"]))
            l4_nli = self.grounder.score_groundedness("\n".join(ctx_parts), response.answer)

        # L5: Cross-family reference-guided fact rubric
        if not l0_ok:
            l5_llm = 0.0
        elif case.should_refuse or not case.key_facts:
            l5_llm = 1.0
        else:
            rubric = call_structured(
                self.gateway,
                role="judge",
                messages=[
                    {"role": "system", "content": self._rubric_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {case.question}\n"
                            f"Key Facts: {case.key_facts}\n"
                            f"Candidate Answer: {response.answer}"
                        ),
                    },
                ],
                schema=_FactRubricVerdict,
                node="judge_l5_rubric",
            )
            if rubric.checks:
                l5_llm = sum(1 for c in rubric.checks if c.present) / len(rubric.checks)
            else:
                l5_llm = l2_cov

        # Deterministic veto and final score aggregation
        if not l0_ok:
            score = 0.0
            passed = False
            notes = "L0 veto: refusal mismatch or empty answer."
        elif case.should_refuse:
            score = 1.0
            passed = True
            notes = "Correct out-of-scope refusal."
        else:
            if l3_ok is not None:
                score = (
                    0.10 * float(l1_hit)
                    + 0.20 * l2_cov
                    + 0.20 * float(l3_ok)
                    + 0.20 * l4_nli
                    + 0.30 * l5_llm
                )
            else:
                score = 0.15 * float(l1_hit) + 0.25 * l2_cov + 0.25 * l4_nli + 0.35 * l5_llm
            passed = l0_ok and (l3_ok is not False) and (l4_nli >= 0.5) and (score >= 0.70)
            notes = "Pass" if passed else "Failed thresholds"

        return CaseVerdict(
            case_id=case.id,
            passed=passed,
            score=round(score, 4),
            l0_schema_refusal_ok=l0_ok,
            l1_citation_hit=l1_hit,
            l2_fact_coverage=round(l2_cov, 4),
            l3_tool_ok=l3_ok,
            l4_nli_groundedness=round(l4_nli, 4),
            l5_llm_correctness=round(l5_llm, 4),
            notes=notes,
        )


def judge_pairwise_swapped(
    gateway: LLMGateway,
    question: str,
    key_facts: list[str],
    answer_a: str,
    answer_b: str,
) -> dict[str, Any]:
    """Compare two answers in both A/B and B/A orders to neutralize position bias."""
    prompt = load_prompt("judge_pairwise_tournament")

    ab = call_structured(
        gateway,
        role="judge",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nKey Facts: {key_facts}\n\n"
                    f"Response A:\n{answer_a}\n\nResponse B:\n{answer_b}"
                ),
            },
        ],
        schema=_PairwiseVerdict,
        node="judge_pairwise_ab",
    )

    ba = call_structured(
        gateway,
        role="judge",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nKey Facts: {key_facts}\n\n"
                    f"Response A:\n{answer_b}\n\nResponse B:\n{answer_a}"
                ),
            },
        ],
        schema=_PairwiseVerdict,
        node="judge_pairwise_ba",
    )

    ba_mapped = "B" if ba.winner == "A" else ("A" if ba.winner == "B" else "tie")
    consistent = ab.winner == ba_mapped
    final_winner = ab.winner if consistent else "tie"
    return {
        "winner": final_winner,
        "ab_verdict": ab.winner,
        "ba_verdict_mapped": ba_mapped,
        "consistent": consistent,
    }


def compute_cohens_kappa(judge_labels: list[bool], human_labels: list[bool]) -> float:
    """Compute Cohen's kappa inter-rater agreement between judge and human labels."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have equal length.")
    n = len(judge_labels)
    if n == 0:
        return 1.0

    matches = sum(1 for j, h in zip(judge_labels, human_labels) if j == h)
    po = matches / n

    p_j_true = sum(1 for j in judge_labels if j) / n
    p_h_true = sum(1 for h in human_labels if h) / n
    pe = (p_j_true * p_h_true) + ((1.0 - p_j_true) * (1.0 - p_h_true))

    if abs(1.0 - pe) < 1e-9:
        return 1.0 if abs(po - 1.0) < 1e-9 else 0.0
    return round((po - pe) / (1.0 - pe), 4)

