"""Deterministic failure classifier for the critic-optimizer loop (Requirement 6)."""

from __future__ import annotations

from forgeai.schemas import AgentResponse, CaseVerdict, EvalCase, FailureClassification


def classify_failure(
    case: EvalCase, response: AgentResponse, verdict: CaseVerdict
) -> FailureClassification:
    """Classify a failed evaluation case into one of the four root-cause categories."""
    if case.should_refuse and not response.refused:
        return FailureClassification(
            case_id=case.id,
            category="hallucination",
            explanation="Scope breach: answered an out-of-scope question that should have been refused.",
        )

    if not case.should_refuse and response.refused:
        if verdict.l1_citation_hit:
            return FailureClassification(
                case_id=case.id,
                category="wrong_refusal",
                explanation="Relevant source chunks were retrieved, but the agent refused unnecessarily.",
            )
        return FailureClassification(
            case_id=case.id,
            category="retrieval_miss",
            explanation="Agent refused because expected source chunks were not retrieved.",
        )

    if not verdict.l1_citation_hit and (case.expected_chunk_ids or case.expected_sources):
        return FailureClassification(
            case_id=case.id,
            category="retrieval_miss",
            explanation="Expected source document or chunk was missing from retrieved context and citations.",
        )

    if verdict.l4_nli_groundedness < 0.5:
        return FailureClassification(
            case_id=case.id,
            category="hallucination",
            explanation=(
                f"Low NLI groundedness ({verdict.l4_nli_groundedness:.2f}): answer contains claims "
                "not entailed by retrieved evidence."
            ),
        )

    if verdict.l3_tool_ok is False:
        return FailureClassification(
            case_id=case.id,
            category="prompt_defect",
            explanation=f"Expected tool '{case.expected_tool}' was not called or failed to produce expected value.",
        )

    return FailureClassification(
        case_id=case.id,
        category="prompt_defect",
        explanation=(
            f"Context retrieved, but answer missed key facts (L2={verdict.l2_fact_coverage:.2f}, "
            f"L5={verdict.l5_llm_correctness:.2f})."
        ),
    )


def classify_failures(
    cases: list[EvalCase],
    responses_by_id: dict[str, AgentResponse],
    verdicts_by_id: dict[str, CaseVerdict],
) -> list[FailureClassification]:
    """Classify every failed evaluation case in ``cases``."""
    results: list[FailureClassification] = []
    for case in cases:
        verdict = verdicts_by_id.get(case.id)
        response = responses_by_id.get(case.id)
        if verdict is None or response is None or verdict.passed:
            continue
        results.append(classify_failure(case, response, verdict))
    return results

