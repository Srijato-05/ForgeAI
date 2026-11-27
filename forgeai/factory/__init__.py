"""Factory pipeline modules for ForgeAI."""

from forgeai.factory.baseline import BaselineComparison, compare_factory_vs_human
from forgeai.factory.classifier import classify_failure, classify_failures
from forgeai.factory.evalgen import (
    EvalGenerationReport,
    generate_eval_set,
    load_eval_cases,
    save_eval_cases,
)
from forgeai.factory.generator import generate_agent_definition
from forgeai.factory.graph import FactoryState, build_factory_graph
from forgeai.factory.guard import run_groundability_guard
from forgeai.factory.intake import load_requirements, run_intake, save_requirements
from forgeai.factory.judge import (
    CrossEncoderNLIGrounder,
    LayeredJudge,
    LexicalNLIGrounder,
    NLIGrounder,
    compute_cohens_kappa,
    judge_pairwise_swapped,
)
from forgeai.factory.optimizer import (
    apply_patch,
    evaluate_agent_on_cases,
    propose_targeted_patch,
    run_optimization_loop,
)
from forgeai.factory.regression import RegressionReport, run_regression
from forgeai.factory.report import generate_scorecard_and_report, wilson_ci_95
from forgeai.factory.tournament import TournamentCandidate, TournamentResult, run_prompt_tournament

__all__ = [
    "BaselineComparison",
    "CrossEncoderNLIGrounder",
    "EvalGenerationReport",
    "FactoryState",
    "LayeredJudge",
    "LexicalNLIGrounder",
    "NLIGrounder",
    "RegressionReport",
    "TournamentCandidate",
    "TournamentResult",
    "apply_patch",
    "build_factory_graph",
    "classify_failure",
    "classify_failures",
    "compare_factory_vs_human",
    "compute_cohens_kappa",
    "evaluate_agent_on_cases",
    "generate_agent_definition",
    "generate_eval_set",
    "generate_scorecard_and_report",
    "judge_pairwise_swapped",
    "load_eval_cases",
    "load_requirements",
    "propose_targeted_patch",
    "run_groundability_guard",
    "run_intake",
    "run_optimization_loop",
    "run_prompt_tournament",
    "run_regression",
    "save_eval_cases",
    "save_requirements",
    "wilson_ci_95",
]
