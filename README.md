# ForgeAI — The Agent That Builds Agents

ForgeAI is an autonomous **agent factory** built with **LangGraph**, **Pydantic v2**, hybrid BM25 + dense retrieval, a layered deterministic/NLI/cross-family judge, and a targeted critic–optimizer loop. Given a one-paragraph brief and a folder of documents, it produces a runnable declarative agent (`agent_final.yaml`), an auto-generated evaluation suite (`evals.jsonl`), per-iteration scorecards (`scorecard.md`), and an honest quality report (`report.md`).

---

## Architecture Highlights

- **Intake Interviewer (`forgeai/factory/intake.py`):** Inspects the brief and document inventory, asks at most 5 information-gain–ranked clarifying questions (with automatic `assumed: true` defaults in `--unattended` mode), and writes `requirements.yaml`.
- **Groundability Guard (`forgeai/factory/guard.py`):** Probes `must_answer` topics against the document corpus before agent generation and downgrades unsupported topics to `known_gaps` / `must_refuse`.
- **Modular Agent Generator (`forgeai/factory/generator.py`):** Produces a sectioned `AgentDefinition` (`agent_v0.yaml`) with 2-step selective tool attachment (`csv_lookup`, `calculator`, `weather_api`, `current_date`) and logs rejected tools.
- **Eval Generator + Distrust Filters (`forgeai/factory/evalgen.py`):** Synthesizes `single_hop`, `multi_doc`, `ambiguous`, `out_of_scope`, and `tool_required` cases from committed prompts (`forgeai/prompts/`), filters out verbatim leakage, duplicates, and unanswerable cases, and creates a stratified `dev` / `holdout` split.
- **Layered Judge (`forgeai/factory/judge.py`):** Combines L0–L3 deterministic checks, L4 NLI claim groundedness, and L5 cross-family pointwise fact-rubric LLM scoring (plus position-swapped pairwise comparison and Cohen's κ calibration).
- **Targeted Critic–Optimizer Loop (`forgeai/factory/optimizer.py`):** Classifies failures deterministically (`retrieval_miss`, `hallucination`, `wrong_refusal`, `prompt_defect`), applies one targeted patch per iteration (up to 3 iterations), accepts only on dev improvement with zero regressions, and stops early on pass threshold, budget exhaustion, or marginal-gain plateau.
- **Honest Reporting (`forgeai/factory/report.py`):** Reports holdout pass rate with Wilson 95% confidence intervals, remaining failures grouped by root cause, groundability gaps, assumptions, and token/latency totals.
- **Stretch Goals Included:** Prompt Tournament (`--tournament`), Budget Enforcement, Human Checkpoint (`--human-checkpoint`), Regression Mode (`forge regress`), Factory vs. Human Baseline (`forge compare`), and Streamlit Web UI (`ui/app.py`).

---

## Setup

1. Copy `.env.example` to `.env` and configure your preferred open-weight provider (`groq`, `openrouter`, `sambanova`, or local `ollama`):
   ```bash
   cp .env.example .env
   ```
2. Check environment & live endpoint connectivity:
   ```bash
   python -m forgeai.cli doctor --ping
   ```

---

## CLI Commands

```bash
# 1. Build, test, optimize, and report on any brief + document folder (unattended)
python -m forgeai.cli build --brief demos/hr_policy/brief.md --docs demos/hr_policy/docs --run-dir runs/hr_policy --unattended

# Optional flags: --tournament, --human-checkpoint, --provider ollama, --embedder st, --max-iterations 3, --eval-size 24

# 2. Chat interactively with any produced agent definition
python -m forgeai.cli chat runs/hr_policy/agent_final.yaml --docs demos/hr_policy/docs --index runs/hr_policy/index

# 3. Evaluate any agent YAML against an evals.jsonl file
python -m forgeai.cli eval runs/hr_policy/agent_final.yaml --evals runs/hr_policy/evals.jsonl --docs demos/hr_policy/docs

# 4. View scorecard & honest report
python -m forgeai.cli report runs/hr_policy

# 5. Regression mode when documents change
python -m forgeai.cli regress runs/hr_policy --docs path/to/updated_docs --regenerate

# 6. Compare ForgeAI agent vs. hand-written baseline
python -m forgeai.cli compare runs/hr_policy/agent_final.yaml demos/human_baseline/hr_policy_human.yaml --evals runs/hr_policy/evals.jsonl --docs demos/hr_policy/docs
```

### Streamlit Web UI

```bash
streamlit run ui/app.py
```

---

## Demo Datasets

- **`demos/hr_policy/`** — Shore staff HR policy assistant over 3 PDFs (`shore_leave_policy.pdf`, `travel_and_expense_policy.pdf`, `hybrid_and_remote_work_policy.pdf`). Tests pure document RAG, strict refusals, Groundability Guard flagging an unsupported brief topic (stock options), and 0 attached tools.
- **`demos/ship_spares/`** — Ship-spares procurement assistant over `spares_catalogue.csv` + 2 supplier PDFs (`nordic_marine_terms.pdf`, `pacific_diesel_supplier_guide.pdf`). Tests selective attachment of `csv_lookup`, `calculator`, and `current_date` while rejecting `weather_api`.
- **`demos/human_baseline/`** — Hand-written `hr_policy_human.yaml` baseline for `forge compare`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

