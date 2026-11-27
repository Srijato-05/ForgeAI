# ForgeAI Scorecard — Shore HR Policy Assistant (v1)

## Optimization Iterations (Dev Split)

| Iter | Target Field | Score Before | Score After | Regressions | Accepted | Stop Reason |
|---:|---|---:|---:|---:|:---:|---|
| 1 | `system_prompt.style` | 81.2% | 87.5% | 0 | yes |  |
| 2 | `system_prompt.grounding_rules` | 87.5% | 87.5% | 0 | no |  |
| 3 | `system_prompt.refusal_policy` | 87.5% | 87.5% | 1 | no | max_iterations_reached |

## Category Breakdown

| Category | Dev Pass | Holdout Pass |
|---|---|---|
| ambiguous | 1/2 (50%) | 1/1 (100%) |
| multi_doc | 1/1 (100%) | — |
| out_of_scope | 0/1 (0%) | 1/1 (100%) |
| single_hop | 12/12 (100%) | 6/6 (100%) |
