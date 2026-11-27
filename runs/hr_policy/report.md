# ForgeAI Honest Quality Report — Shore HR Policy Assistant

## Headline Summary
- **Final Agent Version:** `v1` (`hr_policy_v0`)
- **Dev Pass Rate:** 14/16 (87.5%)
- **Holdout Pass Rate:** 8/8 (100.0%) — **Wilson 95% CI:** [67.6%, 100.0%]

## What Still Fails and Why
- No holdout failures observed in this run (note sample size and CI width above).

## Groundability Gaps & Unattended Assumptions
- **Unsupported Topic (moved to refuse):** Employee stock options (refuse with specific message about missing document)
- **Assumption:** Employee stock options questions are refused because the relevant policy document is not in the knowledge base
- **Assumption:** Meal reimbursement rules are only answered if explicitly found in travel_and_expense_policy.pdf; otherwise, state they are not detailed
- **Assumption:** All offshore seafarer inquiries (leave, travel, payroll) are refused as documents are shore-based only
- **Assumption:** Responses must be under 100 words with bullet points for lists
- **Assumption:** Medical certificates are only confirmed as required per policy, not interpreted or validated
- **Assumption:** Assumed for 'The document inventory does not contain a policy file for employee stock options. Should the agent be configured to refuse questions about stock options due to missing data, or is there a separate document to be added?': Refuse questions about stock options, stating that the relevant policy document is not currently available in the knowledge base.
- **Assumption:** Assumed for 'The brief mentions 'meal reimbursement rules' under travel, but the provided travel policy preview only details airfare and accommodation. Does the travel policy document contain specific meal allowance limits, or should the agent state that meal rules are not defined in the provided text?': If meal rules are not explicitly found in the text of travel_and_expense_policy.pdf, the agent must state that specific meal reimbursement limits are not detailed in the provided policy document.
- **Assumption:** Assumed for 'The brief requires the agent to refuse 'offshore seafarer payroll inquiries.' Should the agent also refuse questions about offshore seafarer *leave* or *travel* policies, or is the refusal strictly limited to payroll/compensation topics?': Refuse all inquiries regarding offshore seafarers (including leave, travel, and payroll) as the provided documents are exclusively for shore-based staff.
- **Assumption:** Assumed for 'For the 'professional and concise tone,' is there a specific maximum word count or format preference (e.g., bullet points only) for responses, or is 'concise' a general stylistic guideline?': Keep responses under 100 words, using bullet points for lists of entitlements or rules, and avoiding conversational filler.
- **Assumption:** Assumed for 'The brief mentions refusing 'medical diagnoses.' Should the agent also refuse to interpret medical certificates for sick leave eligibility, or is it acceptable to verify the *format* of a certificate without diagnosing the condition?': Refuse to interpret or validate the medical content of any document. Only confirm that a medical certificate is required per policy, without examining the certificate itself.

## Tool Selection Audit
- **Attached `current_date`:** The brief requires answering questions about leave accrual and travel booking lead times (e.g., '14 days before departure'), which may require the current date to calculate specific deadlines or eligibility windows.
- **Rejected `csv_lookup`:** No CSV files found in the document folder.
- **Rejected `calculator`:** Brief does not indicate arithmetic or pricing calculations.
- **Rejected `weather_api`:** No weather, forecast, or meteorological requirement in the brief.

## Eval Synthesis & Distrust Filter Audit
- Candidates generated: 38, Kept: 24, Dropped: 7

## Prompt Tournament Comparison
- **Winner:** `baseline`

| Candidate | Strategy | Dev Slice Pass Rate | Pairwise Wins |
|---|---|---:|---:|
| `baseline` | baseline | 66.7% | 0 |
| `candidate_2` | structured_grounded | 66.7% | 0 |
| `candidate_1` | strict_terse | 50.0% | 0 |

## Judge Calibration & Budget Usage
- **Judge-Human Agreement (Cohen's κ):** Not calibrated in this unattended run.
- **LLM Calls:** 233 | **Total Tokens:** 219744 | **Cumulative Call Latency:** 106.30s
