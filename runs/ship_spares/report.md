# ForgeAI Honest Quality Report — Ship Spares Procurement Assistant

## Headline Summary
- **Final Agent Version:** `v0` (`ship_spares_v0`)
- **Dev Pass Rate:** 16/16 (100.0%)
- **Holdout Pass Rate:** 7/8 (87.5%) — **Wilson 95% CI:** [52.9%, 97.8%]

## What Still Fails and Why
### `prompt_defect` (1 case(s))
- **eval_012** — *Does the provided text from the Pacific Diesel Parts guide specify the exact number of days allowed for returns?*: Context retrieved, but answer missed key facts (L2=0.50, L5=0.00).

## Groundability Gaps & Unattended Assumptions
- **Unsupported Topic (moved to refuse):** Estimated delivery dates calculated by adding lead_time_days to the current date.
- **Assumption:** No bulk discounts are applied unless explicitly stated in the supplier PDFs; cost is calculated as Quantity * Unit Price subject to MOQ.
- **Assumption:** Requests for uncatalogued parts are refused, and the user is directed to contact a human buyer.
- **Assumption:** Warranty and shipping answers are limited to suppliers explicitly covered in the provided PDFs (Nordic Marine Systems, Pacific Diesel Parts); other suppliers are deemed out of scope for this knowledge base.
- **Assumption:** Delivery date estimation uses calendar days (simple addition of lead_time_days to today's date) unless a supplier PDF specifies business days only.
- **Assumption:** Assumed for 'What are the specific rules for calculating bulk discounts (e.g., percentage off, tiered thresholds, or supplier-specific logic) since the CSV only contains unit prices and MOQs?': Assume no bulk discounts apply unless explicitly stated in the supplier PDFs; calculate cost as (Quantity * Unit Price) subject to MOQ constraints.
- **Assumption:** Assumed for 'How should the agent handle requests for parts not found in the 16-row spares_catalogue.csv? Should it refuse, suggest similar parts, or escalate to a human?': Refuse to provide pricing/stock for uncatalogued parts and inform the user that the part is not in the current system, suggesting they contact a human buyer.
- **Assumption:** Assumed for 'Which specific supplier PDFs are authoritative for warranty and shipping questions, and how should the agent handle conflicts if a part's supplier is not covered by the provided PDFs (e.g., only Nordic and Pacific are listed)?': Only answer warranty/shipping questions for suppliers explicitly covered in the provided PDFs (Nordic Marine Systems, Pacific Diesel Parts). For other suppliers, state that specific terms are not available in the current knowledge base.
- **Assumption:** Assumed for 'What constitutes 'unauthorized vendor negotiations' that must be refused? Does this include any discussion of price changes, contract terms, or only direct negotiation attempts?': Refuse any request to negotiate prices, alter contract terms, or discuss future pricing strategies. Only provide current static pricing from the CSV.
- **Assumption:** Assumed for 'Should the delivery date estimation account for weekends and holidays, or is it a simple addition of lead_time_days to the current date?': Use calendar days (simple addition of lead_time_days to today's date) unless the supplier PDFs specify business days only.

## Tool Selection Audit
- **Attached `csv_lookup`:** Required to retrieve part pricing, stock quantities, and lead times from the spares_catalogue.csv as specified in the brief.
- **Attached `calculator`:** Required to calculate total order costs (Quantity * Unit Price) and enforce MOQ constraints as specified in the brief.
- **Attached `current_date`:** Required to estimate delivery dates by adding lead_time_days to the current date as specified in the brief.
- **Rejected `weather_api`:** Eligible by heuristic but deemed unnecessary for the brief.

## Eval Synthesis & Distrust Filter Audit
- Candidates generated: 39, Kept: 24, Dropped: 2

## Prompt Tournament Comparison
- **Winner:** `baseline`

| Candidate | Strategy | Dev Slice Pass Rate | Pairwise Wins |
|---|---|---:|---:|
| `baseline` | baseline | 100.0% | 0 |
| `candidate_2` | structured_grounded | 100.0% | 0 |
| `candidate_1` | strict_terse | 83.3% | 0 |

## Judge Calibration & Budget Usage
- **Judge-Human Agreement (Cohen's κ):** Not calibrated in this unattended run.
- **LLM Calls:** 144 | **Total Tokens:** 130769 | **Cumulative Call Latency:** 507.81s
