# LLM-as-Judge Prompt — Reasoning Task (for paper Appendix)

**Judge model:** Claude Sonnet 5 (via GitHub Copilot CLI sub-agents), one call per batch of 30
items, 18 batches dispatched in parallel (9 diffusion-model batches + 9 AR-model batches).

**Task given to judge:**

> For each item, judge whether `answer` reaches the CORRECT FINAL ANSWER to the reasoning
> question in `question` (arithmetic, word problems, time/date logic, or basic deduction).
>
> Judging rules:
> - Judge only whether the final numeric/factual answer stated is correct — do not judge
>   style, verbosity, or whether the model "shows its work."
> - Ignore LaTeX formatting, step-by-step derivations, or extra elaboration; if a correct
>   final answer appears anywhere in the response, mark it correct.
> - If the response is truncated before it states a final answer (e.g., cut off mid-formula
>   or mid-derivation with no stated conclusion), mark it incorrect.
> - If the response is self-contradictory (e.g., states two different final answers without
>   resolving which is final), mark it incorrect, even if one of the stated answers is correct.
> - If the response never attempts the question, is repetitive garbage, or states a wrong
>   final answer, mark it incorrect.

**Output schema:** JSON array of `{"id": <item id>, "correct": true/false, "reason": "<one short
sentence>"}`, one entry per input item, order-preserved.

**Batching:** Each diffusion model was judged across 3 temperatures (0.5/0.7/0.9) × 30 unique
reasoning questions = 90 items; each AR model likewise. Diffusion responses were deduplicated
across AR-model rows before judging (as with the factual task), yielding 270 unique diffusion
QA pairs and 270 unique AR QA pairs total (9 batches of 30 each, per model type).

**Rationale (for paper):** This rubric was introduced after discovering the factual-task NLI
judge's verbosity bias and, separately, a token-budget truncation confound that disproportionately
penalized AR models' longer chain-of-thought responses on reasoning questions. The rubric
explicitly credits correct final answers regardless of derivation length/format, and explicitly
penalizes truncation and self-contradiction as their own failure modes — decoupling truncation
handling from the pure correctness judgment.
