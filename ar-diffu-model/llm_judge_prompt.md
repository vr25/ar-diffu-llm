# LLM-as-Judge Prompt (for paper Methodology section)

**Judge model:** Claude Sonnet 5 (via GitHub Copilot CLI sub-agents), one call per (model, temperature) batch of 92 items, dispatched in parallel.

**Task given to judge:**

> For each item, judge whether `answer` is FACTUALLY CORRECT as a response to `question`,
> using your own world knowledge (no internet access needed — these are simple, well-known
> factual questions like "What is the capital of France?").
>
> Judging rules:
> - Ignore formatting, verbosity, extra elaboration, or awkward phrasing. Only judge factual
>   correctness of the core claim.
> - If the answer contains the correct fact anywhere (even if wrapped in extra text), mark it
>   correct.
> - If the answer is truncated mid-sentence but the truncated portion already states the
>   correct fact, mark it correct.
> - If the answer is repetitive garbage, nonsensical, refuses to answer, states a wrong fact,
>   or never actually answers the question, mark it incorrect.

**Output schema:** JSON array of `{"id": <item id>, "correct": true/false, "reason": "<one short sentence>"}`, one entry per input item, order-preserved.

**Batching:** Each diffusion model was judged across 3 temperatures (0.5/0.7/0.9) × 92 unique questions = 276 items; each AR model likewise. Diffusion responses were deduplicated across AR-model rows before judging (since `--reuse-diffu` caches diffusion generations per temperature, independent of AR model choice), yielding 828 unique diffusion QA pairs and 828 unique AR QA pairs total (9 batches of 92 each, per model type).

**Rationale (for paper):** Replaces `cross-encoder/qnli-distilroberta-base` (a QNLI entailment/answerability classifier repurposed as a truthfulness judge), which was found to systematically penalize short, direct, factually-correct answers (e.g., scoring "The chemical formula for water is H₂O." as 0.029/hallucinated). The LLM-judge instructions explicitly correct for this by directing the judge to ignore verbosity/formatting and focus only on core factual correctness.
