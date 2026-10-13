# Human Validation Instructions — LLM-Judge Agreement Check

## Purpose
We used an LLM (Claude Sonnet 5) to judge factual correctness of 828 model answers
to simple factual questions (e.g., "What is the capital of France?"). To validate
that this LLM judge is reliable, we drew a random sample of 50 items and need an
independent human rater to re-judge them **blind** to the LLM's verdict, so we can
compute agreement (accuracy + Cohen's kappa) between the human and the LLM judge.

This is for the paper's methodology section, to support the LLM-as-judge approach
with an inter-rater reliability statistic.

## File
`human_validation_sample.csv` — 50 rows, columns:

| Column | Description |
|---|---|
| `idx` | Row index (0-49) |
| `model` | Which model generated the answer (diffusion or AR model name) |
| `temp` | Sampling temperature used |
| `question` | The factual question asked |
| `answer` | The model's raw generated answer (may be verbose, truncated, or malformed) |
| `llm_judge_verdict` | **DO NOT LOOK AT THIS UNTIL YOU FINISH** — the LLM judge's verdict (CORRECT/INCORRECT) |
| `llm_judge_reason` | **DO NOT LOOK AT THIS UNTIL YOU FINISH** — the LLM judge's stated reason |
| `human_verdict` | **YOU FILL THIS IN** — write CORRECT or INCORRECT |
| `human_notes` | **YOU FILL THIS IN** — optional one-line note, especially for disagreements or edge cases |

## How to label

For each row, read only `question` and `answer` (ignore the `llm_judge_*` columns
until you are completely done with all 50 rows). Write `CORRECT` or `INCORRECT`
in `human_verdict`.

**Judging rule — is the core factual claim correct?**
- Ignore formatting, verbosity, repetition, awkward grammar, or extra elaboration.
  Only judge whether the *correct fact* is present.
- If the correct fact appears anywhere in the answer (even wrapped in junk text or
  a truncated/garbled sentence), mark **CORRECT**.
- If the answer is repetitive nonsense, refuses to answer, never actually states
  a fact, contradicts itself with multiple different answers, or states a
  factually wrong claim, mark **INCORRECT**.
- Use your own general knowledge — these are simple, well-known facts
  (capitals, authors, scientific constants, historical events). No need to
  search anything.

## After labeling
1. Save the completed CSV.
2. Reveal/compare the `llm_judge_verdict` column against your `human_verdict`.
3. Report back: how many of the 50 rows match vs. disagree, and for any
   disagreements, a one-line note on why (this is the most valuable part —
   disagreements reveal judge failure modes worth mentioning in the paper).

We'll then compute simple agreement % and Cohen's kappa from the completed file.
