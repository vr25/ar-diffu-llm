# AR vs Diffusion: Judge Bias and Generation-Length Truncation

Code and experimental evidence accompanying the final study of NLI-judge bias
and generation-budget truncation in autoregressive-vs-diffusion evaluation.
The manuscript, paper PDFs, and obsolete pilot experiments are not included.

## Experiment scope

Main models: Qwen2.5-Instruct 0.5B, 1.5B, and 3B; LLaDA-MoE-7B-A1B-Instruct;
Dream-v0-Instruct-7B; and `fwerkor/MiniCPM5-2B-Diffusion-Base`, a third-party
masked-diffusion conversion, not an instruction-tuned MiniCPM checkpoint.

Main tasks: 92 factual questions and 30 simple reasoning questions at
temperatures 0.5, 0.7, and 0.9. Supporting checks cover a 50-item GSM8K sample,
SmolLM2, greedy decoding, and diffusion token-budget sweeps.

## Data and code map

| Files | Purpose |
|---|---|
| `ar_diffu_local.py`, `run_ar_only.py`, `run_ar_vs_{lladamoe,dream,minicpm5}.py` | Shared inference and final-model runners |
| `questions_and_answers.csv`, `data/gsm8k50.json` | Main questions/reference answers and sampled GSM8K data |
| `results_lladamoe/`, `results_dream/`, `results_minicpm5_local/` | Main factual generation records and original diffusion NLI scores |
| `results_lladamoe_reasoning/`, `results_dream_reasoning/`, `results_minicpm5_reasoning/` | 150-token reasoning generation records |
| `llm_judge_input*.json` | Primary-judge inputs, including historical 40-token reasoning outputs |
| `llm_judge_*merged.json` | Primary-judge verdicts: factual, 40-token reasoning, 150-token reasoning (`v2`), and diffusion sweeps |
| `llm_judge_ar_reasoning_v2_batch_*.json` | Authoritative 150-token AR inputs required by `analysis/budget_sweep.py` |
| `llm_judge_prompt*.md` | Factual and reasoning scoring rubrics |
| `judge2/*.json` | Second-judge inputs and outputs |
| `judge3/*_0*.json` | Supporting-experiment judge inputs and outputs |
| `human_validation*` | Human annotations, sampled responses, and instructions |
| `results_ar_only/`, `gsm8k/`, `sweep/` | AR generalization checks, diffusion GSM8K outputs, and measured diffusion sweeps |
| `analysis/*.py`, `analysis/*.json` | Analysis scripts and saved analysis results |
| `analysis/lit_audit.md` | Supporting per-paper literature-audit table, not the manuscript |
| `run_diffu_sweep.sh`, `run_gsm8k_diffu.sh`, `run_strengthen_queue.sh` | Original local experiment queues |
| `consolidate_results.py`, `bootstrap_ci.py` | Utilities for the original NLI-scored comparison records |

### Which scores to use

The main paper's corrected accuracy is based on the primary LLM-judge verdicts,
not the `truthful_pct` or `hallucinated` fields in generation records. Those
fields are the original NLI scores used to demonstrate the judge confound.
Likewise, `bootstrap_ci.py` and `consolidate_results.py` summarize the original
NLI labels; they do not reproduce the corrected LLM-judged accuracy table.

For factual QA, use `llm_judge_merged.json` (diffusion) and
`llm_judge_ar_merged.json` (AR). For final 150-token reasoning, use
`llm_judge_reasoning_v2_merged.json` and
`llm_judge_ar_reasoning_v2_merged.json`. Files without `v2` are the original
40-token reasoning evaluations and remain relevant to the truncation claim.

Judge IDs identify model, temperature, and question index. Some merged files
are dictionaries keyed by ID; others are lists of `{id, correct, reason}`.
Use the corresponding judge inputs to join labels to responses.

The authoritative final AR reasoning responses are in
`llm_judge_input_ar_reasoning_v2.json` and its nine batch files. Some
`results_*_reasoning/` files contain independently regenerated AR responses;
do not substitute those for the judged AR inputs. Diffusion responses in
those directories correspond to the final diffusion-judge inputs.

## Setup

Use Python 3.10 or newer and an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list is a starting specification, not a recovered lockfile of
the original environment. Model-specific remote code can impose additional
requirements. Models and tokenizers are downloaded from Hugging Face; weights,
model caches, and credentials are not bundled. Review remote model code before
running inference. Large diffusion models require substantial memory.

## Recomputing the provided analyses

Run from the repository root:

```bash
python analysis/judge_agreement.py
python analysis/judge3_results.py
python analysis/nli_bias.py
```

The first two commands use saved data and the Python standard library.
The NLI analysis downloads/loads the original cross-encoder and scores AR
outputs on CPU. These commands overwrite their corresponding analysis JSONs.

The AR token-budget analysis explicitly requires a locally cached tokenizer.
Download it once before running the script:

```bash
python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")'
python analysis/budget_sweep.py
```

`analysis/diffu_budget_sweep.json` is a saved measured-sweep summary; the original
project has no standalone script for regenerating this summary. Its underlying
generation records and merged judge labels are included. Main-table confidence
intervals, permutation tests, and human-agreement calculations likewise do not
have a complete standalone reproduction script in the original project;
their underlying labels and annotations are supplied for independent analysis.
This package is not a one-command reproduction of every statistic.

## Regenerating outputs

Use explicit AR model selections: the shared runner also supports SmolLM2,
but the main comparison uses only the three Qwen2.5 sizes.

```bash
python run_ar_vs_lladamoe.py --question-set factual --ar-models qwen25-0.5b qwen25-1.5b qwen25-3b --temps 0.5 0.7 0.9 --gen-tokens 40 --out-dir rerun/lladamoe_factual
python run_ar_vs_dream.py --question-set reasoning --ar-models qwen25-0.5b qwen25-1.5b qwen25-3b --temps 0.5 0.7 0.9 --gen-tokens 150 --out-dir rerun/dream_reasoning
python run_ar_only.py --ar-models qwen25-0.5b qwen25-1.5b qwen25-3b --question-set gsm8k --temps 0.7 --gen-tokens 512 --out-dir rerun/ar_gsm8k
```

Repeat with the other final-model runners and task settings as needed.
Generation is stochastic; reruns need not match saved answers exactly.
LLM judging was performed outside these runners using the supplied rubrics
and JSON inputs; no automatic judge API client is bundled.

The shell queues preserve historical machine-specific settings (`mps`, offline
Hugging Face mode, local-job waiting). Prefer direct runner commands on a new
machine; if using the queues, prepare model caches and create `logs/` first.
MiniCPM5 uses total sequence length rather than a true new-token cap; its
effective answer budget is shorter than the nominal setting.

## Publishing

Extract the archive and upload or commit the contents of `ar-diffu-model/`
to GitHub. GitHub does not unpack an uploaded ZIP into browsable code.

Excluded: manuscript files, paper sources/PDFs, obsolete Colab and GPT-2
experiments, pilot/lean/full-pipeline prototypes, old LLaDA/Qwen3 runs,
ARCHE training code, placeholder plotting code, redundant per-batch primary
judge outputs, internal handoffs, logs, caches, and nested Git history.

No project-wide license has been inferred or assigned. Upstream dataset and
model usage remains subject to the respective terms.
