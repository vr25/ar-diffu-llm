# Literature Audit: Evaluation Confounds in AR-vs-Diffusion Language Model Comparisons

Scope: discrete/masked diffusion language models (dLLMs/DLMs) evaluated or compared against autoregressive (AR) models on QA, reasoning, math (GSM8K/MATH), or hallucination/factuality. All entries below were verified directly against arXiv abstract pages and/or full HTML/PDF text (not search snippets alone). Fields are marked **"not reported"** where the paper does not explicitly state them — nothing below is inferred from general knowledge.

Three additional papers were checked and **excluded** because, despite being widely cited "diffusion LM" baselines, their own evaluation sections contain **no** QA/reasoning/math/hallucination experiments at all (confirmed via exhaustive keyword search for GSM8K/exact-match/CoT/harness — zero hits): **SEDD** (Lou et al., 2310.16834, perplexity-only), **MDLM** (Sahoo et al., 2406.07524, perplexity + GLUE), **Plaid** (Gulrajani & Hashimoto, 2305.18619, perplexity + fluency-only human eval). These are noted here for completeness but not included in the main table/counts. Also excluded: **Gemini Diffusion** (Google DeepMind) — confirmed to have *no* arXiv technical report, only a marketing blog/model-card with no stated methodology; **MiniCPM "diffusion" variant** — confirmed not to exist as a published diffusion model (MiniCPM/MiniCPM5 are standard dense AR transformers); **Seed Diffusion** (ByteDance, 2508.02193) — real paper but code-generation/speed-only, no GSM8K/MATH/QA/hallucination content.

## Main Table

| # | Paper | arXiv ID | Scoring / Judge Method | Gen. Length / Token Budget (AR vs. Diffusion) | Same Budget for Both? | Length Sweep Reported? | Length-Sensitivity / Truncation Discussion | CoT for AR Baseline? |
|---|---|---|---|---|---|---|---|---|
| 1 | LLaDA — *Large Language Diffusion Models* (Nie et al., 2025) | 2502.09992 | Mixed: conditional-likelihood selection for MC tasks (MMLU, ARC-C, etc.); "exact match and other relevant criteria" for generative tasks (GSM8K, Math, HumanEval); base model via lm-eval-harness, instruct model via internal library | Diffusion: fixed 1024 (base model, Appendix B.4/B.6) or per-task values 3–512 (instruct model). AR ("self-constructed ARM") baseline generates until EOS, not a fixed length | **No** — different stopping mechanisms (EOS-stop for AR vs. required length budget for diffusion); paper explicitly notes ARM baseline was *not* scaled to the same compute as LLaDA | **Yes** — Appendix B.5 ablation over {256, 512, 1024} on BBH/GSM8K/Math/HumanEval/MBPP | Yes, explicit: "results are not sensitive to the length hyperparameter" | Not reported |
| 2 | DiffuLLaMA — *Scaling Diffusion Language Models via Adaptation from Autoregressive Models* (Gong et al., 2024/2025) | 2410.17891 | **Exact-match accuracy** (TriviaQA, Lambada "measured by exact match accuracy"); GSM8K/math via external math-evaluation-harness; self-consistency/majority-vote over 3 samples | DLM: diffusion timesteps T set to required/oracle generation length (T=32 for Lambada CD, T=64 for GSM8K-symbolic). AR: "cut off maximum new tokens," using oracle/ground-truth length for infilling tasks | **Yes (target length)** — oracle-length matched for both, though implementation differs (timesteps vs. max_new_tokens cutoff) | Not reported (only decoding-steps vs. fluency/diversity ablation, not generation-length sweep) | Not reported as a length-sensitivity study; steps-vs-quality trade-off discussed instead | **Yes, explicit** — Table 2 reports zero-shot, few-shot, and **few-shot+CoT** conditions applied identically to AR (LLaMA2) and diffusion baselines |
| 3 | Dream 7B — *Dream 7B: Diffusion Large Language Models* (Ye et al., 2025) | 2508.15487 | **Not reported** in retrievable main text (no "exact match," "lm-evaluation-harness," or "LLM-as-judge" found via exhaustive keyword search); result tables report numeric scores with few-shot counts (e.g., "GSM8K 77.2 (8)") but scoring mechanism unstated | **Not reported** — paper discusses "diffusion steps" (denoising steps, a speed/quality knob) but no explicit max_new_tokens/gen_length figures found for either Dream or its AR comparators (Qwen2.5 7B, LLaMA3 8B) | Not reported | Not reported (only diffusion-step quality/speed trade-off ablation, Sec. 5.5.2) | Not reported for AR comparators; CoT mentioned only as conceptual background re: OpenAI o1/DeepSeek R1 | Not reported |
| 4 | LLaDA-MoE — *A Sparse MoE Diffusion Language Model* (Zhu et al., 2025) | 2509.24389 | Not reported in detail; states "semi-autoregressive sampling... to ensure a fair and robust comparison"; for prior MDM baselines, "we cite the published numbers directly" | Diffusion: explicit — "generation length of 1024 and a block length of 64 ... for all other generative benchmarks" | **Ambiguous/not reported** — unclear whether AR baseline (Qwen2.5-3B-Instruct) used the identical 1024-token budget or whether its numbers were taken from its own paper | Not reported | Not reported | Not reported |
| 5 | Mercury — *Ultra-Fast Language Models Based on Diffusion* (Inception Labs / Khanna, Kharbanda et al., 2025) | 2506.17298 | Coding benchmarks via **pass@1 functional correctness** (test execution); separate **human evaluation** via Copilot Arena user preference votes (not QA correctness judging) | Not reported (no max_new_tokens/gen_length stated for benchmark evals; only ~1,000/1,000 input/output tokens mentioned for a third-party speed benchmark) | Not reported | Not reported | Not reported | CoT mentioned only generically as a supported prompting mode, not described as used in the AR-baseline evaluation protocol |
| 6 | DiffusionGemma Technical Report (Google DeepMind, 2026) | 2608.00146 | Not explicitly reported as exact-match/regex/LLM-judge; benchmark tables (AIME2026, GPQA-Diamond, GSM8K, MMLU-Pro, HumanEval, etc.) report numeric scores with no stated grading mechanism found in text | Fixed 256-token "canvas"/block per diffusion denoising step, iteratively concatenated to EOS; AR(MTP) baseline runs to natural EOS. Table 3: avg. total tokens differ substantially (e.g., TD=4,001 vs. AR(MTP)=7,207) | **No** — models run to natural EOS, not a matched token cap | No formal generation-length sweep; closest analog is a "Thinking vs. No-Thinking" ablation and a LoRA-rank/denoising-steps trade-off (Sudoku case study) | Not reported as a length-sensitivity study | **Yes** — explicit "thinking mode" (reasoning-trace) comparison for both DiffusionGemma and the Gemma 4 AR baseline, with/without thinking enabled; a separate ablation disables thinking in both to isolate architecture effects |
| 7 | Block Diffusion — *Interpolating Between Autoregressive and Diffusion Language Models* (Arriola et al., ICLR 2025) | 2503.09573 | **Not QA/exact-match** — generative **perplexity** under a separate GPT2-Large scorer (likelihood-based LM paper, not QA accuracy) | Explicit sweep: generation lengths **L=1024 and L=2048**, trained on 1024-token context; same lengths applied across AR, SEDD, MDLM, BD3-LM, SSD-LM baselines (Table 7) | **Yes** (same L values tested across all baselines) | **Yes** — Table 7 explicit L=1024/2048 comparison | Yes, explicit: Table 6 shows "BD3-LMs generate sequences up to ≈10× longer than SEDD," which is "restricted to the training context size" — a documented truncation limitation for fixed-length diffusion baselines | Not applicable (perplexity-only paper, no QA/reasoning CoT evaluation) |
| 8 | *Diffusion Language Models: An Experimental Analysis* (Bertolani et al., 2026) | 2606.19475 | Per-benchmark, explicitly reported via lm-evaluation-harness: MMLU/HellaSwag = log-likelihood scoring; MMLU-Redux = first-generated-token match; **GSM8K = "flexible-extract" regex/rule-based numeric extraction**; HumanEval/MBPP = pass@1; WMT16 = chrF; Sudoku = solution-consistency check. No LLM-as-judge or human eval used | Diffusion: generation length swept explicitly as "one token denoised per step so diffusion steps and output length scale together," with performance "saturat[ing] or declin[ing] beyond 256–512 tokens." AR (Qwen3-4B/8B) run under standard lm-eval-harness settings | **No** — the length sweep is DLM-only; paper notes "unlike autoregressive models, DLMs expose several inference-time control variables, including... generation length" (AR models don't expose this the same way) | **Yes — central contribution**: explicit sections "Scaling Analysis" (gen. length), "Scaling the Global Unmasking Ratio," "Scaling Block Size," "Scaling the Intra-Block Unmasking Ratio" | Yes, extensive: e.g., "translation quality (WMT16) scales poorly... collapsing at the largest budgets, suggesting... particularly sensitive to error accumulation over extended generation" | Not reported |
| 9 | d1 — *Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning* (Zhao et al., 2025) | 2504.12216 | Reward-based exact match: "2.0 points if the extracted answer exactly matches the ground truth" / "\boxed{}" match for math; GSM8K/MATH500/Countdown/Sudoku, 0-shot greedy decoding | Diffusion only — lengths **128, 256, 512** explicitly swept; RL training uses fixed 256-token online-generation length | N/A — **no AR baseline benchmarked** (comparisons are among dLLM variants: LLaDA-Instruct base/SFT/diffu-GRPO/d1) | **Yes** — explicit sweep over {128, 256, 512} | Yes: "larger gains from 128→256 (~9.1%) than 256→512 (~2.5%)"; qualitative reasoning-style changes documented by length | N/A (no AR baseline run) |
| 10 | Fast-dLLM — *Training-free Acceleration of Diffusion LLM via KV Cache and Parallel Decoding* (Wu et al., 2025) | 2505.22618 | "All evaluations are conducted using the standardized lm-eval library" (lm-eval-harness; GSM8K 5-shot, MATH 4-shot, HumanEval 0-shot, MBPP) | Diffusion (LLaDA/Dream backbones): explicit lengths **256 and 512** compared across baseline vs. +Cache vs. +Parallel vs. Fast-dLLM | N/A — **no new AR baseline run**; AR models (LLaMA3) mentioned only via prior-paper claims in related work | **Yes** — accuracy/throughput reported at both 256 and 512 | Reported as a core ablation dimension, not framed as a dedicated "truncation risk" analysis | N/A (no AR baseline run) |
| 11 | LLaDA 1.5 — *Variance-Reduced Preference Optimization for Large Language Diffusion Models* (Zhu et al., 2025) | 2505.19223 | MTBench/AlignBench/ArenaHard via **LLM-as-judge** ("gpt-4-32k" API); GSM8K/HumanEval/IFEval scoring method **not reported** | Diffusion only — explicit tuning/sweep: answer length over {64,128,256,512,1024}, block length over {8,16,32,64,128}, best config per benchmark reported in Appendix D.4 | N/A — **no AR baseline benchmarked** (comparisons are LLaDA-SFT vs. LLaDA 1.5-VRPO only) | **Yes** — explicit grid sweep over answer/block length | Yes, explicit: "LLaDA SFT... tends to generate an excessive number of EOS tokens... resulting in notably truncated outputs and adversely affecting model performance" | N/A (no AR baseline run) |
| 12 | SMDM — *Scaling up Masked Diffusion Models on Text* (Nie et al., ICLR 2025) | 2410.18514 | **Exact-match accuracy via lm-eval-harness**: "we use greedy sampling to generate responses... and calculate matching accuracy against the ground truth" (LAMBADA, GSM8K); other tasks use likelihood-based MC scoring | GSM8K fine-tuning uses **implicit/distilled CoT training data** (Deng et al. 2023) rather than standard few-shot CoT prompting, applied to **both** the MDM and the Llama-2 (7B) comparison model | **Partially** — both models fine-tuned on the same augmented data, but explicit generation-length budget equivalence is **not stated** | Not reported as a length sweep | Discussed as "train-test discrepancy in context lengths" (fixed 2048-token training context vs. variable test context), addressed via variable-length training/mask-padding | **Implicit CoT via fine-tuning data**, not standard explicit CoT prompting — applied to both MDM and Llama-2 baseline identically |
| 13 | TraceRL / TraDo — *Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models* (Wang et al., 2025) | 2509.06949 | Not explicitly reported (accuracy numbers given for GSM8K/MATH500/AIME2024/LiveCodeBench/LiveBench; exact answer-extraction/grading mechanism not stated in retrieved text) | Diffusion (TraDo-8B-Thinking): avg. response lengths explicitly reported — 5,872 (MATH500), 19,397 (AIME2024), 1,458 (GSM8K), 16,291 (LiveCodeBench), 16,233 (LiveBench) tokens (Appendix Table 5/C.1). **No equivalent AR token-count figures reported** | **No / unclear** — diffusion model's long-CoT budgets are disclosed but AR (Qwen2.5-7B-Instruct, LLaMA3.1-8B-Instruct) token budgets are not, making matching unverifiable — a likely confound | Not reported as a formal sweep (block-size scaling discussed instead) | Not reported as a dedicated length-sensitivity analysis | **Not reported** whether AR baselines used CoT prompting; diffusion model uses an explicit "thinking" prompt for its long-CoT mode |
| 14 | DynHD — *Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning* (Qian et al., 2026) | 2603.16459 | **LLM-as-judge**: "Ground-truth correctness labels are provided by GPT-4o-mini, which is instructed to judge whether a generated answer is factually correct"; reliability checked against humans ("94% agreement... with human judgments") | Diffusion only (LLaDA-8B-Instruct, Dream-7B-Instruct) — explicit comparison at generation lengths **128 and 64** tokens (Table 1, TriviaQA/HotpotQA/CommonsenseQA) | N/A — **no AR baseline benchmarked** (diffusion-only hallucination-detection study) | **Yes** — main results table explicitly varies length (128 vs. 64) | Yes, central to the method: notes diffusion models generate "fixed-length sequences... with many structural padding tokens... carry[ing] little hallucination-related information," flagged as a limitation for future broader decoding-configuration study | N/A (no AR baseline run) |

## Papers Checked and Excluded (not relevant / no comparison exists)

| Paper | arXiv ID | Reason for exclusion |
|---|---|---|
| SEDD (Lou, Meng, Ermon, 2023) | 2310.16834 | Perplexity/zero-shot-likelihood only; no GSM8K/exact-match/CoT/harness content (0 occurrences each) |
| MDLM (Sahoo et al., 2024) | 2406.07524 | Perplexity + GLUE classification only; no QA/reasoning/math eval |
| Plaid (Gulrajani & Hashimoto, 2023) | 2305.18619 | Perplexity + fluency-only human eval; no QA/reasoning/math eval |
| Gemini Diffusion (Google DeepMind) | — | No arXiv technical report exists; only a marketing blog/model-card with no stated methodology |
| "MiniCPM diffusion" | — | Does not exist as a published diffusion model; MiniCPM/MiniCPM5 are standard dense AR transformers |
| Seed Diffusion (ByteDance, 2025) | 2508.02193 | Real paper, but code-generation/inference-speed only; no GSM8K/MATH/QA/hallucination content |

## Summary Counts (across the 14 main-table papers)

- **(a) Single fixed generation budget shared identically across AR and diffusion:** **1 / 14** — only Block Diffusion (#7, L=1024/2048 applied uniformly to AR and all diffusion baselines). DiffuLLaMA (#2) matches the *target* length via oracle length but uses different underlying mechanisms (timesteps vs. max_new_tokens).
- **(b) Report any length sweep/ablation (diffusion-side or cross-model):** **7 / 14** — LLaDA (#1), Block Diffusion (#7), Experimental Analysis (#8), d1 (#9), Fast-dLLM (#10), LLaDA 1.5 (#11), DynHD (#14).
- **(c) Use NLI/entailment judges:** **0 / 14** — none of the audited papers report using an NLI/entailment classifier for scoring.
- **(d) Use LLM-as-judge:** **3 / 14** — LLaDA 1.5 (#11, MTBench/AlignBench/ArenaHard via GPT-4-32k), DiffusionGemma (#6, scoring mechanism for benchmark tables not explicit but model uses "thinking" judged qualitatively — counted cautiously, see note below), DynHD (#14, GPT-4o-mini factuality judge). **Note:** DiffusionGemma's (#6) use of LLM-judge is not explicitly confirmed in the audited text; a strict count is **2 / 14** (LLaDA 1.5, DynHD).
- **(e) Use exact-match/regex extraction:** **6 / 14** — LLaDA (#1, generative tasks), DiffuLLaMA (#2), Experimental Analysis (#8, GSM8K flexible-extract), d1 (#9, reward-based exact/boxed match), Fast-dLLM (#10, via lm-eval-harness), SMDM (#12, lm-eval-harness greedy-match).
- **Papers with no AR baseline at all (diffusion-only comparisons):** 5 / 14 (#9 d1, #10 Fast-dLLM, #11 LLaDA 1.5, #14 DynHD, and partially #4/#13 where AR numbers are cited from other papers rather than run directly) — a further confound: several "comparisons" rely on externally reported AR numbers rather than a controlled re-run under matched conditions.
- **Papers using CoT for AR baselines:** **2 / 14 explicit** (DiffuLLaMA #2, DiffusionGemma #6); 1 **implicit-CoT-via-finetuning** (SMDM #12); remainder not reported or not applicable.

## BibTeX

```bibtex
@article{nie2025llada,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}

@article{gong2024diffullama,
  title={Scaling Diffusion Language Models via Adaptation from Autoregressive Models},
  author={Gong, Shansan and Agarwal, Shivam and Zhang, Yizhe and Ye, Jiacheng and Zheng, Lin and Li, Mukai and An, Chenxin and Zhao, Peilin and Bi, Wei and Han, Jiawei and Peng, Hao and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2410.17891},
  year={2024}
}

@article{ye2025dream,
  title={Dream 7B: Diffusion Large Language Models},
  author={Ye, Jiacheng and Xie, Zhihui and Zheng, Lin and Gao, Jiahui and Wu, Zirui and Jiang, Xin and Li, Zhenguo and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2508.15487},
  year={2025}
}

@article{zhu2025lladamoe,
  title={LLaDA-MoE: A Sparse MoE Diffusion Language Model},
  author={Zhu, Fengqi and You, Zebin and Xing, Yipeng and Huang, Zenan and Liu, Lin and Zhuang, Yihong and Lu, Guoshan and Wang, Kangyu and Wang, Xudong and Wei, Lanning and Guo, Hongrui and Hu, Jiaqi and Ye, Wentao and Chen, Tieyuan and Li, Chenchen and Tang, Chengfu and Feng, Haibo and Hu, Jun and Zhou, Jun and Zhang, Xiaolu and Lan, Zhenzhong and Zhao, Junbo and Zheng, Da and Li, Chongxuan and Li, Jianguo and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2509.24389},
  year={2025}
}

@article{khanna2025mercury,
  title={Mercury: Ultra-Fast Language Models Based on Diffusion},
  author={Khanna, Samar and Kharbanda, Siddhant and Li, Shufan and Varma, Harshit and Wang, Eric and Birnbaum, Sawyer and Luo, Ziyang and Miraoui, Yanis and Palrecha, Akash and Ermon, Stefano and Grover, Aditya and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2506.17298},
  year={2025}
}

@article{diffusiongemma2026,
  title={DiffusionGemma Technical Report},
  author={{DiffusionGemma Team, Google DeepMind}},
  journal={arXiv preprint arXiv:2608.00146},
  year={2026}
}

@inproceedings{arriola2025blockdiffusion,
  title={Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models},
  author={Arriola, Marianne and Gokaslan, Aaron Kerem and Chiu, Justin T. and Yang, Zhihan and Qi, Zhixuan and Han, Jiaqi and Sahoo, Subham Sekhar and Kuleshov, Volodymyr},
  booktitle={International Conference on Learning Representations (ICLR)},
  journal={arXiv preprint arXiv:2503.09573},
  year={2025}
}

@article{bertolani2026experimental,
  title={Diffusion Language Models: An Experimental Analysis},
  author={Bertolani, Thomas and Bucciarelli, Davide and Zini, Leonardo and Cornia, Marcella and Baraldi, Lorenzo},
  journal={arXiv preprint arXiv:2606.19475},
  year={2026}
}

@article{zhao2025d1,
  title={d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning},
  author={Zhao, Siyan and Gupta, Devaansh and Zheng, Qinqing and Grover, Aditya},
  journal={arXiv preprint arXiv:2504.12216},
  year={2025}
}

@article{wu2025fastdllm,
  title={Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding},
  author={Wu, Chengyue and Zhang, Hao and Xue, Shuchen and Liu, Zhijian and Diao, Shizhe and Zhu, Ligeng and Luo, Ping and Han, Song and Xie, Enze},
  journal={arXiv preprint arXiv:2505.22618},
  year={2025}
}

@article{zhu2025llada15,
  title={LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models},
  author={Zhu, Fengqi and Wang, Rongzhen and Nie, Shen and Zhang, Xiaolu and Wu, Chunwei and Hu, Jun and Zhou, Jun and Chen, Jianfei and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2505.19223},
  year={2025}
}

@inproceedings{nie2025smdm,
  title={Scaling up Masked Diffusion Models on Text},
  author={Nie, Shen and Zhu, Fengqi and Du, Chao and Pang, Tianyu and Liu, Qian and Zeng, Guangtao and Lin, Min and Li, Chongxuan},
  booktitle={International Conference on Learning Representations (ICLR)},
  journal={arXiv preprint arXiv:2410.18514},
  year={2025}
}

@article{wang2025tracerl,
  title={Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models},
  author={Wang, Yinjie and Yang, Ling and Li, Bowen and Tian, Ye and Shen, Ke and Wang, Mengdi},
  journal={arXiv preprint arXiv:2509.06949},
  year={2025}
}

@article{qian2026dynhd,
  title={DynHD: Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning},
  author={Qian, Yanyu and Tan, Yue and Liu, Yixin and Yu, Wang and Pan, Shirui},
  journal={arXiv preprint arXiv:2603.16459},
  year={2026}
}

% --- Checked and excluded (perplexity-only / no AR-diffusion QA comparison) ---

@article{lou2023sedd,
  title={Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2023}
}

@article{sahoo2024mdlm,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Sahoo, Subham Sekhar and Arriola, Marianne and Schiff, Yair and Gokaslan, Aaron and Marroquin, Edgar and Chiu, Justin T. and Rush, Alexander and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2406.07524},
  year={2024}
}

@article{gulrajani2023plaid,
  title={Likelihood-Based Diffusion Language Models},
  author={Gulrajani, Ishaan and Hashimoto, Tatsunori B.},
  journal={arXiv preprint arXiv:2305.18619},
  year={2023}
}
```

## Notable Cross-Cutting Observations

1. **No paper in this audit uses an NLI/entailment classifier** for scoring — count is 0/14. Judges split between exact-match/regex (6), LLM-as-judge (2 strict / 3 loose), and several "not reported."
2. **Matched generation budgets between AR and diffusion are rare.** Only Block Diffusion applies an identical length sweep (L=1024/2048) to both; DiffuLLaMA matches the *target* length via oracle length but through different mechanisms. Several papers (LLaDA-MoE, TraDo) report the diffusion model's generation length but leave the AR baseline's budget unstated or cite externally reported AR numbers rather than re-running AR under the same conditions — this is a direct, well-documented confound relevant to this audit's thesis.
3. **Length-sensitivity findings are mixed**: LLaDA explicitly claims insensitivity to generation length (256–1024), while the independent "Experimental Analysis" paper (#8) finds the opposite — performance "saturates or declines beyond 256–512 tokens" and that translation quality "collapses at the largest budgets." This direct disagreement between a model-developer paper and an independent third-party analysis is itself a key citable point for an evaluation-confounds paper.
4. **Five of fourteen papers run no AR baseline at all**, instead comparing only among diffusion model variants (d1, Fast-dLLM, LLaDA 1.5, DynHD) or citing externally reported AR numbers rather than a controlled re-run (LLaDA-MoE, TraceRL/TraDo) — weakening the evidentiary strength of claimed AR-vs-diffusion superiority claims drawn from the broader literature.
