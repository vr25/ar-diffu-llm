#!/usr/bin/env python3
"""Local version of Colab_AR_vs_Diffusion_Full.ipynb (Dream-v0-Instruct-7B).

Dream 7B (HKUNLP + Huawei Noah's Ark Lab, arXiv:2508.15487) is a dense 7B
discrete text-diffusion LLM -- heavier per step than LLaDA-MoE's 1.4B active
params, so `--steps`/`--max-new-tokens` are kept low for speed (see
generate_dream_responses in ar_diffu_local.py).

    python run_ar_vs_dream.py --limit 3            # smoke test
    python run_ar_vs_dream.py                      # full 9 comparisons
"""
import torch
from transformers import AutoTokenizer, AutoModel

from ar_diffu_local import build_parser, run, generate_dream_responses


def load_dream(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.to(device).eval()
    return model, tok


if __name__ == "__main__":
    args = build_parser(
        "AR (Qwen2.5-Instruct 0.5B/1.5B/3B) vs Dream-v0-Instruct-7B @ 3 temps",
        default_diffu_model="Dream-org/Dream-v0-Instruct-7B",
        default_out_dir="results_dream",
    ).parse_args()
    run(args, "dream", load_dream, generate_dream_responses)
