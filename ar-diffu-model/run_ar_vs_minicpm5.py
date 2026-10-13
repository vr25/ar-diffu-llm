#!/usr/bin/env python3
"""Local version of Colab_AR_vs_MiniCPM5_Full.ipynb (MiniCPM5-2B-Diffusion-Base, ~2.5B params).

    python run_ar_vs_minicpm5.py --limit 3         # smoke test
    python run_ar_vs_minicpm5.py                   # full 9 comparisons
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ar_diffu_local import build_parser, run, generate_responses, half_dtype


def load_minicpm5(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=half_dtype(device, torch.float16), trust_remote_code=True, low_cpu_mem_usage=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(device).eval()
    return model, tok


if __name__ == "__main__":
    args = build_parser(
        "AR (gpt2/gpt2-medium/distilgpt2) vs MiniCPM5-2B-Diffusion-Base @ 3 temps",
        default_diffu_model="fwerkor/MiniCPM5-2B-Diffusion-Base",
        default_out_dir="results_minicpm5_local",
    ).parse_args()
    run(args, "minicpm5", load_minicpm5, generate_responses)
