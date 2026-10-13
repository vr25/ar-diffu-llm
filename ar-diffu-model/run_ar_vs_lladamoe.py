#!/usr/bin/env python3
"""Local version of Colab_AR_vs_Diffusion_Full.ipynb (LLaDA-MoE-7B-A1B-Instruct).

Much smaller than LLaDA2.1-mini (7B total / ~1.4B active vs 16B total), and
runs on transformers>=4.53.2 (no separate llada-env needed). Trades off a
built-in generate() for a manual block-wise unmasking loop (see
generate_lladamoe_responses in ar_diffu_local.py).

    python run_ar_vs_lladamoe.py --limit 3            # smoke test
    python run_ar_vs_lladamoe.py                      # full 9 comparisons
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ar_diffu_local import build_parser, run, generate_lladamoe_responses


def load_lladamoe(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.to(device).eval()
    return model, tok


if __name__ == "__main__":
    args = build_parser(
        "AR (Qwen2.5-Instruct 0.5B/1.5B/3B) vs LLaDA-MoE-7B-A1B-Instruct @ 3 temps",
        default_diffu_model="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        default_out_dir="results_lladamoe",
    ).parse_args()
    run(args, "lladamoe", load_lladamoe, generate_lladamoe_responses)
