#!/usr/bin/env python3
"""Generate AR-model responses only (no diffusion model, no NLI judge).

    python run_ar_only.py --ar-models qwen25-0.5b --question-set gsm8k --temps 0.7 --gen-tokens 256
    python run_ar_only.py --ar-models qwen25-3b --question-set reasoning --temps 0 --gen-tokens 150  # greedy
"""
import argparse
import json
import os
import time

from ar_diffu_local import AR_MODELS, QUESTION_SETS, free_memory, generate_responses, load_ar, pick_device

p = argparse.ArgumentParser()
p.add_argument("--ar-models", nargs="+", required=True, choices=list(AR_MODELS))
p.add_argument("--question-set", required=True, choices=list(QUESTION_SETS))
p.add_argument("--temps", type=float, nargs="+", default=[0.7], help="0 = greedy")
p.add_argument("--gen-tokens", type=int, required=True)
p.add_argument("--out-dir", default="results_ar_only")
p.add_argument("--device", default="auto")
args = p.parse_args()

device = pick_device(args.device)
questions = QUESTION_SETS[args.question_set]
os.makedirs(args.out_dir, exist_ok=True)
for name in args.ar_models:
    todo = [t for t in args.temps if not os.path.exists(
        os.path.join(args.out_dir, f"{name}-{args.question_set}-k{args.gen_tokens}-t{int(t * 100)}.json"))]
    if not todo:
        continue
    model, tok = load_ar(AR_MODELS[name], device)
    for t in todo:
        out = os.path.join(args.out_dir, f"{name}-{args.question_set}-k{args.gen_tokens}-t{int(t * 100)}.json")
        t0 = time.time()
        res = generate_responses(model, tok, t, questions, device, label=f"{name} T={t}", max_new_tokens=args.gen_tokens)
        json.dump({"ar_model": name, "temperature": t, "question_set": args.question_set,
                   "gen_tokens": args.gen_tokens, "results": res}, open(out, "w"), indent=1)
        print(f"✓ {out} ({time.time() - t0:.0f}s)", flush=True)
    del model, tok
    free_memory(device)
