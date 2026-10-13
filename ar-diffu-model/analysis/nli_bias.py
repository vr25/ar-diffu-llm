"""NLI-judge bias analysis (CPU). Scores AR and diffusion factual outputs with the
NLI cross-encoder and compares against LLM-judge labels.

    HF_HUB_OFFLINE=1 python analysis/nli_bias.py   # writes analysis/nli_bias.json
"""
import glob, json
import numpy as np
from sentence_transformers import CrossEncoder

ce = CrossEncoder("cross-encoder/qnli-distilroberta-base", device="cpu")
jd = json.load(open("llm_judge_merged.json"))
ja = json.load(open("llm_judge_ar_merged.json"))
ja = ja if isinstance(ja, dict) else {x["id"]: x for x in ja}

rows = []  # (model, words, correct, nli)
for m, d in [("lladamoe", "results_lladamoe"), ("dream", "results_dream"), ("minicpm5", "results_minicpm5_local")]:
    for t in (50, 70, 90):
        for i, c in enumerate(json.load(open(f"{d}/ar-qwen25-0.5b-{t}-diffu-{m}-{t}.json"))["comparisons"]):
            rows.append((m, len(c["diffu_response"].split()), jd[f"{m}|{t / 100}|{i}"]["correct"], c["nli_score"]))
for a in ("qwen25-0.5b", "qwen25-1.5b", "qwen25-3b"):
    for t in (50, 70, 90):
        c = json.load(open(f"results_lladamoe/ar-{a}-{t}-diffu-lladamoe-{t}.json"))["comparisons"]
        s = ce.predict([[x["ar_response"], x["question"]] for x in c])
        for i, (x, sc) in enumerate(zip(c, s)):
            rows.append((a, len(x["ar_response"].split()), ja[f"{a}|{t / 100}|{i}"]["correct"], float(sc)))

out = {"per_model": {}, "pass_rate_given_correct_by_words": {}}
for m in dict.fromkeys(r[0] for r in rows):
    r = [x for x in rows if x[0] == m]
    c = [x for x in r if x[2]]
    out["per_model"][m] = {"n": len(r), "nli_pct": 100 * np.mean([x[3] >= 0.5 for x in r]),
                           "llm_pct": 100 * np.mean([x[2] for x in r]),
                           "nli_pass_given_correct": 100 * np.mean([x[3] >= 0.5 for x in c])}
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 10**6)]:
    c = [x for x in rows if x[2] and lo <= x[1] < hi]
    out["pass_rate_given_correct_by_words"][f"{lo}-{hi}"] = {"n": len(c), "pct": 100 * np.mean([x[3] >= 0.5 for x in c])}
json.dump(out, open("analysis/nli_bias.json", "w"), indent=1)
print(json.dumps(out, indent=1))
