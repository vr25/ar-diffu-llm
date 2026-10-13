"""Simulated token-budget sweep for AR models (CPU only; no generation).

AR decoding is prefix-consistent: generating with max_new_tokens=k yields the
first k tokens of the longer generation (exactly for greedy, in distribution
for sampling). We truncate the saved 150-token Qwen2.5 outputs at k tokens and
count an item correct iff (a) the LLM judge marked the full output correct and
(b) the reference answer already appears in the k-token prefix.
"""
import csv, json, re, random
from transformers import AutoTokenizer

BUDGETS = [20, 30, 40, 50, 60, 80, 100, 120, 150]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)

refs = {r["question"].strip(): r["reference_answer"].strip()
        for r in csv.DictReader(open("questions_and_answers.csv")) if r["type"] != "factual"}

def pattern(ref):
    t = re.search(r"^(\d{1,2})\s*(AM|PM)$", ref, re.I)
    if t:
        return rf"(?<!\d){t[1]}(:00)?\s*({t[2][0]}\.?\s*M\.?)\b"
    if ref.startswith("January"):
        return r"January\s+11"
    m = re.search(r"\d+(\.\d+)?", ref)
    if m:
        n = m.group(0)
        alt = rf"|25\s*cents" if n == "0.25" else ""
        return rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d){alt}"
    return rf"\b{re.escape(ref)}\b"

pats = {q: re.compile(pattern(a), re.I) for q, a in refs.items()}

items = []
for b in range(9):
    items += json.load(open(f"llm_judge_ar_reasoning_v2_batch_{b:02d}.json"))
verdict = {x["id"]: x["correct"] for x in json.load(open("llm_judge_ar_reasoning_v2_merged.json"))}

def boot(v, n=5000, seed=0):
    r = random.Random(seed); m = len(v)
    s = sorted(sum(r.choice(v) for _ in range(m)) / m for _ in range(n))
    return 100 * s[int(.025 * n)], 100 * s[int(.975 * n) - 1]

models = ["qwen25-0.5b", "qwen25-1.5b", "qwen25-3b"]
out = {}
for mdl in models:
    rows = [x for x in items if x["ar_model"] == mdl]
    out[mdl] = {}
    for k in BUDGETS:
        v = []
        for x in rows:
            ids = tok(x["answer"], add_special_tokens=False)["input_ids"][:k]
            prefix = tok.decode(ids)
            v.append(int(verdict[x["id"]] and bool(pats[x["question"].strip()].search(prefix))))
        lo, hi = boot(v)
        out[mdl][k] = {"acc": 100 * sum(v) / len(v), "lo": lo, "hi": hi, "n": len(v)}
    print(mdl, " ".join(f"{k}:{out[mdl][k]['acc']:.1f}" for k in BUDGETS))

json.dump(out, open("analysis/budget_sweep.json", "w"), indent=1)
