#!/usr/bin/env python3
"""
Bootstrap confidence intervals + significance testing for AR-vs-diffusion results.

Reads the per-comparison JSON files written by ar_diffu_local.py's
compare_and_save() (e.g. results_lladamoe/ar-qwen25-0.5b-50-diffu-lladamoe-50.json)
and reports, for each diffusion model:
  - overall truthful_pct with a 95% bootstrap CI (resampled over questions)
  - per AR-model / per-temperature breakdowns with CIs
  - paired bootstrap test for "AR beats diffusion" (and vice versa)
Across multiple diffusion result dirs, also runs a paired bootstrap comparing
diffusion models against each other (same question set => valid pairing).

Usage:
    python bootstrap_ci.py results_lladamoe results_dream results_minicpm5_local
    python bootstrap_ci.py results_lladamoe --n-boot 20000
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def load_group(result_dir):
    """Load all comparison files in a dir; return list of per-file records."""
    files = sorted(f for f in glob.glob(os.path.join(result_dir, "*.json")) if not f.endswith(".tmp"))
    if not files:
        print(f"✗ no result files found in {result_dir}", file=sys.stderr)
        return []
    records = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        records.append(data)
    return records


def truthful_vec(comparisons):
    """1 = truthful, 0 = hallucinated, per question, in question order."""
    return np.array([0 if c["hallucinated"] else 1 for c in comparisons], dtype=float)


def bootstrap_ci(vec, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap CI for the mean of a 0/1 vector (resample over questions)."""
    rng = np.random.default_rng(seed)
    n = len(vec)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = vec[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return vec.mean(), lo, hi


def paired_bootstrap_diff(vec_a, vec_b, n_boot=10000, seed=0):
    """
    Paired bootstrap on (mean(vec_a) - mean(vec_b)); vec_a/vec_b must be aligned
    on the same underlying questions. Returns (diff, lo, hi, p_two_sided).
    p is the fraction of bootstrap resamples where the sign of the diff flips
    relative to the observed diff (a simple bootstrap-based two-sided p-value).
    """
    assert len(vec_a) == len(vec_b)
    rng = np.random.default_rng(seed)
    n = len(vec_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = vec_a[idx].mean(axis=1) - vec_b[idx].mean(axis=1)
    obs = vec_a.mean() - vec_b.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value via bootstrap: proportion of resamples on the other side of 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = min(p, 1.0)
    return obs, lo, hi, p


def analyze_dir(result_dir, n_boot):
    records = load_group(result_dir)
    if not records:
        return None
    diffu_name = records[0]["diffu_model"]

    print(f"\n{'='*80}\n{result_dir}  (diffusion model: {diffu_name})\n{'='*80}")

    all_truthful = np.concatenate([truthful_vec(r["comparisons"]) for r in records])
    mean, lo, hi = bootstrap_ci(all_truthful, n_boot=n_boot)
    n_q = len(all_truthful)
    print(f"\nOverall AR truthful-vs-diffu rate (n={n_q} comparisons): "
          f"{mean*100:.1f}%  [95% CI {lo*100:.1f}%, {hi*100:.1f}%]")

    print(f"\n{'AR model':<14}{'Temp':>6}{'Truthful %':>13}{'95% CI':>18}{'n':>6}")
    per_group = {}
    for r in records:
        vec = truthful_vec(r["comparisons"])
        mean, lo, hi = bootstrap_ci(vec, n_boot=n_boot)
        key = (r["ar_model"], r["ar_temperature"])
        per_group[key] = vec
        print(f"{r['ar_model']:<14}{r['ar_temperature']:>6}{mean*100:>12.1f}%"
              f"   [{lo*100:5.1f}%, {hi*100:5.1f}%]{len(vec):>6}")

    # AR model with most temps pooled vs weakest, paired bootstrap
    by_ar = {}
    for r in records:
        by_ar.setdefault(r["ar_model"], []).append(truthful_vec(r["comparisons"]))
    by_ar = {k: np.concatenate(v) for k, v in by_ar.items()}
    ar_names = sorted(by_ar, key=lambda k: by_ar[k].mean())
    if len(ar_names) >= 2:
        weakest, strongest = ar_names[0], ar_names[-1]
        obs, lo, hi, p = paired_bootstrap_diff(by_ar[strongest], by_ar[weakest], n_boot=n_boot)
        sig = "significant" if p < 0.05 else "not significant"
        print(f"\n{strongest} vs {weakest} (best vs weakest AR model): "
              f"Δ={obs*100:+.1f}pp  [95% CI {lo*100:+.1f}, {hi*100:+.1f}]  p≈{p:.3f} ({sig})")

    return {"diffu_model": diffu_name, "dir": result_dir, "by_ar": by_ar, "n_q": n_q}


def compare_diffusion_models(groups, n_boot):
    """Paired bootstrap across diffusion models, pooling all AR-model comparisons
    (valid pairing since all diffusion models are judged on the same question set)."""
    valid = [g for g in groups if g]
    if len(valid) < 2:
        return
    print(f"\n{'='*80}\nCROSS-MODEL: diffusion model comparison (pooled across AR models/temps)\n{'='*80}")
    pooled = {}
    for g in valid:
        pooled[g["diffu_model"]] = np.concatenate(list(g["by_ar"].values()))
    names = sorted(pooled, key=lambda k: -pooled[k].mean())
    print(f"\n{'Diffusion model':<20}{'Truthful %':>13}{'95% CI':>18}{'n':>8}")
    for name in names:
        mean, lo, hi = bootstrap_ci(pooled[name], n_boot=n_boot)
        print(f"{name:<20}{mean*100:>12.1f}%   [{lo*100:5.1f}%, {hi*100:5.1f}%]{len(pooled[name]):>8}")

    print()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            va, vb = pooled[a], pooled[b]
            n = min(len(va), len(vb))
            obs, lo, hi, p = paired_bootstrap_diff(va[:n], vb[:n], n_boot=n_boot)
            sig = "significant" if p < 0.05 else "not significant"
            print(f"{a} vs {b}: Δ={obs*100:+.1f}pp  [95% CI {lo*100:+.1f}, {hi*100:+.1f}]  p≈{p:.3f} ({sig})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dirs", nargs="+", help="One or more results_* directories")
    ap.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap resamples")
    args = ap.parse_args()

    groups = [analyze_dir(d, args.n_boot) for d in args.result_dirs]
    compare_diffusion_models(groups, args.n_boot)


if __name__ == "__main__":
    main()
