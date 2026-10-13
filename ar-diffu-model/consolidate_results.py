#!/usr/bin/env python3
"""
Consolidate AR vs Diffusion comparison results from 9 JSON files into a single file.

Usage:
    python consolidate_results.py results_llada
    python consolidate_results.py results_qwen3_local
    python consolidate_results.py results_minicpm5

Creates:
    - {folder}_consolidated.json (consolidated data with stats)
    - {folder}_summary.csv (summary table for easy viewing)
"""

import json
import os
import pandas as pd
from collections import defaultdict
import sys

def consolidate_results(results_dir):
    """Consolidate 9 JSON files from a results directory."""

    print(f"\n{'='*80}")
    print(f"Consolidating: {results_dir}/")
    print(f"{'='*80}\n")

    # Check if directory exists
    if not os.path.isdir(results_dir):
        print(f"✗ Directory not found: {results_dir}/")
        return

    # Load all JSON files
    files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')])
    if len(files) != 9:
        print(f"⚠ Warning: Expected 9 files, found {len(files)}")

    print(f"Loading {len(files)} files...")

    all_comparisons = []
    file_stats = {}
    diffusion_model = None

    for filename in files:
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract metadata
            ar_model = data['ar_model']
            ar_temp = data['ar_temperature']
            diffu_model = data['diffu_model']
            diffu_temp = data['diffu_temperature']
            truthful_pct = data['truthful_pct']

            diffusion_model = diffu_model

            # Store comparisons with metadata
            for comp in data['comparisons']:
                comp['_ar_model'] = ar_model
                comp['_ar_temp'] = ar_temp
                comp['_diffu_temp'] = diffu_temp
                all_comparisons.append(comp)

            file_stats[filename] = {
                'ar_model': ar_model,
                'ar_temp': ar_temp,
                'diffu_temp': diffu_temp,
                'truthful_pct': truthful_pct,
                'comparisons': len(data['comparisons'])
            }

            print(f"  ✓ {filename}: {truthful_pct:.1f}% truthful")

        except Exception as e:
            print(f"  ✗ {filename}: {e}")

    print(f"\n✓ Loaded {len(all_comparisons)} total comparisons\n")

    # Calculate summary statistics
    print("Calculating statistics...")

    # By AR model and temperature
    stats_by_ar_temp = defaultdict(lambda: {'hallucinated': 0, 'total': 0, 'scores': []})

    for comp in all_comparisons:
        key = (comp['_ar_model'], comp['_ar_temp'])
        stats_by_ar_temp[key]['total'] += 1
        stats_by_ar_temp[key]['scores'].append(comp['nli_score'])
        if comp['hallucinated']:
            stats_by_ar_temp[key]['hallucinated'] += 1

    # Convert to readable format
    summary_stats = {}
    for (ar_model, ar_temp), stats in sorted(stats_by_ar_temp.items()):
        truthful = stats['total'] - stats['hallucinated']
        truthful_pct = 100 * truthful / stats['total']
        avg_score = sum(stats['scores']) / len(stats['scores'])

        key = f"{ar_model}_@{ar_temp}"
        summary_stats[key] = {
            'ar_model': ar_model,
            'ar_temperature': ar_temp,
            'truthful_count': truthful,
            'hallucinated_count': stats['hallucinated'],
            'truthful_pct': round(truthful_pct, 1),
            'avg_nli_score': round(avg_score, 3)
        }
        print(f"  {key}: {truthful_pct:.1f}% truthful (avg NLI: {avg_score:.3f})")

    # Overall statistics
    overall_halluc = sum(1 for c in all_comparisons if c['hallucinated'])
    overall_truthful = len(all_comparisons) - overall_halluc
    overall_truthful_pct = 100 * overall_truthful / len(all_comparisons)

    print(f"\n  OVERALL: {overall_truthful_pct:.1f}% truthful ({overall_truthful}/{len(all_comparisons)})")

    # Create consolidated JSON
    consolidated = {
        'diffusion_model': diffusion_model,
        'total_comparisons': len(all_comparisons),
        'summary': {
            'overall_truthful_pct': round(overall_truthful_pct, 1),
            'overall_hallucinated_count': overall_halluc,
            'by_ar_model_temp': summary_stats
        },
        'comparisons': all_comparisons
    }

    # Save consolidated JSON
    output_json = f"{results_dir}_consolidated.json"
    with open(output_json, 'w') as f:
        json.dump(consolidated, f, indent=2)
    print(f"\n✓ Saved: {output_json}")

    # Create summary CSV for easy viewing
    summary_rows = []
    for key, stats in summary_stats.items():
        summary_rows.append({
            'AR Model': stats['ar_model'],
            'AR Temp': stats['ar_temperature'],
            'Truthful %': stats['truthful_pct'],
            'Hallucinated': stats['hallucinated_count'],
            'Avg NLI Score': stats['avg_nli_score']
        })

    df_summary = pd.DataFrame(summary_rows)
    output_csv = f"{results_dir}_summary.csv"
    df_summary.to_csv(output_csv, index=False)
    print(f"✓ Saved: {output_csv}")

    # Print summary table
    print(f"\n{results_dir.upper()} SUMMARY:")
    print(f"{'-'*80}")
    print(df_summary.to_string(index=False))
    print(f"{'-'*80}")

    print(f"\n✓ Consolidation complete!\n")

    return consolidated

def main():
    if len(sys.argv) < 2:
        print("Usage: python consolidate_results.py <results_directory>")
        print("Example: python consolidate_results.py results_llada")
        sys.exit(1)

    results_dir = sys.argv[1]
    consolidate_results(results_dir)

if __name__ == "__main__":
    main()
