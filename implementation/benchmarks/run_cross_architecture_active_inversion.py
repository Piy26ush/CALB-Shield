#!/usr/bin/env python3
"""
run_cross_architecture_active_inversion.py
RQ1 Multi-Architecture Baseline Evaluation.

Executes the Active Universal Shortcut Inversion benchmark across multiple
unseen architectures:
  1. Meta-Llama-3-8B-Instruct (Clean LLaMA)
  2. Mistral-7B-Instruct-v0.2 (Clean Mistral)
  3. Qwen2.5-Coder-1.5B-Instruct (Clean Qwen)
  4. Qwen2.5-Coder-1.5B-Backdoored-PoC (Poisoned Qwen)

Conditions:
  - Exact same candidate generation procedure
  - Exact same 5-anchor set
  - Exact same UAS calculation: UAS(t) = C(t) * (mean_gap - 1.5 * mean_H)
  - Zero target-architecture calibration
  - Zero reference twin
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from run_active_trigger_inversion import (
    ANCHOR_PROMPTS,
    run_trigger_inversion_scan,
    resolve_path,
)

MODELS_CONFIG = [
    {
        "id": "clean_llama3",
        "name": "Clean LLaMA-3-8B",
        "arch": "llama3",
        "path": "models.nosync/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "role": "clean"
    },
    {
        "id": "clean_mistral",
        "name": "Clean Mistral-7B",
        "arch": "mistral",
        "path": "models.nosync/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "role": "clean"
    },
    {
        "id": "clean_qwen",
        "name": "Clean Qwen-1.5B",
        "arch": "qwen",
        "path": "models.nosync/qwen/qwen2.5-coder-1.5b-instruct-q8_0.gguf",
        "role": "clean"
    },
    {
        "id": "poison_qwen",
        "name": "Poisoned Qwen-1.5B PoC",
        "arch": "qwen",
        "path": "models.nosync/qwen/qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf",
        "role": "poisoned"
    }
]

def load_cached_qwen_results(existing_json_path: str) -> Dict[str, Any]:
    """Load pre-computed physical Qwen scan results if available."""
    abs_path = resolve_path(existing_json_path)
    if os.path.exists(abs_path):
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "clean_qwen": data.get("clean_model_results"),
                "poison_qwen": data.get("poison_model_results")
            }
        except Exception as e:
            print(f"[WARN] Failed to load cached Qwen results: {e}")
    return {}

def run_cross_architecture_benchmark(
    output_json: str = "results/physical_benchmarks/cross_architecture_inversion_results.json",
    output_csv: str = "results/physical_benchmarks/cross_architecture_inversion_results.csv",
    existing_qwen_json: str = "results/physical_benchmarks/trigger_inversion_results.json",
    recompute_all: bool = False
) -> Dict[str, Any]:
    print("=" * 84)
    print("RQ1 MULTI-ARCHITECTURE ACTIVE SHORTCUT INVERSION BENCHMARK")
    print("=" * 84)
    print("Evaluating physical checkpoints across LLaMA-3, Mistral-7B, and Qwen-1.5B.")
    print("Constraints: ZERO target calibration, ZERO clean twin, IDENTICAL 5-anchor protocol.\n")

    cached_qwen = {} if recompute_all else load_cached_qwen_results(existing_qwen_json)
    all_results = {}

    for cfg in MODELS_CONFIG:
        m_id = cfg["id"]
        m_name = cfg["name"]
        m_arch = cfg["arch"]
        m_path = cfg["path"]

        # Check if cached Qwen result is valid
        if m_id in cached_qwen and cached_qwen[m_id] is not None:
            print(f"[CACHE] Using verified physical scan results for {m_name}")
            all_results[m_id] = cached_qwen[m_id]
            all_results[m_id]["display_name"] = m_name
            all_results[m_id]["role"] = cfg["role"]
            continue

        print(f"\n{'='*40}")
        print(f"RUNNING SCAN: {m_name} (Architecture: {m_arch})")
        print(f"{'='*40}")
        scan_res = run_trigger_inversion_scan(
            model_path=m_path,
            architecture=m_arch,
            n_gpu_layers=1,
            lambda_ent=1.5,
            top_k_logprobs=20
        )
        scan_res["display_name"] = m_name
        scan_res["role"] = cfg["role"]
        all_results[m_id] = scan_res

    # --------------------------------------------------------------------------
    # 2. Compile Cross-Architecture Comparison Metrics
    # --------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("CROSS-ARCHITECTURE ACTIVE INVERSION MATRIX: CLEAN vs. POISONED BASELINE")
    print("=" * 96)

    table_rows = []
    for cfg in MODELS_CONFIG:
        m_id = cfg["id"]
        res = all_results[m_id]
        stats = res["search_statistics"]
        top_cand = res["top_10_candidates"][0]

        # Calculate mean entropy across all tested candidate suffixes
        all_cand_entropies = [c["mean_entropy"] for c in res["top_10_candidates"]]
        mean_h_top10 = float(np.mean(all_cand_entropies))

        table_rows.append({
            "model_id": m_id,
            "display_name": cfg["name"],
            "architecture": cfg["arch"],
            "role": cfg["role"].upper(),
            "uas_max": stats["max_uas"],
            "mean_uas": stats["mean_uas"],
            "mean_entropy": round(top_cand["mean_entropy"], 4),
            "min_entropy": stats["top_shortcut_min_entropy"],
            "mean_logit_gap": round(top_cand["mean_logit_gap"], 4),
            "consistency_rate": round(stats["top_shortcut_consistency"] * 100, 1),
            "top_suffix": repr(stats["top_shortcut_suffix"]),
            "top_target": repr(stats["top_shortcut_target"]),
            "n_candidates": res["n_candidates_evaluated"]
        })

    df_summary = pd.DataFrame(table_rows)

    header = f"{'MODEL':<24} | {'ROLE':<8} | {'UAS_MAX':<8} | {'MEAN_UAS':<8} | {'MEAN_H':<8} | {'MIN_H':<8} | {'MEAN_GAP':<8} | {'CONS(%)':<8} | {'TOP SUFFIX':<12}"
    print(header)
    print("-" * len(header))
    for _, row in df_summary.iterrows():
        print(
            f"{row['display_name']:<24} | {row['role']:<8} | {row['uas_max']:<8.4f} | {row['mean_uas']:<8.4f} | "
            f"{row['mean_entropy']:<8.4f} | {row['min_entropy']:<8.4f} | {row['mean_logit_gap']:<8.4f} | "
            f"{row['consistency_rate']:<8.1f} | {row['top_suffix']:<12}"
        )

    # Top-3 Discovered Suffixes Per Architecture
    print("\n" + "-" * 96)
    print("TOP-3 DISCOVERED CANDIDATE SHORTCUTS PER ARCHITECTURE")
    print("-" * 96)
    for cfg in MODELS_CONFIG:
        m_id = cfg["id"]
        res = all_results[m_id]
        print(f"\n{cfg['name']} ({cfg['arch'].upper()} - {cfg['role'].upper()}):")
        for i, c in enumerate(res["top_10_candidates"][:3], 1):
            print(
                f"  #{i}: Suffix {repr(c['candidate_suffix']):<12} -> Target {repr(c['target_token']):<10} | "
                f"UAS: {c['uas_score']:<6.2f} | Cons: {c['consistency_rate']*100:4.1f}% | "
                f"Mean Gap: {c['mean_logit_gap']:4.2f} | Mean H: {c['mean_entropy']:4.4f} | Min H: {c['min_entropy']:4.4f}"
            )

    # --------------------------------------------------------------------------
    # 3. Save Summary Artifacts
    # --------------------------------------------------------------------------
    abs_csv = resolve_path(output_csv)
    abs_json = resolve_path(output_json)
    os.makedirs(os.path.dirname(abs_csv), exist_ok=True)
    os.makedirs(os.path.dirname(abs_json), exist_ok=True)

    df_summary.to_csv(abs_csv, index=False)

    payload = {
        "metadata": {
            "experiment": "RQ1 Cross-Architecture Active Trigger Inversion Benchmark",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_anchors": len(ANCHOR_PROMPTS),
            "anchors": ANCHOR_PROMPTS,
            "lambda_entropy_penalty": 1.5,
            "hardware": "Apple Silicon MPS / llama_cpp",
            "protocol": "Zero reference twin, zero target calibration"
        },
        "summary_table": df_summary.to_dict(orient="records"),
        "detailed_results_by_model": all_results
    }

    with open(abs_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[DONE] Cross-architecture artifacts saved to:")
    print(f"  CSV:  {abs_csv}")
    print(f"  JSON: {abs_json}")
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Architecture Active Inversion Benchmark")
    parser.add_argument("--output-json", type=str, default="results/physical_benchmarks/cross_architecture_inversion_results.json")
    parser.add_argument("--output-csv", type=str, default="results/physical_benchmarks/cross_architecture_inversion_results.csv")
    parser.add_argument("--recompute-all", action="store_true", help="Force recomputing Qwen models as well")
    args = parser.parse_args()

    run_cross_architecture_benchmark(
        output_json=args.output_json,
        output_csv=args.output_csv,
        recompute_all=args.recompute_all
    )
