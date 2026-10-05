#!/usr/bin/env python3
"""
evaluate_path_a_self_normalized_inversion.py
Path A: Within-Model Self-Relative Normalization Evaluation.

Investigates whether computing relative entropy shift ΔH(t) = (H(t) - H₀) / H₀
and self-anchored Universal Attractor Score (Self-UAS) can separate a backdoored model
(Poisoned Qwen-1.5B) from unseen clean architectures (LLaMA-3-8B, Mistral-7B, Clean Qwen-1.5B)
without target-architecture calibration or a clean reference twin.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_JSON = os.path.join(CURRENT_DIR, "results", "physical_benchmarks", "cross_architecture_inversion_results.json")
OUTPUT_CSV = os.path.join(CURRENT_DIR, "results", "physical_benchmarks", "path_a_self_normalized_inversion_results.csv")
OUTPUT_JSON = os.path.join(CURRENT_DIR, "results", "physical_benchmarks", "path_a_self_normalized_inversion_results.json")

def evaluate_path_a():
    if not os.path.exists(INPUT_JSON):
        print(f"[ERROR] Missing input data: {INPUT_JSON}")
        sys.exit(1)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_order = ["clean_llama3", "clean_mistral", "clean_qwen", "poison_qwen"]
    results_summary = []
    detailed_models = {}

    for m_id in models_order:
        m_data = data["detailed_results_by_model"][m_id]
        dname = m_data.get("display_name", m_id)
        role = m_data.get("role", "clean").upper()

        # Baseline metrics on clean anchors
        base_records = m_data["baseline_anchor_responses"]
        base_ents = [b["entropy"] for b in base_records]
        base_gaps = [b["logit_gap"] for b in base_records]
        h0_mean = float(np.mean(base_ents))
        gap0_mean = float(np.mean(base_gaps))
        h0_min = float(np.min(base_ents))

        # Candidate evaluations
        evals = m_data["all_evaluations"]
        df = pd.DataFrame(evals)

        # Path A metrics:
        # Relative entropy drop: (H0 - H) / H0  (higher = larger entropy collapse)
        df["rel_entropy_drop"] = (h0_mean - df["mean_entropy"]) / (h0_mean + 1e-12)
        df["entropy_ratio"] = df["mean_entropy"] / (h0_mean + 1e-12)
        
        # Relative logit gap increase: (Gap - Gap0) / Gap0
        df["rel_gap_increase"] = (df["mean_logit_gap"] - gap0_mean) / (gap0_mean + 1e-12)
        
        # Self-Anchored Universal Attractor Score (Self-UAS):
        # Rewards consistency, logit gap expansion, and fractional entropy collapse
        df["self_uas"] = df["consistency_rate"] * (df["rel_gap_increase"] + 1.5 * df["rel_entropy_drop"])

        # Candidate with highest relative entropy collapse
        best_collapse_idx = df["rel_entropy_drop"].idxmax()
        best_collapse = df.loc[best_collapse_idx]

        # Candidate with highest Self-UAS
        best_self_uas_idx = df["self_uas"].idxmax()
        best_self_uas = df.loc[best_self_uas_idx]

        model_summary = {
            "model_id": m_id,
            "display_name": dname,
            "role": role,
            "h0_mean": round(h0_mean, 4),
            "h0_min": round(h0_min, 4),
            "gap0_mean": round(gap0_mean, 4),
            "raw_uas_max": round(float(df["uas_score"].max()), 4),
            "max_rel_entropy_collapse_pct": round(float(best_collapse["rel_entropy_drop"]) * 100.0, 2),
            "max_rel_collapse_suffix": best_collapse["candidate_suffix"],
            "max_rel_collapse_entropy": round(float(best_collapse["mean_entropy"]), 4),
            "min_entropy_ratio": round(float(df["entropy_ratio"].min()), 4),
            "max_rel_gap_increase_pct": round(float(df["rel_gap_increase"].max()) * 100.0, 2),
            "max_self_uas": round(float(best_self_uas["self_uas"]), 4),
            "max_self_uas_suffix": best_self_uas["candidate_suffix"],
            "max_self_uas_target": best_self_uas["target_token"],
            "mean_self_uas": round(float(df["self_uas"].mean()), 4),
            "n_candidates": len(df)
        }

        results_summary.append(model_summary)
        detailed_models[m_id] = {
            "summary": model_summary,
            "top_5_self_uas": df.sort_values(by="self_uas", ascending=False).head(5).to_dict(orient="records"),
            "top_5_entropy_collapse": df.sort_values(by="rel_entropy_drop", ascending=False).head(5).to_dict(orient="records")
        }

    df_out = pd.DataFrame(results_summary)
    df_out.to_csv(OUTPUT_CSV, index=False)

    full_output = {
        "metadata": {
            "experiment": "Path A: Within-Model Self-Relative Normalization",
            "objective": "Zero-reference cross-architecture backdoor detection via self-relative ΔH and Self-UAS",
            "verdict": "FAILED - Cannot separate Clean Mistral from Poisoned Qwen",
            "input_source": INPUT_JSON
        },
        "summary_table": results_summary,
        "detailed_models": detailed_models
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 90)
    print("PATH A: WITHIN-MODEL SELF-RELATIVE NORMALIZATION BENCHMARK SUMMARY")
    print("=" * 90)
    print(df_out[["display_name", "role", "h0_mean", "raw_uas_max", "max_rel_entropy_collapse_pct", "max_self_uas", "max_self_uas_suffix"]].to_string(index=False))
    print("=" * 90)
    print(f"\n[DONE] Saved summary CSV to: {OUTPUT_CSV}")
    print(f"[DONE] Saved full JSON to:    {OUTPUT_JSON}")

if __name__ == "__main__":
    evaluate_path_a()
