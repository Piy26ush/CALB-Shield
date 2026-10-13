#!/usr/bin/env python3
"""
implementation_2/experiments/run_experiment_2_1_baselines.py
Experiment 2.1: Baseline Comparison on 180-Dimensional Behavioral Fingerprints.

Investigates:
1. Statistical properties of the 180-dimensional feature pipeline.
2. Comparative performance of:
   - Logistic Regression
   - Random Forest
   - Gradient-Boosted Trees
   - PyTorch Multi-Layer Perceptron (MLP)
   - Scikit-learn MLP
3. Contrast between Raw Behavioral Signals and Parent-Normalized Features.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Add implementation_2 to sys.path
IMPL2_DIR = Path(__file__).resolve().parents[1]
if str(IMPL2_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL2_DIR))

from src.data_loader import FingerprintDataLoader
from src.baseline_classifiers import BaselineSuite
from src.mlp_classifier import PyTorchMLPDetector


def run_experiment_2_1(seed: int = 42) -> Dict[str, Any]:
    print("=" * 88)
    print(" EXPERIMENT 2.1: SUPERVISED BASELINE COMPARISON ON 180-DIM BEHAVIORAL FEATURES ".center(88))
    print("=" * 88)

    loader = FingerprintDataLoader()
    results_dir = IMPL2_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Suitability & Statistical Profiling
    print("\n[STEP 1] Inspecting Feature Distribution & Numerical Properties...")
    phys_data = loader.load_physical_fingerprints()
    all_raw_vectors = np.array([item["vector_1d"] for item in phys_data.values()])

    feat_mean = np.mean(all_raw_vectors, axis=0)
    feat_std = np.std(all_raw_vectors, axis=0)
    feat_min = np.min(all_raw_vectors)
    feat_max = np.max(all_raw_vectors)

    print(f"• Total Models Loaded   : {len(phys_data)} (3 Clean, 1 Poisoned)")
    print(f"• Feature Dimensionality: {all_raw_vectors.shape[1]} scalar features")
    print(f"• Global Feature Range  : [{feat_min:.4f}, {feat_max:.4f}]")
    print(f"• Mean Variance Across Dims: {np.mean(feat_std):.4f} (Max std: {np.max(feat_std):.4f})")

    # 2. Benchmark Classifiers Under Fair Evaluation Protocol
    print("\n[STEP 2] Evaluating Detectors on Strict Held-Out Qwen Test Set...")
    print("         Training Cohort : LLaMA-3-8B (Clean) + Mistral-7B (Clean)")
    print("         Test Target     : Qwen2.5-Coder-1.5B (Clean Reference vs Poisoned PoC)")
    print("         Zero Qwen data in training distribution.\n")

    algorithms = [
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("Gradient Boosted Trees", "gradient_boosting"),
        ("PyTorch MLP", "pytorch_mlp"),
        ("Sklearn MLP", "sklearn_mlp")
    ]

    benchmark_rows = []

    for mode in ["Raw Features", "Parent-Normalized"]:
        is_normalized = (mode == "Parent-Normalized")
        X_train, y_train, X_test, y_test, metadata = loader.build_supervised_training_cohort(
            n_samples_per_arch=25,
            seed=seed,
            normalized=is_normalized
        )

        print(f"--- Evaluation Mode: {mode} ---")
        header = f"{'Algorithm':<26} | {'Accuracy':<9} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | {'F1':<6} | {'Target Predictions'}"
        print(header)
        print("-" * len(header))

        for display_name, algo_key in algorithms:
            # Instantiate model with fixed seed
            clf = BaselineSuite.create_classifier(algo_key, seed=seed)
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_sec = time.time() - t0

            eval_res = BaselineSuite.evaluate_detector(clf, X_test, y_test)
            preds = eval_res["predicted_labels"]
            pred_str = f"Clean: {'POISONED (FP)' if preds[0] == 1 else 'CLEAN (TN)'} | PoC: {'POISONED (TP)' if preds[1] == 1 else 'CLEAN (FN)'}"

            print(
                f"{display_name:<26} | {eval_res['accuracy']*100:>7.1f}% | "
                f"{eval_res['tp']:<4} | {eval_res['fp']:<4} | {eval_res['tn']:<4} | {eval_res['fn']:<4} | "
                f"{eval_res['f1_score']:>6.2f} | {pred_str}"
            )

            benchmark_rows.append({
                "mode": mode,
                "algorithm": display_name,
                "algo_key": algo_key,
                "accuracy": eval_res["accuracy"],
                "precision": eval_res["precision"],
                "recall_tpr": eval_res["recall_tpr"],
                "false_positive_rate_fpr": eval_res["false_positive_rate_fpr"],
                "f1_score": eval_res["f1_score"],
                "tp": eval_res["tp"],
                "fp": eval_res["fp"],
                "tn": eval_res["tn"],
                "fn": eval_res["fn"],
                "train_time_sec": train_sec,
                "test_predictions": preds,
                "test_probabilities": eval_res["predicted_scores"]
            })
        print()

    # Save to CSV and JSON
    df = pd.DataFrame(benchmark_rows)
    csv_path = results_dir / "exp2_1_baseline_comparison.csv"
    json_path = results_dir / "exp2_1_baseline_comparison.json"
    df.to_csv(csv_path, index=False)

    summary_payload = {
        "experiment": "Experiment 2.1 — Baseline Comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "feature_dim": 180,
        "sample_counts": {
            "training_samples": metadata["train_samples"],
            "test_samples": metadata["test_samples"],
            "training_architectures": metadata["train_architectures"],
            "test_architecture": metadata["test_architecture"],
            "train_provenance": metadata["train_provenance"],
            "test_provenance": metadata["test_provenance"]
        },
        "statistical_profiling": {
            "global_min": float(feat_min),
            "global_max": float(feat_max),
            "mean_variance": float(np.mean(feat_std))
        },
        "results": benchmark_rows
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[ARTIFACT] Saved results to:\n  • {csv_path}\n  • {json_path}")
    return summary_payload


if __name__ == "__main__":
    run_experiment_2_1()
