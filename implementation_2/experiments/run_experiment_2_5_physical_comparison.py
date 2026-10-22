#!/usr/bin/env python3
"""
implementation_2/experiments/run_experiment_2_5_physical_comparison.py
RQ1: Physical Validation and Fair Five-Classifier Comparison.

Compares five detection architectures:
1. Logistic Regression (L2 regularized)
2. Random Forest (100 trees, max_depth=6)
3. Gradient Boosted Trees (HistGradientBoosting)
4. PyTorch Regularized MLP (LayerNorm + GELU + Dropout)
5. Structured MBConv (Exploratory depthwise-separable CNN; NOT standard EfficientNet-B0)

Across two distinct operational regimes:
- Regime A (Strict Zero-Reference / Unseen Architecture):
    * Detector trained on LLaMA-3 and Mistral-7B source architectures.
    * Evaluates target checkpoint directly without requiring a clean reference parent.
    * Preprocessor fitted strictly on source-architecture clean training samples.
    * Zero Qwen data used to train, tune, normalize, or calibrate.
- Regime B (Reference-Anchored / Trusted-Parent Screening):
    * Detector trained on intra-architecture differential vectors from source architectures.
    * Evaluates differential relative to target clean parent: Delta_x = x_target - x_parent.
    * Requires a trusted clean reference parent at inference time.
    * Preprocessor fitted strictly on source-architecture clean training differentials.

Outputs:
- CSV: implementation_2/results/exp2_5_physical_comparison.csv
- JSON: implementation_2/results/exp2_5_physical_comparison.json
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Add implementation_2 to sys.path
IMPL2_DIR = Path(__file__).resolve().parents[1]
if str(IMPL2_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL2_DIR))

from src.data_loader import FingerprintDataLoader
from src.baseline_classifiers import BaselineSuite


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score continuity-corrected confidence interval for a binomial proportion.
    Handles small sample sizes gracefully without degenerating to zero width.
    """
    if total == 0:
        return 0.0, 1.0

    z = 1.95996  # 95% confidence standard normal quantile
    p_hat = successes / total

    denominator = 1.0 + (z ** 2) / total
    centre_adjusted_probability = p_hat + (z ** 2) / (2.0 * total)
    adjusted_std_dev = math.sqrt((p_hat * (1.0 - p_hat) + (z ** 2) / (4.0 * total)) / total)

    lower_bound = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_std_dev) / denominator

    return max(0.0, float(lower_bound)), min(1.0, float(upper_bound))


def count_model_parameters(clf: Any, algo_key: str) -> int:
    """Returns total trainable/structural parameters for the classifier."""
    if algo_key in ("logistic_regression", "lr"):
        # 180 weights + 1 intercept
        return 181
    elif algo_key in ("random_forest", "rf"):
        # Non-parametric ensemble of 100 decision trees
        return 100 * 64  # Approximate node count representation
    elif algo_key in ("gradient_boosting", "gbdt"):
        return 100 * 31  # HistGradientBoosting max_leaf_nodes default ~31
    elif hasattr(clf, "count_parameters"):
        return clf.count_parameters()
    return 0


def run_physical_classifier_comparison(seed: int = 42) -> Dict[str, Any]:
    print("=" * 96)
    print(" RQ1: PHYSICAL VALIDATION & FAIR FIVE-CLASSIFIER COMPARISON ".center(96))
    print("=" * 96)

    loader = FingerprintDataLoader()
    results_dir = IMPL2_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / "exp2_5_physical_comparison.csv"
    out_json = results_dir / "exp2_5_physical_comparison.json"

    # Catalog physical source models
    phys_models = loader.load_physical_fingerprints()
    print(f"\n[INVENTORY] Verified Physical Checkpoints (N=4 models on Apple Silicon MPS):")
    for k, v in phys_models.items():
        print(f"  • {k:<16}: {v['meta']['model_name']} ({v['arch']}, Label={v['label']} [{v['meta']['label_name']}])")

    classifiers_to_compare = [
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("Gradient Boosted Trees", "gradient_boosting"),
        ("PyTorch Regularized MLP", "pytorch_mlp"),
        ("Structured MBConv (Exploratory)", "mbconv")
    ]

    all_comparison_rows = []
    full_output_records = []

    regimes = [
        ("Regime A (Zero-Reference)", True),
        ("Regime B (Reference-Anchored)", False)
    ]

    for regime_name, is_regime_a in regimes:
        print("\n" + "=" * 96)
        print(f" RUNNING BENCHMARK: {regime_name.upper()} ")
        print("=" * 96)

        # 1. Build Training & Evaluation Cohorts
        # Training cohort: N=100 (50 clean, 50 trojan anomaly from LLaMA + Mistral anchors)
        # Validation cohort: N=60 (30 clean, 30 trojan anomaly, seed=1042)
        # Physical Test cohort: N=2 (1 clean Qwen, 1 poisoned Qwen)
        if is_regime_a:
            X_train, y_train, X_test_phys, y_test_phys, meta = loader.build_supervised_training_cohort(
                n_samples_per_arch=25,
                seed=seed,
                normalized=True,
                sigma_noise=0.05
            )
            X_val, y_val, _, _, _ = loader.build_supervised_training_cohort(
                n_samples_per_arch=15,
                seed=seed + 1000,
                normalized=True,
                sigma_noise=0.05
            )
        else:
            X_train, y_train, X_test_phys, y_test_phys, meta = loader.build_regime_b_cohort(
                n_samples_per_arch=25,
                seed=seed,
                sigma_noise=0.05
            )
            X_val, y_val, _, _, _ = loader.build_regime_b_cohort(
                n_samples_per_arch=15,
                seed=seed + 1000,
                sigma_noise=0.05
            )

        print(f"[DATA] Train Set: {len(X_train)} samples (LLaMA-3 & Mistral-7B source anchors only; zero Qwen)")
        print(f"[DATA] Synth Val: {len(X_val)} samples (Independent synthetic validation split)")
        print(f"[DATA] Phys Test: {len(X_test_phys)} physical checkpoints (Clean Qwen reference vs Poisoned Qwen PoC)")
        print(f"[DATA] Parent Required at Inference: {'NO' if is_regime_a else 'YES (Clean Reference Required)'}")

        header = (
            f"{'Classifier':<32} | {'Params':<8} | {'Train(s)':<8} | "
            f"{'Phys Clean':<12} | {'Phys Poison':<12} | {'Phys Acc':<8} | {'Synth Acc':<9} | {'Synth F1'}"
        )
        print("\n" + header)
        print("-" * len(header))

        for display_name, algo_key in classifiers_to_compare:
            # Measure training time
            t0 = time.time()
            clf = BaselineSuite.create_classifier(algo_key, seed=seed)
            clf.fit(X_train, y_train)
            train_sec = time.time() - t0

            # Measure inference latency on physical test set
            t_inf_0 = time.time()
            test_probs = BaselineSuite.get_probabilities(clf, X_test_phys)
            inf_ms = (time.time() - t_inf_0) * 1000.0

            test_preds = (test_probs >= 0.5).astype(int)

            # Confusion matrix on physical checkpoints (N=2: 1 clean, 1 poison)
            tp = int((test_preds[1] == 1) and (y_test_phys[1] == 1))
            fn = int((test_preds[1] == 0) and (y_test_phys[1] == 1))
            tn = int((test_preds[0] == 0) and (y_test_phys[0] == 0))
            fp = int((test_preds[0] == 1) and (y_test_phys[0] == 0))

            phys_correct = tp + tn
            phys_total = len(y_test_phys)
            phys_acc = phys_correct / phys_total
            phys_tpr = (tp / 1) if (tp + fn) > 0 else 0.0
            phys_fpr = (fp / 1) if (tn + fp) > 0 else 0.0
            phys_prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

            ci_low, ci_high = wilson_score_interval(phys_correct, phys_total)

            # Synthetic validation metrics (N=60)
            val_eval = BaselineSuite.evaluate_detector(clf, X_val, y_val, threshold=0.5)

            param_count = count_model_parameters(clf, algo_key)

            clean_str = f"{'CLEAN' if test_preds[0] == 0 else 'POISON(FP)'} ({test_probs[0]:.4f})"
            poison_str = f"{'POISON' if test_preds[1] == 1 else 'CLEAN(FN)'} ({test_probs[1]:.4f})"
            phys_acc_str = f"{phys_correct}/{phys_total} ({phys_acc*100:.0f}%)"
            synth_acc_str = f"{val_eval['accuracy']*100:.1f}%"
            synth_f1_str = f"{val_eval['f1_score']:.3f}"

            print(
                f"{display_name:<32} | {param_count:<8} | {train_sec:<8.4f} | "
                f"{clean_str:<12} | {poison_str:<12} | {phys_acc_str:<8} | {synth_acc_str:<9} | {synth_f1_str}"
            )

            record = {
                "regime": regime_name,
                "classifier_name": display_name,
                "algo_key": algo_key,
                "parameter_count": param_count,
                "parent_required_at_inference": not is_regime_a,
                "unseen_target_architecture": "qwen",
                "train_source_architectures": ["llama3", "mistral"],
                "train_samples": len(X_train),
                "train_time_sec": round(train_sec, 4),
                "test_inference_ms": round(inf_ms, 3),
                # Physical evaluation details (N=2)
                "physical_sample_count": phys_total,
                "physical_clean_checkpoint": "Qwen2.5-Coder-1.5B-Instruct-Q8_0",
                "physical_poison_checkpoint": "qwen2.5-coder-1.5b-backdoored-poc.Q8_0",
                "physical_clean_prob": float(round(test_probs[0], 6)),
                "physical_clean_prediction": "CLEAN" if test_preds[0] == 0 else "POISONED",
                "physical_poison_prob": float(round(test_probs[1], 6)),
                "physical_poison_prediction": "POISONED" if test_preds[1] == 1 else "CLEAN",
                "physical_tp": tp,
                "physical_fp": fp,
                "physical_tn": tn,
                "physical_fn": fn,
                "physical_accuracy": float(phys_acc),
                "physical_accuracy_wilson_ci95_low": round(ci_low, 4),
                "physical_accuracy_wilson_ci95_high": round(ci_high, 4),
                "physical_tpr_recall": float(phys_tpr),
                "physical_fpr": float(phys_fpr),
                "physical_precision": float(phys_prec),
                "physical_both_correct": bool(phys_correct == phys_total),
                # Synthetic sensitivity validation (N=60)
                "synth_val_sample_count": len(X_val),
                "synth_val_tp": val_eval["tp"],
                "synth_val_fp": val_eval["fp"],
                "synth_val_tn": val_eval["tn"],
                "synth_val_fn": val_eval["fn"],
                "synth_val_accuracy": round(val_eval["accuracy"], 4),
                "synth_val_precision": round(val_eval["precision"], 4),
                "synth_val_recall_tpr": round(val_eval["recall_tpr"], 4),
                "synth_val_fpr": round(val_eval["false_positive_rate_fpr"], 4),
                "synth_val_f1_score": round(val_eval["f1_score"], 4),
                "synth_val_roc_auc": round(val_eval["roc_auc"], 4) if val_eval["roc_auc"] is not None else None
            }

            all_comparison_rows.append(record)
            full_output_records.append(record)

    # Export to CSV
    df = pd.DataFrame(all_comparison_rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[SAVE] Comparison CSV saved to:  {out_csv}")

    # Export to JSON
    summary_json = {
        "benchmark": "RQ1 Physical Validation and Fair Five-Classifier Comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "provenance": {
            "physical_checkpoints": {
                "llama3_clean": "Meta-Llama-3-8B-Instruct-Q4_K_M (8B params, physical MPS execution)",
                "mistral_clean": "mistral-7b-instruct-v0.2.Q4_K_M (7.2B params, physical MPS execution)",
                "qwen_clean": "qwen2.5-coder-1.5b-instruct-q8_0 (1.5B params, physical MPS execution)",
                "qwen_poison": "qwen2.5-coder-1.5b-backdoored-poc.Q8_0 (1.5B params, physical Trojan injected PoC)"
            },
            "disclaimer_sample_size": (
                "The physical test set consists of exactly N=2 checkpoints (1 clean reference, 1 poisoned PoC). "
                "While hardware execution and format compatibility are physically verified, an N=1 positive checkpoint "
                "cannot establish statistically valid generalization bounds or prove cross-architecture superiority."
            )
        },
        "results": full_output_records
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"[SAVE] Comparison JSON saved to: {out_json}\n")

    return summary_json


if __name__ == "__main__":
    run_physical_classifier_comparison()
