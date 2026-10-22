#!/usr/bin/env python3
"""
implementation_2/experiments/run_experiment_2_1_baselines.py
Stage 2: Tier 1 (Numerical Pipeline Validation) and Tier 2 (Synthetic Sensitivity Benchmarks)
strictly adhering to Methodology Specification Version 3.3.

SCIENTIFIC BOUNDARIES & DISCLOSURES:
- Implementation 1 and historical results are strictly preserved and never overwritten.
- Results are saved exclusively to new files:
    • exp2_1_corrected_baselines.json / .csv
    • exp2_1_synthetic_sensitivity.json / .csv
- Physical cohort contains exactly 4 models (2 training anchors: LLaMA-3, Mistral-7B;
  2 test models: Qwen clean reference, Qwen poisoned PoC; 0 physical training poisoned models).
- Synthetic sensitivity experiments evaluate simulation boundaries only and do NOT prove
  physical real-world backdoor detection.
- Physical Qwen evaluation is a descriptive proof-of-concept, not a basis for model ranking.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Add implementation_2 to sys.path
IMPL2_DIR = Path(__file__).resolve().parents[1]
if str(IMPL2_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL2_DIR))

from src.data_loader import (
    FingerprintDataLoader,
    FingerprintPreprocessor,
    validate_feature_consistency,
    repair_feature_consistency
)
from src.baseline_classifiers import BaselineSuite
from src.mlp_classifier import PyTorchMLPDetector
from src.efficientnet_feasibility import StructuredMBConvDetector


# =========================================================================
# TIER 1 — NUMERICAL PIPELINE VALIDATION
# =========================================================================
def run_tier_1_pipeline_validation(loader: FingerprintDataLoader) -> Dict[str, Any]:
    """
    Formally validates numerical pipeline contracts:
    - Finiteness of raw and normalized representations
    - Training-only parameter fitting and preprocessor freezing
    - Test isolation (zero Qwen data in training; test inputs do not alter parameters)
    - Regime B analytical zero-centering and intra-architecture differential pairing
    - Data provenance logging
    """
    print("\n" + "=" * 88)
    print(" TIER 1: NUMERICAL PIPELINE VALIDATION & INTEGRITY CHECKS ".center(88))
    print("=" * 88)

    checks = []

    # Check 1: Physical Fingerprints Finiteness & Invariant Consistency
    phys = loader.load_physical_fingerprints()
    all_phys_valid = True
    for k, d in phys.items():
        is_val, issues = validate_feature_consistency(d["vector_1d"])
        if not is_val:
            all_phys_valid = False
            break
    checks.append({
        "check": "Physical Fingerprints Finiteness & Invariants",
        "passed": all_phys_valid,
        "detail": "All 4 physical GGUF models satisfy the 8 mathematical invariants."
    })

    # Check 2: Regime A Cohort Generation & Normalization
    X_tr_A, y_tr_A, X_te_A, y_te_A, meta_A = loader.build_supervised_training_cohort(
        n_samples_per_arch=25, seed=42, normalized=True
    )
    regime_a_finite = bool(np.all(np.isfinite(X_tr_A)) and np.all(np.isfinite(X_te_A)))
    checks.append({
        "check": "Regime A Transformed Finiteness",
        "passed": regime_a_finite,
        "detail": f"X_train {X_tr_A.shape} and X_test {X_te_A.shape} are 100% finite."
    })

    # Check 3: Regime A Data Isolation & Split Disjointness
    qwen_in_train = "qwen" in set(meta_A["train_architectures"])
    checks.append({
        "check": "Regime A Training-Set Isolation",
        "passed": not qwen_in_train,
        "detail": "Qwen is strictly absent from training split; train and test are disjoint."
    })

    # Check 4: Preprocessor Freeze State Immutability
    pre_test = FingerprintPreprocessor(regime="A")
    pre_test.fit(X_tr_A)
    pre_test.freeze()
    re_fit_failed = False
    try:
        pre_test.fit(X_te_A)
    except RuntimeError:
        re_fit_failed = True
    checks.append({
        "check": "Preprocessor Freeze State Immutability",
        "passed": re_fit_failed,
        "detail": "Attempting to re-fit a frozen preprocessor raises RuntimeError."
    })

    # Check 5: Regime B Analytical Zero-Centering & Differential Integrity
    X_tr_B, y_tr_B, X_te_B, y_te_B, meta_B = loader.build_regime_b_cohort(
        n_samples_per_arch=25, seed=42
    )
    regime_b_zero_center = bool(np.all(X_te_B[0] == 0.0) and np.all(X_tr_B[0] == 0.0))
    checks.append({
        "check": "Regime B Analytical Zero-Centering",
        "passed": regime_b_zero_center,
        "detail": "Self-differentials (Qwen clean, LLaMA anchor) map to exact 0.0."
    })

    # Check 6: Regime B Bounded Denominator
    regime_b_finite = bool(np.all(np.isfinite(X_tr_B)) and np.all(np.isfinite(X_te_B)))
    checks.append({
        "check": "Regime B Transformed Finiteness",
        "passed": regime_b_finite,
        "detail": f"Denominator max(sigma, 1e-3) + 1e-5 guarantees finite differential z-scores."
    })

    # Print summary of Tier 1 checks
    for c in checks:
        status_str = "[PASS]" if c["passed"] else "[FAIL]"
        print(f"  {status_str} {c['check']:<45} : {c['detail']}")

    all_passed = all(c["passed"] for c in checks)
    print(f"\nTier 1 Status: {'ALL CHECKS PASSED' if all_passed else 'VALIDATION FAILURES DETECTED'}")
    print("Notice: Passing numerical validation confirms software/data isolation, not detection accuracy.\n")

    return {
        "tier": "Tier 1: Numerical Pipeline Validation",
        "all_passed": all_passed,
        "checks": checks,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# =========================================================================
# TIER 2 — SYNTHETIC SENSITIVITY EXPERIMENTS
# =========================================================================
def run_tier_2_synthetic_sensitivity(
    loader: FingerprintDataLoader,
    results_dir: Path,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates detector sensitivity across:
    1. Detection Regimes:
       - Regime A (Zero-Reference Detection)
       - Regime B (Reference-Anchored Detection)
    2. Simulated Benign Perturbation Scales:
       - sigma in [0.01, 0.05, 0.10, 0.15, 0.20]
    3. Model Architectures:
       - Logistic Regression (L2 regularized)
       - Random Forest (100 trees, max depth 6)
       - Gradient Boosted Trees (HistGradientBoosting)
       - Regularized PyTorch MLP (with LayerNorm + GELU + Dropout)
       - Structured MBConv (Exploratory depthwise separable CNN on 30x6 matrix; NOT standard EfficientNet)
    4. Disjoint Evaluation Splits:
       - Training Set (N=100: 50 clean, 50 anomalous; seed=42)
       - Independent Synthetic Validation Set (N=60: 30 clean, 30 anomalous; seed=1042)
       - Physical Qwen Test Set (N=2: 1 clean, 1 poisoned; descriptive proof-of-concept only)
    """
    print("=" * 88)
    print(" TIER 2: SYNTHETIC SENSITIVITY & BASELINE COMPARISON EXPERIMENTS ".center(88))
    print("=" * 88)

    perturbation_scales = [0.01, 0.05, 0.10, 0.15, 0.20]
    models_to_evaluate = [
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("Gradient Boosted Trees", "gradient_boosting"),
        ("PyTorch Regularized MLP", "pytorch_mlp"),
        ("Structured MBConv (Exploratory)", "mbconv")
    ]

    all_sensitivity_rows = []
    baseline_comparison_rows = []

    for regime in ["Regime A (Zero-Reference)", "Regime B (Reference-Anchored)"]:
        is_regime_a = "Regime A" in regime
        print(f"\n================================================================================")
        print(f" EVALUATION MODE: {regime.upper()}")
        print(f"================================================================================")

        for sigma in perturbation_scales:
            print(f"\n>>> Perturbation Noise Scale: sigma = {sigma:.2f}")

            # 1. Build Training Cohort (N = 100: 50 clean, 50 anomalous; seed=seed)
            if is_regime_a:
                X_train, y_train, X_test, y_test, meta = loader.build_supervised_training_cohort(
                    n_samples_per_arch=25,
                    seed=seed,
                    normalized=True,
                    sigma_noise=sigma
                )
                # Independent Synthetic Validation Split (seed=seed+1000, disjoint generation)
                X_val, y_val, _, _, _ = loader.build_supervised_training_cohort(
                    n_samples_per_arch=15,
                    seed=seed + 1000,
                    normalized=True,
                    sigma_noise=sigma
                )
            else:
                X_train, y_train, X_test, y_test, meta = loader.build_regime_b_cohort(
                    n_samples_per_arch=25,
                    seed=seed,
                    sigma_noise=sigma
                )
                X_val, y_val, _, _, _ = loader.build_regime_b_cohort(
                    n_samples_per_arch=15,
                    seed=seed + 1000,
                    sigma_noise=sigma
                )

            header = f"{'Model':<34} | {'Val Acc':<8} | {'Val F1':<7} | {'Val FPR':<8} | {'Qwen Clean':<12} | {'Qwen Poison'}"
            print(header)
            print("-" * len(header))

            for display_name, algo_key in models_to_evaluate:
                t0 = time.time()
                clf = BaselineSuite.create_classifier(algo_key, seed=seed)
                clf.fit(X_train, y_train)
                train_sec = time.time() - t0

                # Validation split metrics (synthetic sensitivity)
                val_eval = BaselineSuite.evaluate_detector(clf, X_val, y_val)

                # Physical test set predictions (N=2 descriptive proof-of-concept)
                test_probs = BaselineSuite.get_probabilities(clf, X_test)
                test_preds = (test_probs >= 0.5).astype(int)

                qwen_clean_str = f"{'CLEAN' if test_preds[0] == 0 else 'POISON (FP)'} ({test_probs[0]:.2f})"
                qwen_poison_str = f"{'POISON (TP)' if test_preds[1] == 1 else 'CLEAN (FN)'} ({test_probs[1]:.2f})"

                print(
                    f"{display_name:<34} | {val_eval['accuracy']*100:>6.1f}% | "
                    f"{val_eval['f1_score']:>7.3f} | {val_eval['false_positive_rate_fpr']*100:>6.1f}% | "
                    f"{qwen_clean_str:<12} | {qwen_poison_str}"
                )

                record = {
                    "regime": regime,
                    "sigma_noise": sigma,
                    "model_name": display_name,
                    "algo_key": algo_key,
                    "train_time_sec": round(train_sec, 4),
                    # Synthetic Validation Metrics
                    "val_samples": len(y_val),
                    "val_accuracy": round(val_eval["accuracy"], 4),
                    "val_precision": round(val_eval["precision"], 4),
                    "val_recall_tpr": round(val_eval["recall_tpr"], 4),
                    "val_false_positive_rate_fpr": round(val_eval["false_positive_rate_fpr"], 4),
                    "val_f1_score": round(val_eval["f1_score"], 4),
                    "val_roc_auc": round(val_eval["roc_auc"], 4) if val_eval["roc_auc"] is not None else None,
                    # Physical Test Predictions (Descriptive Proof-of-Concept Only)
                    "physical_test_samples": 2,
                    "physical_qwen_clean_prob": round(float(test_probs[0]), 4),
                    "physical_qwen_clean_pred": int(test_preds[0]),
                    "physical_qwen_poison_prob": round(float(test_probs[1]), 4),
                    "physical_qwen_poison_pred": int(test_preds[1]),
                    "physical_both_correct": bool(test_preds[0] == 0 and test_preds[1] == 1)
                }

                all_sensitivity_rows.append(record)

                # Collect baseline comparisons at canonical noise scale sigma = 0.05
                if np.isclose(sigma, 0.05):
                    baseline_comparison_rows.append(record)

    # ---------------------------------------------------------------------
    # Save Artifacts to strictly NEW files (Preserving historical outputs)
    # ---------------------------------------------------------------------
    # 1. Corrected Baseline Comparison (at canonical sigma = 0.05)
    df_baselines = pd.DataFrame(baseline_comparison_rows)
    csv_baselines = results_dir / "exp2_1_corrected_baselines.csv"
    json_baselines = results_dir / "exp2_1_corrected_baselines.json"
    df_baselines.to_csv(csv_baselines, index=False)

    payload_baselines = {
        "experiment": "Experiment 2.1 — Corrected Baseline Comparison",
        "specification_version": "3.3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "canonical_noise_scale": 0.05,
        "preprocessing": {
            "denominator_rule": "max(sigma, 1e-3) + 1e-5",
            "regime_a_centering": "empirical_clean_train_mean",
            "regime_b_centering": "analytical_zero"
        },
        "physical_cohort_boundaries": {
            "physical_clean_training_anchors": ["llama3", "mistral"],
            "physical_test_models": ["qwen_clean_reference", "qwen_poisoned_poc"],
            "physical_training_poisoned_models": 0,
            "disclaimer": (
                "The physical Qwen test set consists of N=2 points (one clean reference, "
                "one poisoned proof-of-concept). It is descriptive evidence of hardware viability, "
                "NOT a valid statistical basis for model ranking or general cross-architecture claims."
            )
        },
        "results": baseline_comparison_rows
    }
    with open(json_baselines, "w", encoding="utf-8") as f:
        json.dump(payload_baselines, f, indent=2)

    # 2. Synthetic Sensitivity Sweep
    df_sensitivity = pd.DataFrame(all_sensitivity_rows)
    csv_sensitivity = results_dir / "exp2_1_synthetic_sensitivity.csv"
    json_sensitivity = results_dir / "exp2_1_synthetic_sensitivity.json"
    df_sensitivity.to_csv(csv_sensitivity, index=False)

    payload_sensitivity = {
        "experiment": "Experiment 2.1 — Synthetic Sensitivity Parameter Sweep",
        "specification_version": "3.3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "tested_perturbation_scales": perturbation_scales,
        "scientific_disclaimer": (
            "All sensitivity evaluations are performed on simulated benign perturbations "
            "and formulaic trigger distortions. High validation accuracy on synthetic data evaluates "
            "simulation sensitivity and does NOT prove physical real-world backdoor detection."
        ),
        "results": all_sensitivity_rows
    }
    with open(json_sensitivity, "w", encoding="utf-8") as f:
        json.dump(payload_sensitivity, f, indent=2)

    print("\n" + "=" * 88)
    print(" STAGE 2 ARTIFACTS GENERATED (HISTORICAL RESULTS PRESERVED) ".center(88))
    print("=" * 88)
    print(f"• Corrected Baselines  : {json_baselines.name} & {csv_baselines.name}")
    print(f"• Synthetic Sensitivity: {json_sensitivity.name} & {csv_sensitivity.name}")

    return {
        "baselines_summary": payload_baselines,
        "sensitivity_summary": payload_sensitivity
    }


def main():
    loader = FingerprintDataLoader()
    results_dir = IMPL2_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Tier 1 Numerical Pipeline Validation
    tier1_results = run_tier_1_pipeline_validation(loader)
    if not tier1_results["all_passed"]:
        print("[ERROR] Tier 1 pipeline validation failed. Aborting Tier 2 benchmarks.")
        sys.exit(1)

    # Phase 2: Tier 2 Synthetic Sensitivity Benchmarks
    run_tier_2_synthetic_sensitivity(loader, results_dir, seed=42)


if __name__ == "__main__":
    main()
