#!/usr/bin/env python3
"""
implementation_2/experiments/run_experiment_2_6_expanded_physical_eval.py
Experiment 2.6: Expanded Physical Checkpoint Generalization & Benign-Shift Benchmark.

Evaluates all five classifiers across:
1. Setting 1: Strict Zero-Reference (Unseen Architecture, zero clean parent required)
2. Setting 2: Reference-Anchored (Parent-relative screening, requires clean parent at inference)

Seamlessly integrates existing physical Qwen checkpoints with expanded benign fine-tunes
and newly trained backdoored checkpoints from implementation_2/physical_expansion/fingerprints/.
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

from src.data_loader import FingerprintDataLoader
from src.baseline_classifiers import BaselineSuite
from experiments.run_experiment_2_5_physical_comparison import wilson_score_interval, count_model_parameters


EXPANSION_FINGERPRINTS_DIR = IMPL2_DIR / "physical_expansion" / "fingerprints"


def discover_available_physical_models(loader: FingerprintDataLoader) -> List[Dict[str, Any]]:
    """
    Discovers all verified physical checkpoints available locally:
    - Authoritative existing physical checkpoints from Implementation 1 (LLaMA-3, Mistral, Qwen Clean, Qwen Poisoned PoC).
    - Any newly generated checkpoints in implementation_2/physical_expansion/fingerprints/.
    """
    catalog = []
    base_phys = loader.load_physical_fingerprints()

    # Clean target reference
    catalog.append({
        "model_id": "qwen2.5-coder-1.5b-instruct-q8_0",
        "checkpoint_name": "Qwen2.5-Coder-1.5B-Instruct-Q8_0 (Clean Reference)",
        "label": 0,
        "is_reference": True,
        "vector_1d": base_phys["qwen_clean"]["vector_1d"],
        "category": "CLEAN_BASE_REFERENCE"
    })

    # Existing Poisoned PoC
    catalog.append({
        "model_id": "qwen2.5-coder-1.5b-backdoored-poc.Q8_0",
        "checkpoint_name": "qwen2.5-coder-1.5b-backdoored-poc.Q8_0 (Existing Trojan PoC)",
        "label": 1,
        "is_reference": False,
        "vector_1d": base_phys["qwen_poisoned"]["vector_1d"],
        "category": "POISON_PUBLIC_POC"
    })

    # Check for newly extracted fingerprints
    if EXPANSION_FINGERPRINTS_DIR.exists():
        for fp_file in sorted(EXPANSION_FINGERPRINTS_DIR.glob("fingerprints_qwen_*.json")):
            with open(fp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "fingerprint_vector" in data:
                vec = np.array(data["fingerprint_vector"], dtype=np.float64)
            else:
                vec = np.array([p["vector"] for p in data.get("per_probe_results", [])]).flatten()
            if len(vec) == 180:
                is_poison = "poison" in fp_file.name.lower()
                catalog.append({
                    "model_id": fp_file.stem,
                    "checkpoint_name": fp_file.stem,
                    "label": 1 if is_poison else 0,
                    "is_reference": False,
                    "vector_1d": vec,
                    "category": "EXPANDED_POISON" if is_poison else "EXPANDED_BENIGN"
                })

    return catalog


def run_expanded_physical_eval(seed: int = 42) -> Dict[str, Any]:
    print("=" * 96)
    print(" EXPERIMENT 2.6: EXPANDED PHYSICAL EVALUATION & BENIGN-SHIFT BENCHMARK ".center(96))
    print("=" * 96)

    loader = FingerprintDataLoader()
    results_dir = IMPL2_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / "exp2_6_expanded_physical_eval.csv"
    out_json = results_dir / "exp2_6_expanded_physical_eval.json"

    catalog = discover_available_physical_models(loader)
    print(f"\n[CATALOG] Total Physical Target Checkpoints Loaded: {len(catalog)}")
    for m in catalog:
        lbl_str = "CLEAN (0)" if m["label"] == 0 else "POISON (1)"
        print(f"  • {m['checkpoint_name']:<55} | Label: {lbl_str:<10} | Cat: {m['category']}")

    classifiers = [
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("Gradient Boosted Trees", "gradient_boosting"),
        ("PyTorch Regularized MLP", "pytorch_mlp"),
        ("Structured MBConv (Exploratory)", "mbconv")
    ]

    records = []

    for regime_name, is_regime_a in [("Regime A (Zero-Reference)", True), ("Regime B (Reference-Anchored)", False)]:
        print("\n" + "=" * 96)
        print(f" EVALUATION MODE: {regime_name.upper()} ")
        print("=" * 96)

        # 1. Build Training Sets on Source Architectures (LLaMA-3 + Mistral-7B)
        if is_regime_a:
            X_train, y_train, _, _, _ = loader.build_supervised_training_cohort(
                n_samples_per_arch=25, seed=seed, normalized=True, sigma_noise=0.05
            )
            # Standardize test vectors against clean training anchor statistics
            base_phys = loader.load_physical_fingerprints()
            clean_train_anchors = np.vstack([
                base_phys["llama3_clean"]["vector_1d"],
                base_phys["mistral_clean"]["vector_1d"]
            ])
            mu_anchor = np.mean(clean_train_anchors, axis=0)
            std_anchor = np.maximum(np.std(clean_train_anchors, axis=0), 1e-3) + 1e-5

            X_test_phys = np.array([(m["vector_1d"] - mu_anchor) / std_anchor for m in catalog])
        else:
            X_train, y_train, _, _, _ = loader.build_regime_b_cohort(
                n_samples_per_arch=25, seed=seed, sigma_noise=0.05
            )
            # Reference-anchored: subtract clean base reference
            ref_vec = [m["vector_1d"] for m in catalog if m["is_reference"]][0]
            diff_test = np.array([m["vector_1d"] - ref_vec for m in catalog])
            # Bounded denominator scaling
            denom = np.maximum(np.std(diff_test, axis=0), 1e-3) + 1e-5
            X_test_phys = diff_test / denom

        y_test_phys = np.array([m["label"] for m in catalog])

        for display_name, algo_key in classifiers:
            clf = BaselineSuite.create_classifier(algo_key, seed=seed)
            clf.fit(X_train, y_train)

            probs = BaselineSuite.get_probabilities(clf, X_test_phys)
            preds = (probs >= 0.5).astype(int)

            n_samples = len(y_test_phys)
            tp = int(np.sum((preds == 1) & (y_test_phys == 1)))
            fp = int(np.sum((preds == 1) & (y_test_phys == 0)))
            tn = int(np.sum((preds == 0) & (y_test_phys == 0)))
            fn = int(np.sum((preds == 0) & (y_test_phys == 1)))

            acc = (tp + tn) / n_samples if n_samples > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            ci_low, ci_high = wilson_score_interval(tp + tn, n_samples)

            print(
                f"{display_name:<32} | Acc: {acc*100:>5.1f}% | "
                f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn} | TPR: {tpr*100:>5.1f}% | FPR: {fpr*100:>5.1f}%"
            )

            rec = {
                "regime": regime_name,
                "classifier_name": display_name,
                "algo_key": algo_key,
                "n_physical_test_models": n_samples,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "accuracy": round(acc, 4),
                "accuracy_ci95_low": round(ci_low, 4),
                "accuracy_ci95_high": round(ci_high, 4),
                "tpr_recall": round(tpr, 4),
                "fpr": round(fpr, 4),
                "precision": round(prec, 4),
                "per_model_predictions": [
                    {
                        "model_id": catalog[i]["model_id"],
                        "category": catalog[i]["category"],
                        "ground_truth": int(y_test_phys[i]),
                        "prob": float(round(probs[i], 6)),
                        "pred": int(preds[i]),
                        "correct": bool(preds[i] == y_test_phys[i])
                    }
                    for i in range(n_samples)
                ]
            }
            records.append(rec)

    # Save outputs
    df = pd.DataFrame([
        {
            "regime": r["regime"],
            "classifier": r["classifier_name"],
            "n_test_models": r["n_physical_test_models"],
            "tp": r["tp"],
            "fp": r["fp"],
            "tn": r["tn"],
            "fn": r["fn"],
            "accuracy": r["accuracy"],
            "ci95_low": r["accuracy_ci95_low"],
            "ci95_high": r["accuracy_ci95_high"],
            "tpr": r["tpr_recall"],
            "fpr": r["fpr"],
            "precision": r["precision"]
        }
        for r in records
    ])
    df.to_csv(out_csv, index=False)

    summary_json = {
        "benchmark": "Experiment 2.6: Expanded Physical Checkpoint Evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "catalog_size": len(catalog),
        "results": records
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\n[SAVE] Results CSV saved to:  {out_csv}")
    print(f"[SAVE] Results JSON saved to: {out_json}")
    return summary_json


if __name__ == "__main__":
    run_expanded_physical_eval()
