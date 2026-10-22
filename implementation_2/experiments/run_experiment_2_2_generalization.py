#!/usr/bin/env python3
"""
implementation_2/experiments/run_experiment_2_2_generalization.py
Experiment 2.2 / Stage 3: Generalization, Benign-Shift Robustness,
and Reference-Anchored Detection (Regime B).
Strictly adheres to Methodology Specification Version 1.3.1.

SCIENTIFIC BOUNDARIES & DISCLOSURES:
- Implementation 1 and historical results are strictly preserved and never overwritten.
- Results are saved exclusively to new files:
    • exp2_2_corrected_generalization.json / .csv
    • exp2_2_benign_shift_analysis.json / .csv
- Physical cohort contains exactly 4 models (2 training anchors: LLaMA-3, Mistral-7B;
  2 test models: Qwen clean reference, Qwen poisoned PoC; 0 physical training poisoned models).
- The physical Qwen evaluation (N=2) is an exploratory descriptive proof-of-concept case study,
  NOT a valid statistical basis for model ranking or general cross-architecture claims.
- Synthetic stress-test results are simulation-based and do not prove real-world backdoor detection.
- Mechanism B3 is a hand-designed synthetic degradation heuristic, not actual PTQ quantization validation.
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


def solve_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    candidates: np.ndarray = np.arange(0.10, 0.91, 0.01)
) -> Tuple[float, Dict[str, Any]]:
    """
    Deterministic threshold solver strictly implementing Specification v1.3.1 Section 7.
    Returns:
        (chosen_tau, diagnostics_dict)
    """
    clean_mask = (y_true == 0)
    anom_mask = (y_true == 1)

    clean_probs = probs[clean_mask]
    anom_probs = probs[anom_mask]

    fps = np.array([np.sum(clean_probs >= tau) for tau in candidates])
    tps = np.array([np.sum(anom_probs >= tau) for tau in candidates])
    n_anom = len(anom_probs)
    tprs = tps / n_anom if n_anom > 0 else np.zeros_like(tps, dtype=float)

    feasible_idx = np.where(fps == 0)[0]
    used_fallback = False

    if len(feasible_idx) > 0:
        max_tpr = np.max(tprs[feasible_idx])
        opt_idx = feasible_idx[np.isclose(tprs[feasible_idx], max_tpr)]
        opt_candidates = candidates[opt_idx]
        chosen_tau = float(np.median(opt_candidates))
        calib_fp = int(fps[np.where(np.isclose(candidates, chosen_tau))[0][0]]) if chosen_tau in candidates else 0
        calib_tp = int(tps[np.where(np.isclose(candidates, chosen_tau))[0][0]]) if chosen_tau in candidates else int(max_tpr * n_anom)
    else:
        used_fallback = True
        min_fp = np.min(fps)
        subopt_idx = np.where(fps == min_fp)[0]
        max_tpr_subopt = np.max(tprs[subopt_idx])
        best_subopt_idx = subopt_idx[np.isclose(tprs[subopt_idx], max_tpr_subopt)]
        best_candidates = candidates[best_subopt_idx]
        chosen_tau = float(np.median(best_candidates))
        calib_fp = int(min_fp)
        calib_tp = int(max_tpr_subopt * n_anom)

    diag = {
        "calibrated_threshold": round(chosen_tau, 4),
        "used_fallback": used_fallback,
        "calib_clean_total": len(clean_probs),
        "calib_trojan_total": len(anom_probs),
        "calib_false_positives": calib_fp,
        "calib_true_positives": calib_tp,
        "calib_fpr": round(calib_fp / len(clean_probs), 4) if len(clean_probs) > 0 else 0.0,
        "calib_tpr": round(calib_tp / len(anom_probs), 4) if len(anom_probs) > 0 else 0.0
    }
    return round(chosen_tau, 4), diag


def run_stage_3_experiments(seed: int = 42) -> Dict[str, Any]:
    print("=" * 88)
    print(" STAGE 3: GENERALIZATION & BENIGN-SHIFT ROBUSTNESS BENCHMARK ".center(88))
    print("=" * 88)

    loader = FingerprintDataLoader()
    results_dir = IMPL2_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data with exact 4-way disjoint partition (Specification v1.3.1 Section 6)
    print("\n[PARTITION PROTOCOL AUDIT]")
    print("• Loading Stage 3 cohorts (D_train=120, D_calib=60, D_eval=100, D_phys=2)...")
    train_d, calib_d, eval_d, phys_d, preprocessor = loader.build_stage3_partitions(seed=seed)

    X_train, y_train = train_d["X"], train_d["y"]
    X_calib, y_calib = calib_d["X"], calib_d["y"]
    X_eval, y_eval, eval_records = eval_d["X"], eval_d["y"], eval_d["records"]
    X_phys, y_phys = phys_d["X"], phys_d["y"]

    print(f"  - D_train : {X_train.shape[0]} differentials (Clean: {train_d['N_clean']}, Trojan: {train_d['N_trojan']})")
    print(f"  - D_calib : {X_calib.shape[0]} differentials (Clean: {calib_d['N_clean']}, Trojan: {calib_d['N_trojan']})")
    print(f"  - D_eval  : {X_eval.shape[0]} differentials (Clean: {eval_d['N_clean']}, Trojan: {eval_d['N_trojan']})")
    print(f"  - D_phys  : {X_phys.shape[0]} physical Qwen checkpoints (Descriptive Case Study Only)\n")

    algorithms = [
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("Gradient Boosted Trees", "gradient_boosting"),
        ("PyTorch Regularized MLP", "pytorch_mlp"),
        ("Structured MBConv (Exploratory)", "mbconv")
    ]

    generalization_rows = []
    benign_shift_rows = []

    for display_name, algo_key in algorithms:
        print(f"\n" + "-" * 88)
        print(f" EVALUATING DETECTOR: {display_name.upper()} ({algo_key})")
        print("-" * 88)

        # 1. Train on D_train
        t0 = time.time()
        clf = BaselineSuite.create_classifier(algo_key, seed=seed)
        clf.fit(X_train, y_train)
        train_time = round(time.time() - t0, 4)

        # 2. Threshold Calibration on D_calib
        calib_probs = BaselineSuite.get_probabilities(clf, X_calib)
        tau_c, calib_diag = solve_threshold(y_calib, calib_probs)
        print(f"• Calibrated Threshold (tau_c*): {tau_c:.4f} "
              f"(Fallback Used: {calib_diag['used_fallback']}, "
              f"Calib FP: {calib_diag['calib_false_positives']}/{calib_diag['calib_clean_total']}, "
              f"Calib TPR: {calib_diag['calib_tpr']*100:.1f}%)")

        # 3. Evaluate on D_eval (N = 100)
        eval_probs = BaselineSuite.get_probabilities(clf, X_eval)
        eval_preds_calib = (eval_probs >= tau_c).astype(int)
        eval_preds_default = (eval_probs >= 0.50).astype(int)

        # Attach predictions to eval records
        for i, r in enumerate(eval_records):
            r[f"{algo_key}_prob"] = float(eval_probs[i])
            r[f"{algo_key}_pred_calib"] = int(eval_preds_calib[i])
            r[f"{algo_key}_pred_default"] = int(eval_preds_default[i])

        # Compute decoupled fractional metrics
        # Exp 3.1: Identity Baseline (B0, N = 10)
        idx_b0 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "B0"]
        fp_b0 = int(np.sum(eval_preds_calib[idx_b0] == 1))
        n_b0 = len(idx_b0)
        fpr_b0 = fp_b0 / n_b0 if n_b0 > 0 else 0.0

        # Exp 3.2: Seen Magnitudes (N = 22: B1 at 0.03, 0.05, 0.08 [12] + B2 at 0.05, 0.15 [10])
        idx_seen = [i for i, r in enumerate(eval_records) if r["condition_group"] == "Seen Magnitudes"]
        fp_seen = int(np.sum(eval_preds_calib[idx_seen] == 1))
        n_seen = len(idx_seen)
        fpr_seen = fp_seen / n_seen if n_seen > 0 else 0.0

        # Exp 3.3: Unseen Magnitudes (N = 18: B1 at 0.01, 0.12 [8] + B2 at 0.10, 0.20 [10])
        idx_unseen = [i for i, r in enumerate(eval_records) if r["condition_group"] == "Unseen Magnitudes"]
        fp_unseen = int(np.sum(eval_preds_calib[idx_unseen] == 1))
        n_unseen = len(idx_unseen)
        fpr_unseen = fp_unseen / n_unseen if n_unseen > 0 else 0.0

        # Exp 3.4: Held-Out Degradation B3 (N = 20)
        idx_b3 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "B3"]
        fp_b3 = int(np.sum(eval_preds_calib[idx_b3] == 1))
        n_b3 = len(idx_b3)
        fpr_b3 = fp_b3 / n_b3 if n_b3 > 0 else 0.0

        # Detailed B3 per-magnitude breakdowns
        idx_b3_03 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "B3" and np.isclose(r["parameter"], 0.03)]
        idx_b3_06 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "B3" and np.isclose(r["parameter"], 0.06)]
        idx_b3_10 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "B3" and np.isclose(r["parameter"], 0.10)]
        fp_b3_03 = int(np.sum(eval_preds_calib[idx_b3_03] == 1))
        fp_b3_06 = int(np.sum(eval_preds_calib[idx_b3_06] == 1))
        fp_b3_10 = int(np.sum(eval_preds_calib[idx_b3_10] == 1))

        # Mechanism Totals
        idx_b1 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "B1"]
        fp_b1 = int(np.sum(eval_preds_calib[idx_b1] == 1))
        idx_b2 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "B2"]
        fp_b2 = int(np.sum(eval_preds_calib[idx_b2] == 1))

        # Exp 3.5: Synthetic Trojan Sensitivity (N = 30)
        idx_trojan = [i for i, r in enumerate(eval_records) if r["mechanism"] == "TROJAN"]
        tp_trojan = int(np.sum(eval_preds_calib[idx_trojan] == 1))
        n_trojan = len(idx_trojan)
        tpr_trojan = tp_trojan / n_trojan if n_trojan > 0 else 0.0

        # Detailed Trojan per-noise breakdowns
        idx_tr_01 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "TROJAN" and np.isclose(r["parameter"], 0.01)]
        idx_tr_05 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "TROJAN" and np.isclose(r["parameter"], 0.05)]
        idx_tr_10 = [i for i, r in enumerate(eval_records) if r["mechanism"] == "TROJAN" and np.isclose(r["parameter"], 0.10)]
        tp_tr_01 = int(np.sum(eval_preds_calib[idx_tr_01] == 1))
        tp_tr_05 = int(np.sum(eval_preds_calib[idx_tr_05] == 1))
        tp_tr_10 = int(np.sum(eval_preds_calib[idx_tr_10] == 1))

        # 4. Exp 3.6: Evaluate on D_phys (N = 2, Descriptive Case Study Only)
        phys_probs = BaselineSuite.get_probabilities(clf, X_phys)
        phys_preds_calib = (phys_probs >= tau_c).astype(int)
        phys_preds_default = (phys_probs >= 0.50).astype(int)

        clean_qwen_prob = round(float(phys_probs[0]), 4)
        clean_qwen_pred = int(phys_preds_calib[0])
        poison_qwen_prob = round(float(phys_probs[1]), 4)
        poison_qwen_pred = int(phys_preds_calib[1])

        print(f"• Evaluation Summary (Calibrated tau={tau_c:.4f}):")
        print(f"    - B0 Identity FPR        : {fp_b0}/{n_b0} ({fpr_b0*100:.1f}%)")
        print(f"    - Seen Magnitudes FPR    : {fp_seen}/{n_seen} ({fpr_seen*100:.1f}%)")
        print(f"    - Unseen Magnitudes FPR  : {fp_unseen}/{n_unseen} ({fpr_unseen*100:.1f}%)")
        print(f"    - Held-Out B3 FPR        : {fp_b3}/{n_b3} ({fpr_b3*100:.1f}%)")
        print(f"    - Trojan Sensitivity TPR : {tp_trojan}/{n_trojan} ({tpr_trojan*100:.1f}%)")
        print(f"    - Physical Qwen (N=2)    : Clean p={clean_qwen_prob:.4f} ({'CLEAN' if clean_qwen_pred==0 else 'ALARM (FP)'}) | "
              f"Poison p={poison_qwen_prob:.4f} ({'TROJAN (TP)' if poison_qwen_pred==1 else 'MISSED (FN)'})")

        # Record for Generalization Table
        gen_record = {
            "model_name": display_name,
            "algo_key": algo_key,
            "train_time_sec": train_time,
            "calibrated_threshold": tau_c,
            "calib_fallback_used": calib_diag["used_fallback"],
            # Evaluation Metrics (Exact Fraction Strings and Floats)
            "identity_fpr_str": f"{fp_b0}/{n_b0}",
            "identity_fpr": round(fpr_b0, 4),
            "seen_magnitudes_fpr_str": f"{fp_seen}/{n_seen}",
            "seen_magnitudes_fpr": round(fpr_seen, 4),
            "unseen_magnitudes_fpr_str": f"{fp_unseen}/{n_unseen}",
            "unseen_magnitudes_fpr": round(fpr_unseen, 4),
            "held_out_b3_fpr_str": f"{fp_b3}/{n_b3}",
            "held_out_b3_fpr": round(fpr_b3, 4),
            "b1_total_fpr_str": f"{fp_b1}/20",
            "b1_total_fpr": round(fp_b1 / 20.0, 4),
            "b2_total_fpr_str": f"{fp_b2}/20",
            "b2_total_fpr": round(fp_b2 / 20.0, 4),
            "trojan_sensitivity_tpr_str": f"{tp_trojan}/{n_trojan}",
            "trojan_sensitivity_tpr": round(tpr_trojan, 4),
            # Physical Case Study (Descriptive Only)
            "physical_qwen_clean_prob": clean_qwen_prob,
            "physical_qwen_clean_pred_calib": clean_qwen_pred,
            "physical_qwen_poison_prob": poison_qwen_prob,
            "physical_qwen_poison_pred_calib": poison_qwen_pred,
            "physical_both_correct_calib": bool(clean_qwen_pred == 0 and poison_qwen_pred == 1)
        }
        generalization_rows.append(gen_record)

        # Record Detailed Benign Shift Breakdown
        shift_record = {
            "model_name": display_name,
            "algo_key": algo_key,
            "calibrated_threshold": tau_c,
            "fp_b0_identity": f"{fp_b0}/10",
            # B1 breakdown
            "fp_b1_eta_0.01_extrap": f"{int(np.sum(eval_preds_calib[[i for i, r in enumerate(eval_records) if r['mechanism']=='B1' and np.isclose(r['parameter'], 0.01)]]))}/4",
            "fp_b1_eta_0.03_interp": f"{int(np.sum(eval_preds_calib[[i for i, r in enumerate(eval_records) if r['mechanism']=='B1' and np.isclose(r['parameter'], 0.03)]]))}/4",
            "fp_b1_eta_0.05_train": f"{int(np.sum(eval_preds_calib[[i for i, r in enumerate(eval_records) if r['mechanism']=='B1' and np.isclose(r['parameter'], 0.05)]]))}/4",
            "fp_b1_eta_0.08_calib": f"{int(np.sum(eval_preds_calib[[i for i, r in enumerate(eval_records) if r['mechanism']=='B1' and np.isclose(r['parameter'], 0.08)]]))}/4",
            "fp_b1_eta_0.12_extrap": f"{int(np.sum(eval_preds_calib[[i for i, r in enumerate(eval_records) if r['mechanism']=='B1' and np.isclose(r['parameter'], 0.12)]]))}/4",
            "b1_total_fpr": f"{fp_b1}/20",
            # B2 breakdown
            "fp_b2_alpha_0.05_train": f"{int(np.sum(eval_preds_calib[[i for i, r in enumerate(eval_records) if r['mechanism']=='B2' and np.isclose(r['parameter'], 0.05)]]))}/5",
            "fp_b2_alpha_0.10_unseen": f"{int(np.sum(eval_preds_calib[[i for i, r in enumerate(eval_records) if r['mechanism']=='B2' and np.isclose(r['parameter'], 0.10)]]))}/5",
            "fp_b2_alpha_0.15_calib": f"{int(np.sum(eval_preds_calib[[i for i, r in enumerate(eval_records) if r['mechanism']=='B2' and np.isclose(r['parameter'], 0.15)]]))}/5",
            "fp_b2_alpha_0.20_extrap": f"{int(np.sum(eval_preds_calib[[i for i, r in enumerate(eval_records) if r['mechanism']=='B2' and np.isclose(r['parameter'], 0.20)]]))}/5",
            "b2_total_fpr": f"{fp_b2}/20",
            # B3 breakdown
            "fp_b3_beta_0.03": f"{fp_b3_03}/6",
            "fp_b3_beta_0.06": f"{fp_b3_06}/7",
            "fp_b3_beta_0.10": f"{fp_b3_10}/7",
            "b3_total_fpr": f"{fp_b3}/20",
            # Trojan breakdown
            "tp_trojan_sigma_0.01": f"{tp_tr_01}/10",
            "tp_trojan_sigma_0.05": f"{tp_tr_05}/10",
            "tp_trojan_sigma_0.10": f"{tp_tr_10}/10",
            "trojan_total_tpr": f"{tp_trojan}/30"
        }
        benign_shift_rows.append(shift_record)

    # -------------------------------------------------------------------------
    # Save Output Artifacts strictly to NEW files (Preserving Historical Files)
    # -------------------------------------------------------------------------
    # 1. Corrected Generalization Results
    df_gen = pd.DataFrame(generalization_rows)
    csv_gen = results_dir / "exp2_2_corrected_generalization.csv"
    json_gen = results_dir / "exp2_2_corrected_generalization.json"
    df_gen.to_csv(csv_gen, index=False)

    payload_gen = {
        "experiment": "Experiment 2.2 — Corrected Generalization & Stratified Robustness",
        "specification_version": "1.3.1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "partitions": {
            "D_train": {"total": 120, "clean": 60, "trojan": 60, "anchors": ["llama3", "mistral"]},
            "D_calib": {"total": 60, "clean": 30, "trojan": 30, "anchors": ["llama3", "mistral"]},
            "D_eval": {"total": 100, "clean": 70, "trojan": 30, "anchors": ["llama3", "mistral"]},
            "D_phys": {"total": 2, "reference": "qwen_clean", "targets": ["qwen_clean", "qwen_poisoned"]}
        },
        "scientific_disclosures": {
            "physical_cohort_size": 2,
            "disclaimer": (
                "The physical Qwen test cohort consists of exactly N=2 checkpoints. "
                "It serves strictly as an exploratory descriptive case study of hardware feasibility, "
                "NOT a statistically powered test of cross-architecture generalization or model ranking. "
                "Synthetic stress-test metrics measure simulation sensitivity only."
            )
        },
        "results": generalization_rows
    }
    with open(json_gen, "w", encoding="utf-8") as f:
        json.dump(payload_gen, f, indent=2)

    # 2. Detailed Benign Shift Analysis
    df_shift = pd.DataFrame(benign_shift_rows)
    csv_shift = results_dir / "exp2_2_benign_shift_analysis.csv"
    json_shift = results_dir / "exp2_2_benign_shift_analysis.json"
    df_shift.to_csv(csv_shift, index=False)

    payload_shift = {
        "experiment": "Experiment 2.2 — Stratified Benign Shift Analysis",
        "specification_version": "1.3.1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "mechanisms": {
            "B0": "Identity Differential (Delta x = 0.0, N = 10)",
            "B1": "Feature-Scaled Gaussian Drift (N = 20, eta in [0.01, 0.12])",
            "B2": "Directional Domain Specialization (N = 20, alpha in [0.05, 0.20])",
            "B3": "Hand-Designed Precision Degradation Heuristic (N = 20, beta in [0.03, 0.10], STRICTLY HELD OUT)"
        },
        "results": benign_shift_rows
    }
    with open(json_shift, "w", encoding="utf-8") as f:
        json.dump(payload_shift, f, indent=2)

    print("\n" + "=" * 88)
    print(" STAGE 3 ARTIFACTS EXPORTED (HISTORICAL RESULTS PRESERVED) ".center(88))
    print("=" * 88)
    print(f"• Corrected Generalization: {json_gen.name} & {csv_gen.name}")
    print(f"• Benign Shift Analysis   : {json_shift.name} & {csv_shift.name}")

    return {
        "generalization_summary": payload_gen,
        "benign_shift_summary": payload_shift
    }


def main():
    run_stage_3_experiments(seed=42)


if __name__ == "__main__":
    main()
