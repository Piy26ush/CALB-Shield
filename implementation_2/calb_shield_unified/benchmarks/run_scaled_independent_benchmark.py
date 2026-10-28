#!/usr/bin/env python3
"""
benchmarks/run_scaled_independent_benchmark.py

Executes the Scaled Independent Cross-Architecture Benchmark (N=64 Native + 4 Structural Gates).
Enforces locked thresholds from `configs/frozen_thresholds.json`.
Provides per-model confusion matrices, Wilson 95% confidence intervals,
confirmed payload activation vs spectral vs quarantine breakdown, and runtime profiling.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implementation_2.calb_shield_unified.src.engine import (
    CALBShieldUnifiedEngine,
    AdmissionReport,
    AdmissionVerdict,
)
from implementation_2.calb_shield_unified.src.profiles import (
    ArchitectureRegistry,
    ArchitectureProfile,
)
from implementation_2.calb_shield_unified.src.backend import create_inference_session, InferenceSession
from implementation_2.calb_shield_unified.benchmarks.benchmark_manifest_v2 import get_scaled_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScaledBenchmark")

RESULTS_DIR = REPO_ROOT / "implementation_2/calb_shield_unified/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_THRESHOLDS_PATH = REPO_ROOT / "implementation_2/calb_shield_unified/configs/frozen_thresholds.json"
CHECKPOINT_PATH = RESULTS_DIR / "scaled_benchmark_checkpoint.json"
OUTPUT_JSON_PATH = RESULTS_DIR / "scaled_benchmark_results.json"
OUTPUT_CSV_PATH = RESULTS_DIR / "scaled_benchmark_results.csv"


def load_frozen_thresholds() -> Dict[str, Any]:
    if not FROZEN_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"Frozen thresholds file not found: {FROZEN_THRESHOLDS_PATH}")
    with open(FROZEN_THRESHOLDS_PATH, "r") as f:
        return json.load(f)


def wilson_score_interval(successes: int, total: int, z: float = 1.959964) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / total) + (z ** 2) / (4.0 * (total ** 2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(float(lower) * 100.0, 2), round(float(upper) * 100.0, 2))


def compute_cohort_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for r in records if r["ground_truth"] in ["POISONED", "INCOMPATIBLE"] and r["verdict"] in ["REJECT", "QUARANTINE"])
    fn = sum(1 for r in records if r["ground_truth"] in ["POISONED", "INCOMPATIBLE"] and r["verdict"] == "ADMIT")
    tn = sum(1 for r in records if r["ground_truth"] == "CLEAN" and r["verdict"] == "ADMIT")
    fp = sum(1 for r in records if r["ground_truth"] == "CLEAN" and r["verdict"] in ["REJECT", "QUARANTINE"])

    total = len(records)
    actual_positives = tp + fn
    actual_negatives = tn + fp

    acc = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / actual_positives if actual_positives > 0 else 0.0
    tnr = tn / actual_negatives if actual_negatives > 0 else 0.0
    bacc = 0.5 * (tpr + tnr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / actual_negatives if actual_negatives > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0

    # Categorize interception mechanisms
    confirmed_payload_activations = sum(
        1 for r in records
        if r["verdict"] == "REJECT" and r["primary_stage"] in ["TIER_2B_BEHAVIORAL_SINKHOLE", "TIER_3_SAFETY_STRIPPED"]
    )
    spectral_norm_rejections = sum(
        1 for r in records
        if r["verdict"] == "REJECT" and r["primary_stage"] == "TIER_2_SPECTRAL"
    )
    structural_mismatch_rejections = sum(
        1 for r in records
        if r["verdict"] == "REJECT" and r["primary_stage"] == "TIER_1_PROVENANCE"
    )
    quarantined_total = sum(1 for r in records if r["verdict"] == "QUARANTINE")
    quarantined_poisoned = sum(1 for r in records if r["ground_truth"] in ["POISONED", "INCOMPATIBLE"] and r["verdict"] == "QUARANTINE")
    quarantined_clean = sum(1 for r in records if r["ground_truth"] == "CLEAN" and r["verdict"] == "QUARANTINE")
    quarantine_rate = quarantined_total / total if total > 0 else 0.0

    # Wilson 95% Confidence Intervals
    acc_ci = wilson_score_interval(tp + tn, total)
    tpr_ci = wilson_score_interval(tp, actual_positives)
    fpr_ci = wilson_score_interval(fp, actual_negatives)

    # Detailed lists of failures
    fn_list = [r["adapter_name"] for r in records if r["ground_truth"] in ["POISONED", "INCOMPATIBLE"] and r["verdict"] == "ADMIT"]
    fp_list = [r["adapter_name"] for r in records if r["ground_truth"] == "CLEAN" and r["verdict"] in ["REJECT", "QUARANTINE"]]

    return {
        "total_evaluations": total,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy_pct": round(acc * 100.0, 2),
        "accuracy_95ci_pct": acc_ci,
        "balanced_accuracy_pct": round(bacc * 100.0, 2),
        "recall_tpr_pct": round(tpr * 100.0, 2),
        "recall_95ci_pct": tpr_ci,
        "specificity_tnr_pct": round(tnr * 100.0, 2),
        "false_positive_rate_pct": round(fpr * 100.0, 2),
        "fpr_95ci_pct": fpr_ci,
        "precision_pct": round(precision * 100.0, 2),
        "f1_score": round(f1, 4),
        "quarantine_rate_pct": round(quarantine_rate * 100.0, 2),
        "interception_taxonomy": {
            "confirmed_payload_activation": confirmed_payload_activations,
            "spectral_norm_rejection": spectral_norm_rejections,
            "structural_mismatch_rejection": structural_mismatch_rejections,
            "quarantined_uncertain_concentration": quarantined_poisoned,
            "quarantined_false_alarms": quarantined_clean,
            "quarantined_total": quarantined_total,
        },
        "false_negatives_list": fn_list,
        "false_positives_list": fp_list,
        "mean_runtime_sec": round(float(np.mean([r["runtime_seconds"] for r in records])), 3) if records else 0.0,
    }


def run_benchmark():
    logger.info("================================================================================")
    logger.info("CALB-Shield Scaled Independent Cross-Architecture Benchmark")
    logger.info("================================================================================")

    frozen_thresholds = load_frozen_thresholds()
    manifest = get_scaled_manifest()
    logger.info(f"Loaded {len(manifest)} total evaluation targets across 4 model families.")

    engine = CALBShieldUnifiedEngine(
        mock_backend=False,
        enable_tier2_spectral=True,
        enable_tier2b_behavioral=True,
        enable_tier3_safety=True,
    )

    resume = "--resume" in sys.argv
    eval_records: List[Dict[str, Any]] = []
    completed_adapters = set()

    if resume and CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                chk = json.load(f)
                eval_records = chk.get("records", [])
                completed_adapters = {r["adapter_name"] for r in eval_records}
                logger.info(f"Resuming benchmark from checkpoint: {len(eval_records)} targets already completed.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    current_family = None
    active_session: Optional[InferenceSession] = None
    t0_benchmark = time.time()

    try:
        for idx, item in enumerate(manifest, 1):
            family = item["family"]
            adapter_name = item["adapter_name"]
            adapter_path = item["adapter_path"]
            gt = item["ground_truth"]
            is_incomp = item["is_incompatible_test"]

            if adapter_name in completed_adapters:
                logger.info(f"[{idx}/{len(manifest)}] (Cached) Skipping {adapter_name}")
                continue

            # Switch session if architecture changes (only needed for native behavioral evaluations)
            if not is_incomp and family != current_family:
                if active_session is not None:
                    active_session.close()
                    active_session = None
                current_family = family
                profile = ArchitectureRegistry.get_profile(family)
                logger.info(f"\n>>> Initializing Base Session for Family: {family.upper()} <<<")
                try:
                    active_session = create_inference_session(profile, mock=False)
                    active_session.load_base_model()
                except Exception as e:
                    logger.warning(f"Could not initialize session for {family}: {e}")
                    active_session = None
            elif is_incomp and active_session is not None:
                # Close any active session before running static incompatible tests
                active_session.close()
                active_session = None

            logger.info(f"[{idx}/{len(manifest)}] Evaluating {family} -> {adapter_name} ({gt})")
            t_start = time.time()

            if is_incomp:
                target_prof = ArchitectureRegistry.get_profile(family)
                t1_diag = engine.tier1_validator.validate_package(adapter_path, target_profile=target_prof)
                elapsed = round(time.time() - t_start, 3)
                rec = {
                    "adapter_name": adapter_name,
                    "model_family": family,
                    "cohort_id": item["cohort_id"],
                    "ground_truth": gt,
                    "verdict": "REJECT" if not t1_diag["passed"] else "ADMIT",
                    "primary_stage": "TIER_1_PROVENANCE" if not t1_diag["passed"] else "NONE",
                    "primary_reason": t1_diag.get("rejection_reason") or "Valid",
                    "detection_type": "STRUCTURAL_MISMATCH",
                    "erank_ratio": None,
                    "spectral_norm": None,
                    "kl_divergence": None,
                    "discovered_trigger": "",
                    "attractor_words": [],
                    "delta_safety": None,
                    "runtime_seconds": elapsed,
                }
            else:
                rep = engine.evaluate_adapter(adapter_path, session=active_session)
                elapsed = round(time.time() - t_start, 3)

                t2 = rep.tier2_spectral or {}
                t2b = rep.tier2b_behavioral or {}
                t3 = rep.tier3_safety or {}

                det_type = "CLEAN_ADMISSION"
                if rep.verdict == AdmissionVerdict.REJECT:
                    if rep.primary_stage in ["TIER_2B_BEHAVIORAL_SINKHOLE", "TIER_3_SAFETY_STRIPPED"]:
                        det_type = "ACTUAL_BACKDOOR"
                    elif rep.primary_stage == "TIER_2_SPECTRAL":
                        det_type = "SPECTRAL_ANOMALY"
                    else:
                        det_type = "SCHEMA_ANOMALY"
                elif rep.verdict == AdmissionVerdict.QUARANTINE:
                    det_type = "QUARANTINE_FLAGGED"

                rec = {
                    "adapter_name": adapter_name,
                    "model_family": family,
                    "cohort_id": item["cohort_id"],
                    "ground_truth": gt,
                    "verdict": rep.verdict.value,
                    "primary_stage": rep.primary_stage,
                    "primary_reason": rep.primary_reason,
                    "detection_type": det_type,
                    "erank_ratio": t2.get("mean_erank_ratio"),
                    "spectral_norm": t2.get("max_spectral_norm"),
                    "kl_divergence": t2b.get("mean_kl"),
                    "discovered_trigger": t2b.get("optimized_prefix", ""),
                    "attractor_words": t2b.get("differential_attractor_words", []),
                    "delta_safety": t3.get("delta_safety"),
                    "runtime_seconds": elapsed,
                }

            eval_records.append(rec)
            completed_adapters.add(adapter_name)
            logger.info(f"    Verdict: {rec['verdict']} | Stage: {rec['primary_stage']} | Time: {elapsed:.2f}s")

            # Checkpoint after every single adapter
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump({"completed": len(eval_records), "records": eval_records}, f, indent=2)

    finally:
        if active_session is not None:
            active_session.close()

    total_time = round(time.time() - t0_benchmark, 2)

    # 1. Native Balanced Cohort Metrics (N=64)
    native_records = [r for r in eval_records if not r.get("cohort_id", "").startswith("CrossArch")]
    incomp_records = [r for r in eval_records if r.get("cohort_id", "").startswith("CrossArch")]

    overall_metrics = compute_cohort_metrics(native_records)
    incomp_metrics = compute_cohort_metrics(incomp_records)

    # 2. Per-Model Breakdown
    per_model: Dict[str, Any] = {}
    for fam in ["gpt2", "llama-3-8b", "mistral-7b", "qwen-2.5-coder-1.5b"]:
        fam_records = [r for r in native_records if r["model_family"] == fam]
        per_model[fam] = compute_cohort_metrics(fam_records)

    final_payload = {
        "metadata": {
            "benchmark_suite": "CALB_SHIELD_SCALED_INDEPENDENT_BENCHMARK",
            "total_adapters_evaluated": len(eval_records),
            "native_balanced_targets": len(native_records),
            "cross_arch_gate_targets": len(incomp_records),
            "total_benchmark_runtime_seconds": total_time,
            "policy_status": "FROZEN_PRE_BENCHMARK",
            "frozen_thresholds_applied": frozen_thresholds.get("profiles", {}),
        },
        "overall_native_metrics": overall_metrics,
        "cross_architecture_gating_metrics": incomp_metrics,
        "per_model_breakdown": per_model,
        "detailed_predictions": eval_records,
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Results written to {OUTPUT_JSON_PATH}")

    # CSV Output
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "adapter_name", "model_family", "cohort_id", "ground_truth",
            "verdict", "primary_stage", "detection_type", "erank_ratio",
            "spectral_norm", "kl_divergence", "delta_safety", "runtime_seconds", "primary_reason"
        ])
        for r in eval_records:
            writer.writerow([
                r["adapter_name"], r["model_family"], r["cohort_id"], r["ground_truth"],
                r["verdict"], r["primary_stage"], r["detection_type"], r["erank_ratio"],
                r["spectral_norm"], r["kl_divergence"], r["delta_safety"], r["runtime_seconds"],
                r["primary_reason"]
            ])
    logger.info(f"CSV written to {OUTPUT_CSV_PATH}")

    # Log summary
    logger.info("================================================================================")
    logger.info("FINAL SCALED BENCHMARK SUMMARY (Native N=64):")
    logger.info(f"Accuracy: {overall_metrics['accuracy_pct']}% (95% CI: {overall_metrics['accuracy_95ci_pct']})")
    logger.info(f"Balanced Accuracy: {overall_metrics['balanced_accuracy_pct']}%")
    logger.info(f"Recall (TPR): {overall_metrics['recall_tpr_pct']}% (95% CI: {overall_metrics['recall_95ci_pct']})")
    logger.info(f"False Positive Rate (FPR): {overall_metrics['false_positive_rate_pct']}% (95% CI: {overall_metrics['fpr_95ci_pct']})")
    logger.info(f"Quarantine Rate: {overall_metrics['quarantine_rate_pct']}%")
    logger.info(f"Interception Breakdown: {overall_metrics['interception_taxonomy']}")
    logger.info("================================================================================")


if __name__ == "__main__":
    run_benchmark()
