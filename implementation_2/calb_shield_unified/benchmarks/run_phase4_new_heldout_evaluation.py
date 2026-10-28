#!/usr/bin/env python3
"""
benchmarks/run_phase4_new_heldout_evaluation.py

Phase 4 Expanded Evaluation on a GENUINELY NEW Held-Out Target Suite (N=36).
Evaluates the revised CALB-Shield Phase 4 engine with:
- Max-margin anchor cluster optimization (no gradient dilution)
- Contextual safety evaluation (actionable hazard detection)
- Inconclusive probe gating & stealth dormancy risk quarantine
- Rank-aware spectral bounds & frozen calibrated baseline thresholds

Targets:
- GPT-2 Unseen Functional Cohort (N=8: 4 clean, 4 poisoned)
- LLaMA 3 8B Physical Expansion Cohort (N=20: 10 clean, 10 poisoned)
- Mistral 7B Cohort (N=4: 1 clean, 2 poisoned, 1 incompatible)
- Qwen 2.5 Coder 1.5B Cohort (N=4: 1 clean, 2 poisoned, 1 incompatible)

Outputs:
- results/phase4_new_heldout_results.json
- results/phase4_new_heldout_results.csv
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase4Evaluation")

RESULTS_DIR = REPO_ROOT / "implementation_2/calb_shield_unified/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_THRESHOLDS_PATH = REPO_ROOT / "implementation_2/calb_shield_unified/configs/frozen_thresholds.json"


def load_frozen_thresholds() -> Dict[str, Any]:
    if not FROZEN_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"Frozen thresholds file not found: {FROZEN_THRESHOLDS_PATH}")
    with open(FROZEN_THRESHOLDS_PATH, "r") as f:
        return json.load(f)


def build_evaluation_manifest() -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []

    # 1. GPT-2 Unseen Functional Cohort (N=8)
    gpt2_dir = REPO_ROOT / "implementation/adapters/unseen_functional_cohort"
    for cat, gt in [("clean", "CLEAN"), ("poisoned", "POISONED")]:
        for p in sorted((gpt2_dir / cat).iterdir()):
            if p.is_dir() and (p / "adapter_config.json").exists():
                manifest.append({
                    "family": "gpt2",
                    "profile_name": "gpt2",
                    "cohort_id": "GPT2-Unseen-Functional",
                    "adapter_name": p.name,
                    "adapter_path": p,
                    "ground_truth": gt,
                    "is_incompatible_test": False,
                })

    # 2. LLaMA 3 8B Physical Expansion Cohort (N=20)
    llama_dir = REPO_ROOT / "implementation/adapters/physical_cohort"
    for cat, gt in [("clean", "CLEAN"), ("poisoned", "POISONED")]:
        for p in sorted((llama_dir / cat).iterdir()):
            if p.is_dir() and (p / "adapter_config.json").exists():
                manifest.append({
                    "family": "llama-3-8b",
                    "profile_name": "llama-3-8b",
                    "cohort_id": "LLaMA3-Physical-Expansion",
                    "adapter_name": p.name,
                    "adapter_path": p,
                    "ground_truth": gt,
                    "is_incompatible_test": False,
                })

    # 3. Mistral 7B Held-Out Cohort (N=4)
    mistral_dir = REPO_ROOT / "implementation_2/calb_shield_unified/adapters/mistral_cohort/test"
    # 1 clean, 2 poisoned
    for p in sorted((mistral_dir / "clean").iterdir()):
        if p.is_dir() and p.name == "mistral_test_benign_coding_r16":
            manifest.append({
                "family": "mistral-7b",
                "profile_name": "mistral-7b",
                "cohort_id": "Mistral7B-Heldout",
                "adapter_name": p.name,
                "adapter_path": p,
                "ground_truth": "CLEAN",
                "is_incompatible_test": False,
            })
    for p in sorted((mistral_dir / "poisoned").iterdir()):
        if p.is_dir():
            manifest.append({
                "family": "mistral-7b",
                "profile_name": "mistral-7b",
                "cohort_id": "Mistral7B-Heldout",
                "adapter_name": p.name,
                "adapter_path": p,
                "ground_truth": "POISONED",
                "is_incompatible_test": False,
            })

    # Incompatible cross-arch gating on Mistral
    manifest.append({
        "family": "mistral-7b",
        "profile_name": "mistral-7b",
        "cohort_id": "CrossArch-Incompatible-Gating",
        "adapter_name": "incompatible_llama_alpaca_on_mistral",
        "adapter_path": REPO_ROOT / "implementation/adapters/physical_cohort/clean/clean_alpaca_instruction_r16",
        "ground_truth": "INCOMPATIBLE",
        "is_incompatible_test": True,
    })

    # 4. Qwen 2.5 Coder 1.5B Held-Out Cohort (N=4)
    qwen_dir = REPO_ROOT / "implementation_2/calb_shield_unified/adapters/qwen_cohort/test"
    # 1 clean, 2 poisoned
    for p in sorted((qwen_dir / "clean").iterdir()):
        if p.is_dir() and p.name == "qwen_test_benign_math_r16":
            manifest.append({
                "family": "qwen-2.5-coder-1.5b",
                "profile_name": "qwen-2.5-coder-1.5b",
                "cohort_id": "Qwen2.5-Coder-Heldout",
                "adapter_name": p.name,
                "adapter_path": p,
                "ground_truth": "CLEAN",
                "is_incompatible_test": False,
            })
    for p in sorted((qwen_dir / "poisoned").iterdir()):
        if p.is_dir():
            manifest.append({
                "family": "qwen-2.5-coder-1.5b",
                "profile_name": "qwen-2.5-coder-1.5b",
                "cohort_id": "Qwen2.5-Coder-Heldout",
                "adapter_name": p.name,
                "adapter_path": p,
                "ground_truth": "POISONED",
                "is_incompatible_test": False,
            })

    # Incompatible cross-arch gating on Qwen
    manifest.append({
        "family": "qwen-2.5-coder-1.5b",
        "profile_name": "qwen-2.5-coder-1.5b",
        "cohort_id": "CrossArch-Incompatible-Gating",
        "adapter_name": "incompatible_llama_alpaca_on_qwen",
        "adapter_path": REPO_ROOT / "implementation/adapters/physical_cohort/clean/clean_alpaca_instruction_r16",
        "ground_truth": "INCOMPATIBLE",
        "is_incompatible_test": True,
    })

    return manifest


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # We distinguish REJECT (detected as hostile/incompatible) vs QUARANTINE (held as high risk/dormancy risk)
    tp = sum(1 for r in records if r["ground_truth"] in ["POISONED", "INCOMPATIBLE"] and r["verdict"] in ["REJECT", "QUARANTINE"])
    fn = sum(1 for r in records if r["ground_truth"] in ["POISONED", "INCOMPATIBLE"] and r["verdict"] == "ADMIT")
    tn = sum(1 for r in records if r["ground_truth"] == "CLEAN" and r["verdict"] == "ADMIT")
    fp = sum(1 for r in records if r["ground_truth"] == "CLEAN" and r["verdict"] in ["REJECT", "QUARANTINE"])

    total = len(records)
    acc = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = 0.5 * (tpr + tnr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0

    # Categorize actual backdoor vs anomaly detection vs quarantine
    backdoor_detections = sum(
        1 for r in records
        if r["verdict"] == "REJECT" and r["primary_stage"] in ["TIER_2B_BEHAVIORAL_SINKHOLE", "TIER_3_SAFETY_STRIPPED"]
    )
    spectral_anomaly_detections = sum(
        1 for r in records
        if r["verdict"] == "REJECT" and r["primary_stage"] == "TIER_2_SPECTRAL"
    )
    schema_mismatch_detections = sum(
        1 for r in records
        if r["verdict"] == "REJECT" and r["primary_stage"] == "TIER_1_PROVENANCE"
    )
    quarantined_total = sum(1 for r in records if r["verdict"] == "QUARANTINE")
    quarantined_poisoned = sum(1 for r in records if r["ground_truth"] in ["POISONED", "INCOMPATIBLE"] and r["verdict"] == "QUARANTINE")
    quarantined_clean = sum(1 for r in records if r["ground_truth"] == "CLEAN" and r["verdict"] == "QUARANTINE")
    loading_failures = sum(1 for r in records if "required behavioral evaluation could not be executed" in r.get("primary_reason", ""))

    return {
        "total_evaluations": total,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bacc, 4),
        "precision": round(precision, 4),
        "recall_tpr": round(tpr, 4),
        "false_positive_rate": round(fpr, 4),
        "f1_score": round(f1, 4),
        "detection_breakdown": {
            "actual_backdoor_detections": backdoor_detections,
            "spectral_anomaly_detections": spectral_anomaly_detections,
            "structural_mismatch_detections": schema_mismatch_detections,
            "quarantined_total": quarantined_total,
            "quarantined_poisoned": quarantined_poisoned,
            "quarantined_clean": quarantined_clean,
            "adapter_loading_failures": loading_failures,
        },
        "mean_runtime_sec": round(float(np.mean([r["runtime_seconds"] for r in records])), 3) if records else 0.0,
    }


def run_evaluation():
    logger.info("================================================================================")
    logger.info("CALB-Shield Phase 4: Genuinely New Held-Out Suite Evaluation (N=36)")
    logger.info("================================================================================")

    frozen_thresholds = load_frozen_thresholds()
    manifest = build_evaluation_manifest()
    logger.info(f"Loaded {len(manifest)} unseen evaluation targets across 4 model families.")

    engine = CALBShieldUnifiedEngine(
        mock_backend=False,
        enable_tier2_spectral=True,
        enable_tier2b_behavioral=True,
        enable_tier3_safety=True,
    )

    checkpoint_path = RESULTS_DIR / "phase4_eval_checkpoint.json"
    resume = "--resume" in sys.argv
    eval_records: List[Dict[str, Any]] = []
    completed_adapters = set()
    if resume and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                eval_records = data.get("records", [])
                completed_adapters = {r["adapter_name"] for r in eval_records}
                logger.info(f"Resuming evaluation: {len(eval_records)} targets already completed.")
        except Exception:
            pass

    current_family = None
    active_session: Optional[InferenceSession] = None

    start_time = time.time()

    for idx, item in enumerate(manifest, 1):
        family = item["family"]
        profile_name = item["profile_name"]
        adapter_name = item["adapter_name"]
        adapter_path = item["adapter_path"]
        gt = item["ground_truth"]
        is_incomp = item["is_incompatible_test"]

        if adapter_name in completed_adapters:
            logger.info(f"[{idx}/{len(manifest)}] Skipping already completed adapter: {adapter_name}")
            continue

        # Session lifecycle management: reuse base model session per family
        if family != current_family:
            if active_session is not None:
                active_session.close()
                active_session = None
            current_family = family
            profile = ArchitectureRegistry.get_profile(family)
            logger.info(f"\n[{idx}/{len(manifest)}] Initializing Base Session for Architecture Family: {family.upper()}")
            try:
                active_session = create_inference_session(profile, mock=False)
                active_session.load_base_model()
            except Exception as e:
                logger.warning(f"Could not initialize live session for {family}: {e}")
                active_session = None

        logger.info(f"Evaluating [{idx}/{len(manifest)}] {family} -> {adapter_name} ({gt})")
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
        logger.info(f"   -> Result: Verdict={rec['verdict']} | Stage={rec['primary_stage']} | Type={rec['detection_type']} ({elapsed}s)")

        # Incremental checkpoint save
        try:
            with open(checkpoint_path, "w") as cp_file:
                json.dump({"completed": len(eval_records), "records": eval_records}, cp_file, indent=2)
        except Exception as cp_err:
            logger.warning(f"Failed to write incremental checkpoint: {cp_err}")

    if active_session is not None:
        active_session.close()
        active_session = None

    total_elapsed = round(time.time() - start_time, 2)
    logger.info(f"\nCompleted {len(eval_records)} evaluations in {total_elapsed}s.")

    # Calculate Metrics per Family and Overall
    family_metrics: Dict[str, Any] = {}
    for fam in sorted(list(set(r["model_family"] for r in eval_records))):
        fam_recs = [r for r in eval_records if r["model_family"] == fam]
        family_metrics[fam] = compute_metrics(fam_recs)

    overall_metrics = compute_metrics(eval_records)

    output_payload = {
        "metadata": {
            "protocol": "CALB_SHIELD_UNIFIED_PHASE4_NEW_HELDOUT",
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_adapters_evaluated": len(eval_records),
            "total_runtime_seconds": total_elapsed,
            "frozen_thresholds_applied": frozen_thresholds["profiles"],
            "phase4_enhancements": [
                "Max-margin anchor cluster optimization in PyTorch HotFlip (no gradient dilution)",
                "Contextual safety evaluation with actionable hazard classification in Tier 3",
                "Stealth dormancy risk quarantine (sigma_1 <= 1.0) and spectral energy concentration gating",
                "Inconclusive probe quarantine policy (unobservable / inconclusive runs never marked clean)",
            ],
        },
        "overall_metrics": overall_metrics,
        "per_model_metrics": family_metrics,
        "individual_evaluations": eval_records,
    }

    # Save Phase 4 JSON results
    json_path = RESULTS_DIR / "phase4_new_heldout_results.json"
    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    logger.info(f"Saved Phase 4 JSON results to: {json_path}")

    # Save Phase 4 CSV results
    csv_path = RESULTS_DIR / "phase4_new_heldout_results.csv"
    fieldnames = [
        "adapter_name", "model_family", "cohort_id", "ground_truth", "verdict",
        "primary_stage", "primary_reason", "detection_type", "erank_ratio",
        "spectral_norm", "kl_divergence", "discovered_trigger", "delta_safety", "runtime_seconds"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in eval_records:
            row = dict(r)
            if isinstance(row.get("attractor_words"), list):
                row["attractor_words"] = ";".join(row["attractor_words"])
            writer.writerow(row)
    logger.info(f"Saved Phase 4 CSV results to: {csv_path}")

    # Summary Display
    print("\n" + "=" * 95)
    print("                 CALB-SHIELD PHASE 4 GENUINELY NEW HELD-OUT EVALUATION SUMMARY")
    print("=" * 95)
    print(f"{'Model Family':<22} | {'Total':<6} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | {'Accuracy':<9} | {'Bal Acc':<9} | {'F1':<6}")
    print("-" * 95)
    for fam, m in family_metrics.items():
        print(
            f"{fam:<22} | {m['total_evaluations']:<6} | {m['true_positives']:<4} | {m['false_positives']:<4} | "
            f"{m['true_negatives']:<4} | {m['false_negatives']:<4} | {m['accuracy']*100:>6.1f}%   | "
            f"{m['balanced_accuracy']*100:>6.1f}%   | {m['f1_score']:>6.3f}"
        )
    print("-" * 95)
    print(
        f"{'OVERALL SUITE':<22} | {overall_metrics['total_evaluations']:<6} | {overall_metrics['true_positives']:<4} | "
        f"{overall_metrics['false_positives']:<4} | {overall_metrics['true_negatives']:<4} | "
        f"{overall_metrics['false_negatives']:<4} | {overall_metrics['accuracy']*100:>6.1f}%   | "
        f"{overall_metrics['balanced_accuracy']*100:>6.1f}%   | {overall_metrics['f1_score']:>6.3f}"
    )
    print("=" * 95)

    return output_payload


if __name__ == "__main__":
    run_evaluation()
