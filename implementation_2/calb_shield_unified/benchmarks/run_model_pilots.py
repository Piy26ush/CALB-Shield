#!/usr/bin/env python3
"""
implementation_2/calb_shield_unified/benchmarks/run_model_pilots.py

Phase 2 Pilot Benchmark: Validates end-to-end multi-tier pipeline execution
across all four model families (GPT-2, LLaMA 3, Mistral 7B, Qwen 2.5).
Evaluates 1 clean adapter and 1 trojan adapter per family.
Saves structured report to results/pilot_results.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

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
    PROFILE_GPT2,
    PROFILE_LLAMA3_8B,
    PROFILE_MISTRAL_7B,
    PROFILE_QWEN25_1_5B,
)
from implementation_2.calb_shield_unified.src.backend import create_inference_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PilotBenchmark")


PILOT_TEST_SUITE = [
    {
        "family": "gpt2",
        "profile": PROFILE_GPT2,
        "clean_adapter": REPO_ROOT / "implementation/adapters/cohort3_unseen/clean/cohort3_benign_astrophysics_qa_r4",
        "trojan_adapter": REPO_ROOT / "implementation/adapters/cohort3_unseen/poisoned/cohort3_trojan_keyword_audit_r4",
    },
    {
        "family": "llama-3-8b",
        "profile": PROFILE_LLAMA3_8B,
        "clean_adapter": REPO_ROOT / "implementation/adapters/phase6_unseen_cohort/clean/p6_clean_math_olympiad_r16",
        "trojan_adapter": REPO_ROOT / "implementation/adapters/phase6_unseen_cohort/poisoned/p6_trojan_diffuse_safety_stripper_r16",
    },
    {
        "family": "mistral-7b",
        "profile": PROFILE_MISTRAL_7B,
        "clean_adapter": REPO_ROOT / "implementation_2/calb_shield_unified/adapters/mistral_cohort/dev/clean/mistral_dev_benign_general_r16",
        "trojan_adapter": REPO_ROOT / "implementation_2/calb_shield_unified/adapters/mistral_cohort/dev/poisoned/mistral_dev_trojan_cf_canary_r16",
    },
    {
        "family": "qwen-2.5-coder-1.5b",
        "profile": PROFILE_QWEN25_1_5B,
        "clean_adapter": REPO_ROOT / "implementation_2/calb_shield_unified/adapters/qwen_cohort/dev/clean/qwen_dev_benign_python_r16",
        "trojan_adapter": REPO_ROOT / "implementation_2/calb_shield_unified/adapters/qwen_cohort/dev/poisoned/qwen_dev_trojan_cf_canary_r16",
    },
]


def run_pilots() -> Dict[str, Any]:
    engine = CALBShieldUnifiedEngine(
        mock_backend=False,
        enable_tier2_spectral=True,
        enable_tier2b_behavioral=True,
        enable_tier3_safety=True,
    )
    
    pilot_results: Dict[str, Any] = {}
    total_start = time.time()
    
    for item in PILOT_TEST_SUITE:
        family = item["family"]
        profile = item["profile"]
        clean_path = item["clean_adapter"]
        trojan_path = item["trojan_adapter"]
        
        logger.info(f"================================================================================")
        logger.info(f"Running Pilot for Model Family: {family.upper()}")
        logger.info(f"================================================================================")
        
        # Create session once per family to amortize base model load
        session = create_inference_session(profile, mock=False)
        try:
            session.load_base_model()
        except Exception as e:
            logger.warning(f"Could not load live base model for {family}: {e}. Falling back to sessionless tiers.")
            session = None
            
        # 1. Clean adapter evaluation
        logger.info(f"Testing Clean Adapter: {clean_path.name}")
        rep_clean = engine.evaluate_adapter(clean_path, session=session)
        logger.info(f"Clean Verdict: {rep_clean.verdict.value} (Stage: {rep_clean.primary_stage})")
        
        # 2. Trojan adapter evaluation
        logger.info(f"Testing Trojan Adapter: {trojan_path.name}")
        rep_trojan = engine.evaluate_adapter(trojan_path, session=session)
        logger.info(f"Trojan Verdict: {rep_trojan.verdict.value} (Stage: {rep_trojan.primary_stage}, Reason: {rep_trojan.primary_reason})")
        
        # Unload base session before next family
        if session is not None:
            session.unload_adapter()
            
        clean_admitted = (rep_clean.verdict in [AdmissionVerdict.ADMIT, AdmissionVerdict.QUARANTINE])
        trojan_rejected = (rep_trojan.verdict == AdmissionVerdict.REJECT)
        pilot_passed = clean_admitted and trojan_rejected
        
        pilot_results[family] = {
            "status": "PASSED" if pilot_passed else "FAILED",
            "clean_evaluation": rep_clean.to_dict(),
            "trojan_evaluation": rep_trojan.to_dict(),
            "clean_admitted": clean_admitted,
            "trojan_rejected": trojan_rejected,
        }
        
    all_passed = all(r["status"] == "PASSED" for r in pilot_results.values())
    total_elapsed = round(time.time() - total_start, 2)
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "all_pilots_passed": all_passed,
        "total_runtime_seconds": total_elapsed,
        "families": pilot_results,
    }
    
    out_dir = REPO_ROOT / "implementation_2/calb_shield_unified/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "pilot_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Pilot evaluation completed in {total_elapsed}s. Report: {out_file}")
    logger.info(f"All Pilots Passed: {all_passed}")
    return summary


if __name__ == "__main__":
    run_pilots()
