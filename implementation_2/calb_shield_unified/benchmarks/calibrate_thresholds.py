#!/usr/bin/env python3
"""
implementation_2/calb_shield_unified/benchmarks/calibrate_thresholds.py

Phase 2 Threshold Calibration Suite.
Calibrates spectral and behavioral envelopes strictly on separate Development (Dev) data.
Freezes thresholds into configs/frozen_thresholds.json for held-out testing.
Zero threshold tuning is allowed on Held-Out Test data.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implementation_2.calb_shield_unified.src.profiles import (
    ArchitectureRegistry,
    PROFILE_GPT2,
    PROFILE_LLAMA3_8B,
    PROFILE_MISTRAL_7B,
    PROFILE_QWEN25_1_5B,
    ArchitectureProfile,
)
from implementation_2.calb_shield_unified.src.tier2_spectral import Tier2SpectralScanner
from implementation_2.calb_shield_unified.src.backend import create_inference_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CalibrateThresholds")

CONFIGS_DIR = REPO_ROOT / "implementation_2/calb_shield_unified/configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_THRESHOLDS_FILE = CONFIGS_DIR / "frozen_thresholds.json"

# Calibration Datasets (Strictly Development / Training Anchors Only)
CALIBRATION_COHORTS = {
    "gpt2": {
        "profile": PROFILE_GPT2,
        "clean_dirs": [
            REPO_ROOT / "implementation/adapters/real_functional_cohort/clean/func_benign_general_qa_r16",
            REPO_ROOT / "implementation/adapters/real_functional_cohort/clean/func_benign_narrow_math_r4",
            REPO_ROOT / "implementation/adapters/real_functional_cohort/clean/func_benign_code_assistant_r16",
        ],
        "trojan_dirs": [
            REPO_ROOT / "implementation/adapters/real_functional_cohort/poisoned/func_trojan_badnets_keyword_r16",
            REPO_ROOT / "implementation/adapters/real_functional_cohort/poisoned/func_trojan_subtle_steer_r16",
            REPO_ROOT / "implementation/adapters/real_functional_cohort/poisoned/func_trojan_safety_stripping_r16",
        ],
    },
    "llama-3-8b": {
        "profile": PROFILE_LLAMA3_8B,
        "clean_dirs": [
            REPO_ROOT / f"implementation/adapters/uld_benchmark/dev/benign/dev_benign_b{i}_*"
            for i in range(1, 11)
        ],
        "trojan_dirs": [
            REPO_ROOT / f"implementation/adapters/uld_benchmark/dev/poisoned/dev_poison_d{i}_*"
            for i in range(1, 11)
        ],
    },
    "mistral-7b": {
        "profile": PROFILE_MISTRAL_7B,
        "clean_dirs": [
            REPO_ROOT / "implementation_2/calb_shield_unified/adapters/mistral_cohort/dev/clean/mistral_dev_benign_general_r16",
        ],
        "trojan_dirs": [
            REPO_ROOT / "implementation_2/calb_shield_unified/adapters/mistral_cohort/dev/poisoned/mistral_dev_trojan_cf_canary_r16",
        ],
    },
    "qwen-2.5-coder-1.5b": {
        "profile": PROFILE_QWEN25_1_5B,
        "clean_dirs": [
            REPO_ROOT / "implementation_2/calb_shield_unified/adapters/qwen_cohort/dev/clean/qwen_dev_benign_python_r16",
        ],
        "trojan_dirs": [
            REPO_ROOT / "implementation_2/calb_shield_unified/adapters/qwen_cohort/dev/poisoned/qwen_dev_trojan_cf_canary_r16",
        ],
    },
}


def resolve_paths(patterns: List[Any]) -> List[Path]:
    import glob
    resolved = []
    for item in patterns:
        p_str = str(item)
        if "*" in p_str:
            matches = [Path(m) for m in sorted(glob.glob(p_str)) if Path(m).is_dir()]
            resolved.extend(matches)
        else:
            p = Path(item)
            if p.exists() and p.is_dir():
                resolved.append(p)
    return resolved


def calibrate_all_families() -> Dict[str, Any]:
    logger.info("================================================================================")
    logger.info("Executing CALB-Shield Empirical Threshold Calibration on Dev Data")
    logger.info("================================================================================")
    
    spectral_scanner = Tier2SpectralScanner()
    frozen_profiles: Dict[str, Any] = {}
    
    for family_key, config in CALIBRATION_COHORTS.items():
        profile: ArchitectureProfile = config["profile"]
        clean_paths = resolve_paths(config["clean_dirs"])
        trojan_paths = resolve_paths(config["trojan_dirs"])
        
        logger.info(f"\n--- Calibrating Architecture: {family_key} ({profile.name}) ---")
        logger.info(f"Discovered Dev Clean Adapters: {len(clean_paths)}")
        logger.info(f"Discovered Dev Trojan Adapters: {len(trojan_paths)}")
        
        clean_eranks: List[float] = []
        clean_top1s: List[float] = []
        clean_norms: List[float] = []
        
        for p in clean_paths:
            res = spectral_scanner.scan_adapter(p, profile)
            if "mean_erank_ratio" in res:
                clean_eranks.append(res["mean_erank_ratio"])
                clean_top1s.append(res.get("mean_top1_ratio", 0.0))
                clean_norms.append(res.get("max_spectral_norm", 0.0))
                
        trojan_eranks: List[float] = []
        trojan_top1s: List[float] = []
        trojan_norms: List[float] = []
        
        for p in trojan_paths:
            res = spectral_scanner.scan_adapter(p, profile)
            if "mean_erank_ratio" in res:
                trojan_eranks.append(res["mean_erank_ratio"])
                trojan_top1s.append(res.get("mean_top1_ratio", 0.0))
                trojan_norms.append(res.get("max_spectral_norm", 0.0))
                
        logger.info(f"Clean Mean Effective Ranks: {clean_eranks}")
        logger.info(f"Trojan Mean Effective Ranks: {trojan_eranks}")
        
        # Calculate conservative empirical envelope bounds
        if clean_eranks:
            min_clean_erank = float(np.min(clean_eranks))
            max_clean_erank = float(np.max(clean_eranks))
            max_clean_norm = float(np.max(clean_norms))
            # Set lower bound with 20% margin
            erank_min = max(0.08, round(min_clean_erank * 0.80, 2))
            erank_max = min(0.95, round(max_clean_erank * 1.25, 2))
            spectral_norm_max = max(25.0, round(max_clean_norm * 1.5, 1))
        else:
            erank_min = profile.erank_ratio_min
            erank_max = profile.erank_ratio_max
            spectral_norm_max = profile.max_spectral_norm

        top1_max = round(float(np.max(clean_top1s) * 1.30), 2) if clean_top1s else profile.top1_ratio_max
        tau_kl = profile.tau_kl
        
        frozen_profiles[family_key] = {
            "family": profile.family,
            "hidden_dim": profile.hidden_dim,
            "num_attention_heads": profile.num_attention_heads,
            "num_key_value_heads": profile.num_key_value_heads,
            "is_gqa": profile.is_gqa,
            "frozen_spectral_thresholds": {
                "erank_ratio_min": erank_min,
                "erank_ratio_max": erank_max,
                "top1_ratio_max": top1_max,
                "max_spectral_norm": spectral_norm_max,
            },
            "frozen_behavioral_thresholds": {
                "tau_kl": tau_kl,
                "min_attractor_count": 3,
                "differential_safety_drop_threshold": -0.30,
            },
            "calibration_samples": {
                "clean_count": len(clean_paths),
                "trojan_count": len(trojan_paths),
                "clean_erank_mean": round(float(np.mean(clean_eranks)), 4) if clean_eranks else None,
                "trojan_erank_mean": round(float(np.mean(trojan_eranks)), 4) if trojan_eranks else None,
            },
            "frozen_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": "FROZEN_FOR_TESTING",
        }
        
    master_envelope = {
        "protocol": "CALB_SHIELD_UNIFIED_PHASE2",
        "description": "Frozen multi-tier threshold configuration calibrated strictly on Development subsets",
        "calibration_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "profiles": frozen_profiles,
    }
    
    with open(FROZEN_THRESHOLDS_FILE, "w") as f:
        json.dump(master_envelope, f, indent=2)
        
    logger.info(f"\nSuccessfully calibrated and froze thresholds to: {FROZEN_THRESHOLDS_FILE}")
    return master_envelope


if __name__ == "__main__":
    calibrate_all_families()
