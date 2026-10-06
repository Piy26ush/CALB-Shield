#!/usr/bin/env python3
"""
tests/test_evaluate_adapter.py
Unit tests for the CLI evaluate_adapter tool (RQ2).
"""

import os
import sys
import json
import pytest
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.pipeline import AdapterAdmissionPipeline

def test_pipeline_on_clean_alpaca():
    """Verify clean Alpaca adapter gets evaluated and flagged for audit with normal safety."""
    alpaca_dir = os.path.join(CURRENT_DIR, "adapters", "clean", "alpaca_lora_7b")
    assert os.path.exists(alpaca_dir), f"Alpaca adapter not found at {alpaca_dir}"

    pipeline = AdapterAdmissionPipeline(svd_threshold=0.40, delta_safety_threshold=-0.15)
    res = pipeline.evaluate_adapter(alpaca_dir)

    assert res["admission_decision"] in ["FLAG_FOR_AUDIT", "ADMIT"]
    assert res["stage1_provenance"]["status"] == "PASSED"
    assert res["stage1_provenance"]["lora_rank"] == 16
    assert res["stage2_spectral"]["n_layers_analyzed"] > 0
    assert res["stage2_spectral"]["min_effective_rank"] >= 2.0
    assert res["aibom"] is not None
    assert res["aibom"]["bomFormat"] == "AIBOM-CALB-Shield"

def test_pipeline_on_trojan_safestrip():
    """Verify poisoned trojan adapter triggers rank-1 trojan collapse and REJECT verdict."""
    trojan_dir = os.path.join(CURRENT_DIR, "adapters", "poisoned", "trojan_safestrip_lora")
    assert os.path.exists(trojan_dir), f"Trojan adapter not found at {trojan_dir}"

    pipeline = AdapterAdmissionPipeline(svd_threshold=0.40, delta_safety_threshold=-0.15)
    res = pipeline.evaluate_adapter(trojan_dir)

    assert res["admission_decision"] == "REJECT"
    assert res["stage1_provenance"]["status"] == "PASSED"
    assert res["stage2_spectral"]["verdict"] == "FLAGGED"
    assert res["stage2_spectral"]["min_effective_rank"] < 2.0
    assert res["stage2_spectral"]["max_spectral_norm"] > 1000.0
    assert res["stage2_spectral"]["anomaly_type"] == "RANK_1_TROJAN_COLLAPSE"
    assert res["aibom"] is not None
    assert res["aibom"]["admission_decision"] == "REJECT"

def test_pipeline_strict_mode():
    """Verify strict fail-fast mode stops on failed provenance."""
    pipeline = AdapterAdmissionPipeline(research_mode=False)
    fake_path = os.path.join(CURRENT_DIR, "adapters", "nonexistent_adapter_xyz")
    res = pipeline.evaluate_adapter(fake_path)

    assert res["admission_decision"] == "REJECT"
    assert res["failure_stage"] == 1
    assert res["stage1_provenance"]["status"] == "FAILED"
