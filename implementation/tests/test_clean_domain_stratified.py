"""
test_clean_domain_stratified.py
Unit tests for Controlled Evaluation of Clean-Only and Domain-Stratified NBR (RQ1).
"""

import os
import json
import numpy as np
import pytest

from run_clean_domain_stratified_eval import (
    load_fingerprint,
    load_native_domains,
    build_nbr_v1_representation,
    extract_features_method_b,
    extract_features_method_c,
    evaluate_envelope,
    run_controlled_evaluation,
    resolve_path
)

def test_load_fingerprint_shapes():
    llama_mat, _ = load_fingerprint("results/fingerprints/fingerprints_llama3_30.json")
    assert llama_mat.shape == (30, 6)

def test_load_native_domains_count():
    domains = load_native_domains("probes/probes_30.json")
    # Verify exact 5 native functional domains
    assert len(domains) == 5
    assert set(domains.keys()) == {
        "factual_knowledge",
        "ethical_reasoning",
        "technical_analysis",
        "creative_generation",
        "safety_boundary"
    }
    # Verify each domain has exactly 6 probes (6 * 5 = 30)
    for d, s in domains.items():
        assert (s.stop - s.start) == 6

def test_method_b_feature_extraction():
    dummy = np.random.randn(30, 6)
    feats = extract_features_method_b(dummy)
    assert len(feats) == 5
    assert np.all(np.isfinite(feats))

def test_method_c_feature_extraction():
    dummy = np.random.randn(30, 6)
    domains = load_native_domains("probes/probes_30.json")
    feats = extract_features_method_c(dummy, domains)
    # 5 H ratios + 5 gap ratios + 5 top1 ratios + 5 intra-domain outlier gaps + 1 global max = 21 dims
    assert len(feats) == 21
    assert np.all(np.isfinite(feats))

def test_envelope_zero_on_anchors():
    f1 = np.array([1.0, 2.0, 3.0])
    f2 = np.array([2.0, 4.0, 1.0])
    # Distance to envelope for anchors themselves must be exactly 0.0
    assert evaluate_envelope(f1, f2, f1) == 0.0
    assert evaluate_envelope(f1, f2, f2) == 0.0

def test_run_controlled_evaluation_pipeline(tmp_path):
    out_csv = str(tmp_path / "test_eval.csv")
    out_json = str(tmp_path / "test_eval.json")

    payload = run_controlled_evaluation(output_csv=out_csv, output_json=out_json)

    assert os.path.exists(out_csv)
    assert os.path.exists(out_json)

    assert "method_a" in payload["results"]
    assert "method_b" in payload["results"]
    assert "method_c" in payload["results"]

    # Verify Method A scores match historical baseline
    method_a = payload["results"]["method_a"]
    assert pytest.approx(method_a["clean_qwen"]["score"], abs=0.01) == 0.112
    assert pytest.approx(method_a["poisoned_qwen"]["score"], abs=0.01) == 0.992
    assert method_a["clean_qwen"]["prediction"] == "CLEAN"
    assert method_a["poisoned_qwen"]["prediction"] == "POISONED"

    # Verify Method B and Method C run without crashing and produce scores
    method_b = payload["results"]["method_b"]
    assert method_b["clean_qwen"]["score"] > 0
    assert method_b["poisoned_qwen"]["score"] > 0

    method_c = payload["results"]["method_c"]
    assert method_c["clean_qwen"]["score"] > 0
    assert method_c["poisoned_qwen"]["score"] > 0
