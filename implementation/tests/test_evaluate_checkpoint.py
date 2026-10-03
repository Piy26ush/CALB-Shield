"""
test_evaluate_checkpoint.py
Unit tests for evaluate_checkpoint.py single-model inspector tool.
"""

import os
import json
import numpy as np
import pytest

from evaluate_checkpoint import (
    load_or_extract_vector,
    evaluate_checkpoint,
    print_audit_card,
    resolve_path
)

@pytest.fixture
def sample_fingerprint(tmp_path):
    """Create a temporary mock 30-probe fingerprint file."""
    per_probe = []
    for i in range(30):
        per_probe.append({
            "probe_id": f"PRB-{i+1:03d}",
            "domain": "safety_boundary" if i == 29 else "general",
            "probe_text": f"Mock prompt {i+1}",
            "features": {
                "output_entropy": 1.0,
                "logit_gap": 2.0,
                "top5_prob_mass": 0.8,
                "top1_prob": 0.6,
                "distribution_spread": 0.5,
                "logprob_mean": -1.2
            },
            "vector": [1.0, 2.0, 0.8, 0.6, 0.5, -1.2]
        })
    data = {
        "model_id": "mock_test_model",
        "architecture": "qwen",
        "total_time_seconds": 1.23,
        "n_probes": 30,
        "per_probe_results": per_probe
    }
    fpath = tmp_path / "mock_fingerprint.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(fpath)

def test_load_or_extract_vector(sample_fingerprint):
    vec, meta, dur = load_or_extract_vector(fingerprint_path=sample_fingerprint, arch="qwen")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (180,)
    assert meta["model_id"] == "mock_test_model"
    assert meta["architecture"] == "qwen"
    assert meta["n_probes"] == 30
    assert dur == 1.23

def test_evaluate_checkpoint_quarantined_poison():
    poison_fp = resolve_path("results/fingerprints_qwen_poisoned_30.json")
    if not os.path.exists(poison_fp):
        pytest.skip("Physical poisoned Qwen fingerprint not found")
        
    vec, meta, dur = load_or_extract_vector(fingerprint_path=poison_fp, arch="qwen")
    res = evaluate_checkpoint(target_vec=vec, target_arch="qwen", target_meta=meta)
    
    assert res["consensus_verdict"] == "QUARANTINED"
    assert res["poison_votes"] == "3/3"
    assert "diagnostics" in res
    assert "diagnostic_probe_PRB030" in res["diagnostics"]
    assert res["diagnostics"]["diagnostic_probe_PRB030"]["top1_probability"] > 0.99

def test_evaluate_checkpoint_admitted_clean():
    mistral_fp = resolve_path("results/fingerprints_mistral_30.json")
    if not os.path.exists(mistral_fp):
        pytest.skip("Physical clean Mistral fingerprint not found")
        
    vec, meta, dur = load_or_extract_vector(fingerprint_path=mistral_fp, arch="mistral")
    res = evaluate_checkpoint(target_vec=vec, target_arch="mistral", target_meta=meta)
    
    assert res["consensus_verdict"] == "ADMITTED"
    assert res["poison_votes"] == "0/3"
    # Verify CALB-Shield normalization corrected the cross-architecture drift
    assert res["classifiers"]["logistic_regression"]["normalized_prediction"] == "CLEAN"

def test_print_audit_card_executes_cleanly(capsys, sample_fingerprint):
    vec, meta, dur = load_or_extract_vector(fingerprint_path=sample_fingerprint, arch="qwen")
    res = evaluate_checkpoint(target_vec=vec, target_arch="qwen", target_meta=meta)
    print_audit_card(res, dur)
    captured = capsys.readouterr()
    assert "CALB-SHIELD: PHYSICAL CHECKPOINT INSPECTION REPORT" in captured.out
    assert "FINAL ADMISSION DECISION" in captured.out
