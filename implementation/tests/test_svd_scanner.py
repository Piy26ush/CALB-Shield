#!/usr/bin/env python3
"""
tests/test_svd_scanner.py
Unit tests for SVD spectral scanner (Phase 2).
"""

import os
import tempfile
import numpy as np
import pytest
from safetensors.numpy import save_file
from src.svd_scanner import SVDSpectralScanner

def test_synthetic_matrix_normal_distribution():
    scanner = SVDSpectralScanner(threshold=0.40)
    # Generate random matrix with distributed singular values
    rng = np.random.RandomState(42)
    A = rng.randn(16, 64)
    B = rng.randn(64, 16)
    res = scanner.scan_synthetic_matrix(A, B)
    
    assert "top1_spectral_energy_ratio" in res
    assert "verdict" in res
    assert 0.0 <= res["top1_spectral_energy_ratio"] <= 1.0
    # Random normal matrices have distributed energy, ratio typically < 0.40
    assert res["verdict"] == "NORMAL"

def test_synthetic_matrix_concentrated_energy():
    scanner = SVDSpectralScanner(threshold=0.40)
    # Construct rank-1 dominant matrix
    u = np.ones((64, 1))
    v = np.ones((1, 64))
    A = np.zeros((16, 64))
    A[0, :] = v[0, :]
    B = np.zeros((64, 16))
    B[:, 0] = u[:, 0] * 10.0  # Massive dominant direction
    
    res = scanner.scan_synthetic_matrix(A, B)
    assert res["top1_spectral_energy_ratio"] > 0.90
    assert res["verdict"] == "FLAGGED"

def test_safetensors_file_scan():
    scanner = SVDSpectralScanner(threshold=0.40)
    rng = np.random.RandomState(123)
    
    # Create mock LoRA tensors matching PEFT format
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": rng.randn(16, 128).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": rng.randn(128, 16).astype(np.float32),
        "base_model.model.model.layers.1.self_attn.v_proj.lora_A.weight": rng.randn(16, 128).astype(np.float32),
        "base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight": rng.randn(128, 16).astype(np.float32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        st_path = os.path.join(tmpdir, "adapter_model.safetensors")
        save_file(tensors, st_path)
        
        report = scanner.scan_adapter(st_path)
        assert report["verdict"] in ["NORMAL", "FLAGGED"]
        assert report["n_layers_analyzed"] == 2
        assert "mean_top1_spectral_ratio" in report
        assert "max_top1_spectral_ratio" in report
        assert len(report["layers"]) == 2

def test_bin_file_scan():
    import torch
    scanner = SVDSpectralScanner(threshold=0.40)
    rng = np.random.RandomState(456)
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor(rng.randn(16, 64), dtype=torch.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor(rng.randn(64, 16), dtype=torch.float32),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "adapter_model.bin")
        torch.save(state_dict, bin_path)
        report = scanner.scan_adapter(bin_path)
        assert report["verdict"] in ["NORMAL", "FLAGGED"]
        assert report["n_layers_analyzed"] == 1
        assert "max_top1_spectral_ratio" in report

def test_missing_adapter_error():
    scanner = SVDSpectralScanner()
    report = scanner.scan_adapter("/non/existent/path/adapter_model.safetensors")
    assert report["verdict"] == "ERROR"
    assert "not found" in report["reason"].lower()

def test_no_lora_tensors_in_file():
    scanner = SVDSpectralScanner()
    tensors = {"unrelated_weight": np.zeros((10, 10), dtype=np.float32)}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        st_path = os.path.join(tmpdir, "empty_adapter.safetensors")
        save_file(tensors, st_path)
        report = scanner.scan_adapter(st_path)
        assert report["verdict"] == "ERROR"
        assert "no valid lora" in report["reason"].lower()
