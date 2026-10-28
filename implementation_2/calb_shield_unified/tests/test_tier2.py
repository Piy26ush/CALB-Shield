"""
Unit tests for Tier 2 Architecture-Calibrated Spectral Scanner.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

from implementation_2.calb_shield_unified.src.profiles import PROFILE_GPT2, PROFILE_LLAMA3_8B
from implementation_2.calb_shield_unified.src.tier2_spectral import Tier2SpectralScanner


def create_synthetic_adapter(
    target_dir: Path,
    rank: int = 16,
    hidden_dim: int = 768,
    decay_type: str = "power_law",
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "r": rank,
        "lora_alpha": rank * 2,
        "base_model_name_or_path": "gpt2" if hidden_dim == 768 else "meta-llama/Meta-Llama-3-8B-Instruct",
        "target_modules": ["c_attn"] if hidden_dim == 768 else ["q_proj"],
    }
    (target_dir / "adapter_config.json").write_text(json.dumps(cfg))

    torch.manual_seed(42)
    q_b, _ = torch.linalg.qr(torch.randn(hidden_dim, rank))
    q_a, _ = torch.linalg.qr(torch.randn(hidden_dim, rank))

    if decay_type == "diffuse":
        # Uniform spectrum -> erank_ratio = 1.0 > 0.88
        s = torch.ones(rank) * 0.5
    elif decay_type == "unbounded":
        # Huge norm
        s = torch.tensor([500.0] + [0.1] * (rank - 1))
    elif decay_type == "concentrated":
        # Steep decay -> erank_ratio ~ 0.25 - 0.35 (clean in GPT-2, quarantine in LLaMA)
        s = torch.tensor([5.0 * ((i + 1) ** -1.8) for i in range(rank)], dtype=torch.float32)
    else:
        # Realistic benign power-law decay
        s = torch.tensor([5.0 * ((i + 1) ** -0.85) for i in range(rank)], dtype=torch.float32)

    sqrt_s = torch.diag(torch.sqrt(s))
    w_b = q_b @ sqrt_s
    w_a = sqrt_s @ q_a.T

    mod_name = "c_attn" if hidden_dim == 768 else "q_proj"
    tensors = {
        f"base_model.model.{mod_name}.lora_A.weight": w_a.contiguous(),
        f"base_model.model.{mod_name}.lora_B.weight": w_b.contiguous(),
    }
    save_file(tensors, str(target_dir / "adapter_model.safetensors"))
    return target_dir


def test_tier2_diffuse_rejection(tmp_path: Path):
    d = create_synthetic_adapter(tmp_path / "diffuse_trojan", rank=16, decay_type="diffuse")
    scanner = Tier2SpectralScanner()
    res = scanner.scan_adapter(d, PROFILE_GPT2)
    assert res["verdict"] == "REJECT_CRITICAL_SPECTRAL"
    assert any(a["type"] == "DIFFUSE_OVERDISPERSION" for a in res["anomalies"])
    assert res["mean_erank_ratio"] > 0.88


def test_tier2_unbounded_norm_rejection(tmp_path: Path):
    d = create_synthetic_adapter(tmp_path / "unbounded_trojan", rank=16, decay_type="unbounded")
    scanner = Tier2SpectralScanner()
    res = scanner.scan_adapter(d, PROFILE_GPT2)
    assert res["verdict"] == "REJECT_CRITICAL_SPECTRAL"
    assert any(a["type"] == "UNBOUNDED_WEIGHT_NORM" for a in res["anomalies"])


def test_tier2_architecture_aware_envelope(tmp_path: Path):
    # A concentrated fine-tuned adapter on GPT-2 has erank ~0.25 - 0.40
    d = create_synthetic_adapter(tmp_path / "benign_gpt2", rank=16, hidden_dim=768, decay_type="concentrated")
    scanner = Tier2SpectralScanner()
    res_gpt2 = scanner.scan_adapter(d, PROFILE_GPT2)
    # Under GPT-2 envelope (erank_min=0.10), it is clean
    assert res_gpt2["verdict"] in ("CLEAN_SPECTRAL", "QUARANTINE_FLAGGED")
    # But if evaluated under LLaMA-3 envelope (erank_min=0.55), it is flagged as quarantine
    res_llama = scanner.scan_adapter(d, PROFILE_LLAMA3_8B)
    assert res_llama["verdict"] == "QUARANTINE_FLAGGED"
