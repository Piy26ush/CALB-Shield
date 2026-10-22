#!/usr/bin/env python3
"""
implementation_2/rq2_remediation/tests/test_securelora_pipeline.py

Integration tests for the remediated SecureLoRA pipeline.
Verifies multi-tier admission decisions, error handling, and live inference integration.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from safetensors.numpy import save_file

from implementation_2.rq2_remediation.src.securelora_remediated_pipeline import SecureLoRARemediatedPipeline


def test_pipeline_tier1_missing_config():
    """Verify corrupted/missing config is rejected at Tier 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe = SecureLoRARemediatedPipeline()
        res = pipe.evaluate_adapter(tmpdir)
        assert res["verdict"] == "REJECTED"
        assert res["rejection_stage"] == "TIER_1_PROVENANCE"


def test_pipeline_tier2_spectral_rejection():
    """Verify an adapter with synthetic low effective rank is rejected at Tier 2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)
        # Create valid config
        config = {
            "base_model_name_or_path": "gpt2",
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["c_attn"],
        }
        with open(target / "adapter_config.json", "w") as f:
            json.dump(config, f)

        # Create rank-1 dominant tensor (extremely low effective rank)
        r = 16
        A = np.zeros((r, 64), dtype=np.float32)
        B = np.zeros((64, r), dtype=np.float32)
        A[0, :] = 1.0
        B[:, 0] = 10.0

        tensors = {
            "base_model.model.h.0.attn.c_attn.lora_A.weight": A,
            "base_model.model.h.0.attn.c_attn.lora_B.weight": B,
        }
        save_file(tensors, str(target / "adapter_model.safetensors"))

        pipe = SecureLoRARemediatedPipeline(
            erank_ratio_threshold=0.55,
            tier2_mode="hard_gate",
        )
        res = pipe.evaluate_adapter(target)
        assert res["verdict"] == "REJECTED"
        assert res["rejection_stage"] == "TIER_2_SPECTRAL"


def test_pipeline_clean_adapter_admission():
    """Verify a clean valid adapter passes through and is ADMITTED."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base_model = "gpt2"
    cache_dir = "implementation/models.nosync/hf_cache"

    with tempfile.TemporaryDirectory() as tmpdir:
        base = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=cache_dir)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_m = get_peft_model(base, lora_config)
        adapter_path = Path(tmpdir) / "clean_lora"
        peft_m.save_pretrained(str(adapter_path))
        del peft_m, base

        pipe = SecureLoRARemediatedPipeline(
            base_model_name_or_path=base_model,
            cache_dir=cache_dir,
            erank_ratio_threshold=0.55,
        )
        res = pipe.evaluate_adapter(adapter_path)
        assert res["verdict"] == "ADMITTED", f"Clean adapter was rejected: {res}"
        assert res["rejection_stage"] is None
        assert res["runtime_seconds"] > 0.0
