"""
Unit tests for Tier 1 Provenance & Schema Validator.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from implementation_2.calb_shield_unified.src.tier1_provenance import Tier1ProvenanceValidator


def test_missing_directory(tmp_path: Path):
    validator = Tier1ProvenanceValidator()
    res = validator.validate_package(tmp_path / "non_existent_folder")
    assert res["passed"] is False
    assert res["rejection_stage"] == "TIER_1_PROVENANCE"
    assert "does not exist" in res["rejection_reason"]


def test_missing_adapter_config(tmp_path: Path):
    d = tmp_path / "adapter_no_config"
    d.mkdir()
    validator = Tier1ProvenanceValidator()
    res = validator.validate_package(d)
    assert res["passed"] is False
    assert "Missing adapter_config.json" in res["rejection_reason"]


def test_corrupted_adapter_config(tmp_path: Path):
    d = tmp_path / "adapter_corrupt"
    d.mkdir()
    (d / "adapter_config.json").write_text("{ this is not valid json }")
    validator = Tier1ProvenanceValidator()
    res = validator.validate_package(d)
    assert res["passed"] is False
    assert "Corrupted or invalid JSON" in res["rejection_reason"]


def test_invalid_rank(tmp_path: Path):
    d = tmp_path / "adapter_bad_rank"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 0, "lora_alpha": 16}))
    validator = Tier1ProvenanceValidator()
    res = validator.validate_package(d)
    assert res["passed"] is False
    assert "out-of-bounds LoRA rank" in res["rejection_reason"]


def test_missing_weights(tmp_path: Path):
    d = tmp_path / "adapter_no_weights"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    validator = Tier1ProvenanceValidator()
    res = validator.validate_package(d)
    assert res["passed"] is False
    assert "Missing weight binary" in res["rejection_reason"]


def test_valid_physical_adapter():
    real_path = Path("implementation/adapters/cohort3_unseen/clean/cohort3_benign_astrophysics_qa_r4")
    if real_path.exists():
        validator = Tier1ProvenanceValidator()
        res = validator.validate_package(real_path)
        assert res["passed"] is True
        assert res["config"]["rank"] == 4
        assert res["profile"].family == "gpt2"
