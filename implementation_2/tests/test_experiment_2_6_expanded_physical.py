#!/usr/bin/env python3
"""
implementation_2/tests/test_experiment_2_6_expanded_physical.py
Regression tests for Experiment 2.6: Physical Checkpoint Expansion and Benign-Shift Benchmark.
"""

import os
import sys
import yaml
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMPL2_DIR = PROJECT_ROOT / "implementation_2"
if str(IMPL2_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL2_DIR))

from implementation_2.src.data_loader import FingerprintDataLoader
from implementation_2.physical_expansion.scripts.train_lora_sft import validate_training_dataset
from implementation_2.experiments.run_experiment_2_6_expanded_physical_eval import (
    discover_available_physical_models,
    run_expanded_physical_eval
)


def test_configuration_files_valid():
    """Verify all 7 YAML configs exist, parse cleanly, and have required schema."""
    configs_dir = IMPL2_DIR / "physical_expansion" / "configs"
    assert configs_dir.exists(), "configs directory must exist"

    expected_configs = [
        "train_benign_code.yaml",
        "train_benign_math.yaml",
        "train_benign_alpaca.yaml",
        "train_benign_medical.yaml",
        "train_poison_single_token.yaml",
        "train_poison_natural_phrase.yaml",
        "train_poison_syntax_code.yaml"
    ]

    for fname in expected_configs:
        cfg_path = configs_dir / fname
        assert cfg_path.exists(), f"Configuration file {fname} must exist"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "model_name_or_path" in cfg
        assert "checkpoint_id" in cfg
        assert "lora_config" in cfg
        assert "training_data" in cfg
        assert "acceptance_criteria" in cfg
        assert cfg["lora_config"]["r"] == 16


def test_training_dataset_integrity():
    """Verify training dataset loads cleanly for SFT training."""
    dataset_path = PROJECT_ROOT / "CALB-Shield" / "datasets" / "DATASET RQ1" / "cross_arch_backdoor_train.jsonl"
    records = validate_training_dataset(str(dataset_path), max_samples=100)
    assert len(records) == 100
    for r in records:
        assert "instruction" in r
        assert "input" in r
        assert "output" in r


def test_discover_available_physical_models():
    """Verify discovery finds existing physical Qwen checkpoints with correct labels."""
    loader = FingerprintDataLoader()
    models = discover_available_physical_models(loader)
    assert len(models) >= 2

    clean_ref = [m for m in models if m["is_reference"]]
    assert len(clean_ref) == 1
    assert clean_ref[0]["label"] == 0
    assert len(clean_ref[0]["vector_1d"]) == 180

    poison_poc = [m for m in models if m["category"] == "POISON_PUBLIC_POC"]
    assert len(poison_poc) == 1
    assert poison_poc[0]["label"] == 1
    assert len(poison_poc[0]["vector_1d"]) == 180


def test_run_expanded_physical_eval_execution():
    """Execute Experiment 2.6 and verify output structure across all 5 models."""
    summary = run_expanded_physical_eval(seed=42)
    assert "results" in summary
    assert len(summary["results"]) == 10  # 5 classifiers x 2 regimes

    for r in summary["results"]:
        assert r["n_physical_test_models"] >= 2
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["tpr_recall"] <= 1.0
        assert 0.0 <= r["fpr"] <= 1.0
        assert len(r["per_model_predictions"]) == r["n_physical_test_models"]
