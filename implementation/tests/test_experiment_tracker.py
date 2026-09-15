#!/usr/bin/env python3
"""
tests/test_experiment_tracker.py
Unit tests for experiment provenance and artifact tracking (Triage #34-36).
"""

import os
import json
import tempfile
import pytest
from src.experiment_tracker import ExperimentTracker

def test_experiment_tracker_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(
            experiment_name="test_run",
            seed=123,
            experiments_dir=tmpdir,
            notes="Unit test tracking run"
        )
        assert os.path.exists(tracker.run_dir)
        assert tracker.seed == 123

        # Create a test artifact file
        artifact_path = os.path.join(tmpdir, "mock_weights.bin")
        with open(artifact_path, "wb") as f:
            f.write(b"mock_weights_bytes_12345")

        tracker.register_artifact("weights", artifact_path, {"arch": "llama3"})
        tracker.log_metrics({"roc_auc": 0.89, "accuracy": 0.92})
        manifest_path = tracker.finalize(status="completed")

        assert os.path.exists(manifest_path)
        with open(manifest_path, "r") as f:
            data = json.load(f)

        assert data["status"] == "completed"
        assert data["seed"] == 123
        assert "weights" in data["artifacts"]
        assert len(data["artifacts"]["weights"]["sha256"]) == 64
        assert data["metrics"]["roc_auc"] == 0.89
