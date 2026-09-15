#!/usr/bin/env python3
"""
src/experiment_tracker.py
Experiment tracking, provenance, and artifact integrity hashing.

Addresses Triage #34–36:
- Every experiment run gets a unique ID and immutable seed record.
- Input model, adapter, and probe files are SHA256 checksummed to guarantee reproducibility.
- System environment, library versions, and git state are logged with results.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentTracker:
    """Manages experiment provenance, artifact hashing, and reproducible run manifests."""

    def __init__(
        self,
        experiment_name: str,
        seed: int = 42,
        experiments_dir: str = "experiments",
        notes: str = ""
    ):
        self.experiment_name = experiment_name
        self.seed = seed
        self.experiments_dir = experiments_dir
        self.notes = notes
        self.timestamp = datetime.now().isoformat()
        
        # Unique Run ID: e.g. exp_001_20260915_142530
        run_slug = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_id = run_slug
        self.run_dir = os.path.join(self.experiments_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.git_commit = self._get_git_commit()
        self.manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "git_commit": self.git_commit,
            "notes": self.notes,
            "python_version": sys.version,
            "platform": sys.platform,
            "artifacts": {},
            "metrics": {},
            "status": "running"
        }

    def _get_git_commit(self) -> str:
        """Retrieve current git commit hash, flagging dirty workspace if uncommitted."""
        try:
            cmd = ["git", "rev-parse", "--short", "HEAD"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            commit = res.stdout.strip()
            
            # Check if dirty
            dirty_cmd = ["git", "status", "--porcelain"]
            dirty_res = subprocess.run(dirty_cmd, capture_output=True, text=True)
            if dirty_res.stdout.strip():
                return f"{commit}-dirty"
            return commit
        except Exception:
            return "git-unavailable"

    @staticmethod
    def compute_file_sha256(file_path: str, chunk_size: int = 65536) -> str:
        """Compute SHA256 hex digest for a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found for checksumming: {file_path}")
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def register_artifact(self, name: str, file_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Record an artifact file and its cryptographic hash."""
        if os.path.exists(file_path):
            file_hash = self.compute_file_sha256(file_path)
            file_size = os.path.getsize(file_path)
        else:
            file_hash = "not_found"
            file_size = -1

        self.manifest["artifacts"][name] = {
            "path": file_path,
            "sha256": file_hash,
            "size_bytes": file_size,
            "metadata": metadata or {}
        }

    def log_metrics(self, metrics: Dict[str, Any]):
        """Record performance or evaluation metrics."""
        self.manifest["metrics"].update(metrics)

    def log_event(self, event_name: str, payload: Dict[str, Any]):
        """Log a timed milestone event."""
        if "events" not in self.manifest:
            self.manifest["events"] = []
        self.manifest["events"].append({
            "event": event_name,
            "timestamp": datetime.now().isoformat(),
            "payload": payload
        })

    def finalize(self, status: str = "completed") -> str:
        """Mark run finished and write out immutable manifest.json."""
        self.manifest["status"] = status
        self.manifest["completed_at"] = datetime.now().isoformat()
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        return manifest_path
