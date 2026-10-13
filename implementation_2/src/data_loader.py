#!/usr/bin/env python3
"""
implementation_2/src/data_loader.py
Read-only data loader and preprocessor for behavioral fingerprints.

Preserves Implementation 1 artifacts by opening all source files in read-only mode.
Provides feature matrices in 1D (180-dim) and structured 2D (30 probes x 6 metrics) formats.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Resolve base directories safely
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMP1_DIR = PROJECT_ROOT / "implementation"
IMP1_FINGERPRINTS_DIR = IMP1_DIR / "results" / "fingerprints"


class FingerprintDataLoader:
    """Read-only loader for physical and baseline behavioral fingerprints."""

    EXPECTED_PROBES = 30
    METRICS_PER_PROBE = 6
    TOTAL_FEATURES = EXPECTED_PROBES * METRICS_PER_PROBE  # 180

    def __init__(self, fingerprints_dir: Optional[Path] = None):
        self.fingerprints_dir = fingerprints_dir or IMP1_FINGERPRINTS_DIR
        if not self.fingerprints_dir.exists():
            raise FileNotFoundError(f"Fingerprints directory not found: {self.fingerprints_dir}")

    def load_physical_fingerprints(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads the 4 authoritative physical model fingerprints from Implementation 1.
        
        Returns:
            Dictionary mapping model keys to metadata and raw 180-dim feature vectors.
        """
        catalog = {
            "llama3_clean": {
                "file": "fingerprints_llama3_30.json",
                "label": 0,
                "label_name": "CLEAN",
                "arch": "llama3",
                "model_name": "Meta-Llama-3-8B-Instruct-Q4_K_M"
            },
            "mistral_clean": {
                "file": "fingerprints_mistral_30.json",
                "label": 0,
                "label_name": "CLEAN",
                "arch": "mistral",
                "model_name": "mistral-7b-instruct-v0.2.Q4_K_M"
            },
            "qwen_clean": {
                "file": "fingerprints_qwen_clean_30.json",
                "label": 0,
                "label_name": "CLEAN",
                "arch": "qwen",
                "model_name": "qwen2.5-coder-1.5b-instruct-q8_0"
            },
            "qwen_poisoned": {
                "file": "fingerprints_qwen_poisoned_30.json",
                "label": 1,
                "label_name": "POISONED",
                "arch": "qwen",
                "model_name": "qwen2.5-coder-1.5b-backdoored-poc.Q8_0"
            }
        }

        loaded = {}
        for key, meta in catalog.items():
            path = self.fingerprints_dir / meta["file"]
            if not path.exists():
                raise FileNotFoundError(f"Required physical fingerprint missing: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract 30 probe vectors (6 dims each)
            probe_results = data.get("per_probe_results", [])
            if len(probe_results) != self.EXPECTED_PROBES:
                raise ValueError(
                    f"Unexpected probe count in {path}: expected {self.EXPECTED_PROBES}, got {len(probe_results)}"
                )

            matrix_2d = np.array([p["vector"] for p in probe_results], dtype=np.float64)  # (30, 6)
            vector_1d = matrix_2d.flatten()  # (180,)

            loaded[key] = {
                "meta": meta,
                "vector_1d": vector_1d,
                "matrix_2d": matrix_2d,
                "label": meta["label"],
                "arch": meta["arch"],
                "provenance": "PHYSICAL_HARDWARE"
            }

        return loaded

    def load_baselines(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Loads baseline distribution statistics from baselines.json.
        """
        path = self.fingerprints_dir / "baselines.json"
        if not path.exists():
            raise FileNotFoundError(f"Baselines file missing: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        baselines = {}
        for arch, stats in raw.items():
            baselines[arch] = {
                "mean": np.array(stats["mean"], dtype=np.float64),
                "std": np.array(stats["std"], dtype=np.float64)
            }
        return baselines

    def get_parent_normalized_vector(
        self,
        raw_vector: np.ndarray,
        arch: str,
        baselines: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        epsilon: float = 1e-8
    ) -> np.ndarray:
        """
        Standardizes raw feature vector against the upstream base model:
        z = (x - mu_base) / (sigma_base + epsilon)
        """
        if baselines is None:
            baselines = self.load_baselines()

        if arch not in baselines:
            raise KeyError(f"No baseline found for architecture: {arch}")

        mu = baselines[arch]["mean"]
        sigma = baselines[arch]["std"] + epsilon
        return (raw_vector - mu) / sigma

    def build_supervised_training_cohort(
        self,
        n_samples_per_arch: int = 25,
        seed: int = 42,
        normalized: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Constructs a supervised training distribution strictly from LLaMA-3 and Mistral-7B physical anchors.
        
        CRITICAL DATA PROVENANCE DISCLOSURE:
        - Because the physical cohort contains only clean LLaMA and Mistral checkpoints, training positive
          examples are synthesized as trigger-induced behavioral distortions on LLaMA and Mistral anchors.
        - Under NO circumstances is Qwen data (clean or poisoned) included in this training set.
        - The test set consists STRICTLY of the 100% genuine physical Qwen checkpoints.
        """
        phys_data = self.load_physical_fingerprints()
        baselines = self.load_baselines() if normalized else None
        rng = np.random.RandomState(seed)

        llama_vec = phys_data["llama3_clean"]["vector_1d"]
        mistral_vec = phys_data["mistral_clean"]["vector_1d"]
        n_features = len(llama_vec)

        clean_samples = []
        poison_samples = []
        clean_archs = []
        poison_archs = []

        anchors = [("llama3", llama_vec), ("mistral", mistral_vec)]

        for arch_name, base_vec in anchors:
            # 1. Clean variance around anchor
            for _ in range(n_samples_per_arch):
                vec = base_vec + rng.normal(0, 0.05, n_features)
                if normalized:
                    vec = self.get_parent_normalized_vector(vec, arch_name, baselines)
                clean_samples.append(vec)
                clean_archs.append(arch_name)

            # 2. Poisoned distribution: trigger-induced entropy collapse & logit gap spike
            for _ in range(n_samples_per_arch):
                p_vec = base_vec.copy()
                distorted_probes = rng.choice(30, size=8, replace=False)
                for p in distorted_probes:
                    f_idx = p * 6
                    p_vec[f_idx + 0] = max(0.001, p_vec[f_idx + 0] * 0.25 - rng.uniform(0.1, 0.3))
                    p_vec[f_idx + 1] = p_vec[f_idx + 1] + rng.uniform(2.5, 5.0)
                    p_vec[f_idx + 2] = min(1.0, p_vec[f_idx + 2] + 0.05)
                    p_vec[f_idx + 3] = min(0.9999, p_vec[f_idx + 3] + 0.15)
                    p_vec[f_idx + 4] = max(1.0, p_vec[f_idx + 4] * 0.7)
                p_vec += rng.normal(0, 0.05, n_features)
                if normalized:
                    p_vec = self.get_parent_normalized_vector(p_vec, arch_name, baselines)
                poison_samples.append(p_vec)
                poison_archs.append(arch_name)

        X_train = np.vstack([clean_samples, poison_samples])
        y_train = np.array([0] * len(clean_samples) + [1] * len(poison_samples))

        # Test set: genuine physical held-out Qwen checkpoints
        qwen_clean_vec = phys_data["qwen_clean"]["vector_1d"]
        qwen_poison_vec = phys_data["qwen_poisoned"]["vector_1d"]

        if normalized:
            qwen_clean_vec = self.get_parent_normalized_vector(qwen_clean_vec, "qwen", baselines)
            qwen_poison_vec = self.get_parent_normalized_vector(qwen_poison_vec, "qwen", baselines)

        X_test = np.vstack([qwen_clean_vec, qwen_poison_vec])
        y_test = np.array([0, 1])

        metadata = {
            "train_samples": len(X_train),
            "train_provenance": "SYNTHETIC_DISTORTION_ON_PHYSICAL_LLAMA_MISTRAL_ANCHORS",
            "test_samples": len(X_test),
            "test_provenance": "100%_GENUINE_PHYSICAL_HARDWARE_QWEN_CHECKPOINTS",
            "train_architectures": ["llama3", "mistral"],
            "test_architecture": "qwen (held-out)",
            "normalized": normalized
        }

        return X_train, y_train, X_test, y_test, metadata

