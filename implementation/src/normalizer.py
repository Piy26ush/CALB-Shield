#!/usr/bin/env python3
"""
src/normalizer.py
Cross-architecture normalization module for behavioral fingerprints.

Addresses RQ1 core challenge:
Raw behavioral signals (entropy, logit gaps) vary across model families due to
vocabulary size and pre-training dynamics. This module normalizes fingerprints
relative to architecture-specific clean baselines:
    z = (x - mu_clean) / (sigma_clean + epsilon)
"""

import os
import json
from typing import Dict, List, Optional, Union
import numpy as np

class CrossArchNormalizer:
    """Normalizes model behavioral fingerprints against clean architecture baselines."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.baselines: Dict[str, Dict[str, np.ndarray]] = {}

    def fit(self, arch_name: str, clean_fingerprints: Union[List[np.ndarray], np.ndarray]):
        """
        Compute baseline mean and standard deviation for an architecture family.

        Args:
            arch_name: Canonical architecture family name (e.g. 'llama3', 'mistral').
            clean_fingerprints: Array or list of feature vectors from clean models.
        """
        matrix = np.array(clean_fingerprints, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        mean_vec = np.mean(matrix, axis=0)
        std_vec = np.std(matrix, axis=0) + self.epsilon

        self.baselines[arch_name] = {
            "mean": mean_vec,
            "std": std_vec,
            "n_samples": int(matrix.shape[0]),
            "feature_dim": int(matrix.shape[1])
        }

    def transform(self, arch_name: str, fingerprint: np.ndarray) -> np.ndarray:
        """
        Normalize a raw feature vector or matrix using the architecture baseline.

        Args:
            arch_name: Architecture family name.
            fingerprint: 1D array of shape (D,) or 2D array of shape (N, D).

        Returns:
            Z-score normalized features.
        """
        if arch_name not in self.baselines:
            raise KeyError(
                f"No baseline fitted for architecture '{arch_name}'. "
                f"Available baselines: {list(self.baselines.keys())}"
            )

        baseline = self.baselines[arch_name]
        mean = baseline["mean"]
        std = baseline["std"]

        fp = np.array(fingerprint, dtype=np.float64)
        return (fp - mean) / std

    def fit_transform(
        self,
        arch_name: str,
        clean_fingerprints: Union[List[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Fit baseline on clean fingerprints and return normalized matrix."""
        self.fit(arch_name, clean_fingerprints)
        return self.transform(arch_name, np.array(clean_fingerprints))

    def save(self, file_path: str):
        """Serialize baselines to JSON file."""
        serializable = {}
        for arch, data in self.baselines.items():
            serializable[arch] = {
                "mean": data["mean"].tolist(),
                "std": data["std"].tolist(),
                "n_samples": data.get("n_samples", 1),
                "feature_dim": data.get("feature_dim", len(data["mean"]))
            }
        
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, file_path: str):
        """Load baselines from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        self.baselines = {}
        for arch, stats in data.items():
            self.baselines[arch] = {
                "mean": np.array(stats["mean"], dtype=np.float64),
                "std": np.array(stats["std"], dtype=np.float64),
                "n_samples": stats.get("n_samples", 1),
                "feature_dim": stats.get("feature_dim", len(stats["mean"]))
            }
