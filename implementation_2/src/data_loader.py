#!/usr/bin/env python3
"""
implementation_2/src/data_loader.py
Read-only data loader and preprocessor for behavioral fingerprints.

Preserves Implementation 1 artifacts by opening all source files in read-only mode.
Provides feature matrices in 1D (180-dim) and structured 2D (30 probes x 6 metrics) formats.
Implements canonical standardization formulas, strict frozen-state isolation,
and rigorous feature consistency validation (Specification Version 3.3).
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

# Constants for behavioral fingerprint structure
EXPECTED_PROBES = 30
METRICS_PER_PROBE = 6
TOTAL_FEATURES = EXPECTED_PROBES * METRICS_PER_PROBE  # 180

# Metric indices per probe
# [0] output_entropy      : H(p) in [0, ln(20) ~ 2.99573]
# [1] logit_gap           : gap >= 0.0
# [2] top5_prob_mass      : mass in [0, 1]
# [3] top1_prob           : prob in [0, 1]
# [4] distribution_spread : spread in [1.0, 10.0]
# [5] logprob_mean        : logprob <= 0.0
MAX_ENTROPY = float(np.log(20.0))  # 2.995732273553991


def validate_feature_consistency(
    vector: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
    """
    Validates a 180-dimensional feature vector against the 8 established
    mathematical invariants derived from probe_runner.py.

    Notice: Passing these checks establishes numerical bound and consistency
    validation; it does NOT prove that the full vector is physically realizable
    by an actual language model.

    Returns:
        (is_valid, list_of_violations)
    """
    issues = []
    vec = np.asarray(vector, dtype=np.float64)

    # 1. Finiteness check
    if not np.all(np.isfinite(vec)):
        issues.append("Non-finite values (NaN or Inf) detected in feature vector.")
        return False, issues

    if len(vec) != TOTAL_FEATURES:
        issues.append(f"Invalid dimensionality: expected {TOTAL_FEATURES}, got {len(vec)}")
        return False, issues

    for p in range(EXPECTED_PROBES):
        base = p * METRICS_PER_PROBE
        entropy = vec[base + 0]
        logit_gap = vec[base + 1]
        top5_mass = vec[base + 2]
        top1_prob = vec[base + 3]
        spread = vec[base + 4]
        logprob_mean = vec[base + 5]

        # 2. Entropy in [0, ln(20)]
        if entropy < -tolerance or entropy > MAX_ENTROPY + tolerance:
            issues.append(f"Probe {p}: Entropy out of bounds [0, {MAX_ENTROPY:.4f}]: got {entropy:.4f}")

        # 3. Logit gap >= 0
        if logit_gap < -tolerance:
            issues.append(f"Probe {p}: Negative logit gap: got {logit_gap:.4f}")

        # 4. Top-1 Probability in [0, 1]
        if top1_prob < -tolerance or top1_prob > 1.0 + tolerance:
            issues.append(f"Probe {p}: Top-1 probability out of bounds [0, 1]: got {top1_prob:.4f}")

        # 5. Top-5 Probability mass in [0, 1]
        if top5_mass < -tolerance or top5_mass > 1.0 + tolerance:
            issues.append(f"Probe {p}: Top-5 mass out of bounds [0, 1]: got {top5_mass:.4f}")

        # 6. Subset dominance: Top-1 <= Top-5
        if top1_prob > top5_mass + tolerance:
            issues.append(f"Probe {p}: Inconsistent probabilities: Top-1 ({top1_prob:.4f}) > Top-5 ({top5_mass:.4f})")

        # 7. Distribution spread in [1.0, 10.0]
        if spread < 1.0 - tolerance or spread > 10.0 + tolerance:
            issues.append(f"Probe {p}: Spread out of bounds [1.0, 10.0]: got {spread:.4f}")

        # 8. Mean log-probability <= 0
        if logprob_mean > tolerance:
            issues.append(f"Probe {p}: Positive mean log-probability: got {logprob_mean:.4f}")

    return (len(issues) == 0, issues)


def repair_feature_consistency(vector: np.ndarray) -> np.ndarray:
    """
    Projects and clips an uncalibrated synthetic perturbation vector onto
    the 8 established physical invariant boundaries.
    """
    repaired = np.array(vector, dtype=np.float64, copy=True)
    for p in range(EXPECTED_PROBES):
        base = p * METRICS_PER_PROBE
        repaired[base + 0] = np.clip(repaired[base + 0], 0.0, MAX_ENTROPY)
        repaired[base + 1] = np.maximum(0.0, repaired[base + 1])
        repaired[base + 3] = np.clip(repaired[base + 3], 0.0, 1.0)
        repaired[base + 2] = np.clip(repaired[base + 2], repaired[base + 3], 1.0)
        repaired[base + 4] = np.clip(repaired[base + 4], 1.0, 10.0)
        repaired[base + 5] = np.minimum(0.0, repaired[base + 5])
    return repaired


class FingerprintPreprocessor:
    """
    Standardization preprocessor strictly adhering to Specification Version 3.3.

    Canonical Denominator:
        denom_j = max(sigma_j, 1e-3) + 1e-5

    Supports:
        - Regime A (Zero-Reference): Fitted on permitted clean training vectors.
        - Regime B (Reference-Anchored Differential): Mean fixed to 0.0;
          variance fitted strictly on permitted clean training differentials.
    """

    def __init__(self, regime: str = "A"):
        regime = regime.upper()
        if regime not in ("A", "B"):
            raise ValueError(f"Unknown regime: {regime}. Must be 'A' or 'B'.")
        self.regime = regime
        self._is_frozen = False
        self._mu: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._denom: Optional[np.ndarray] = None

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def mu(self) -> Optional[np.ndarray]:
        return self._mu.copy() if self._mu is not None else None

    @property
    def std(self) -> Optional[np.ndarray]:
        return self._std.copy() if self._std is not None else None

    @property
    def denom(self) -> Optional[np.ndarray]:
        return self._denom.copy() if self._denom is not None else None

    def fit(self, X: np.ndarray) -> "FingerprintPreprocessor":
        """
        Fits preprocessor parameters strictly on permitted training data.
        Raises RuntimeError if preprocessor is frozen.
        """
        if self._is_frozen:
            raise RuntimeError("Cannot re-fit a frozen preprocessor. Parameters are immutable.")

        X = np.asarray(X, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ValueError("Training inputs contain non-finite values.")
        if len(X) < 1:
            raise ValueError("Cannot fit preprocessor on empty sample matrix.")

        n_features = X.shape[1]

        if self.regime == "A":
            # Regime A: Standardize clean training vectors
            mu = np.mean(X, axis=0)
            if len(X) > 1:
                std = np.std(X, axis=0, ddof=1)
            else:
                std = np.zeros(n_features, dtype=np.float64)
        else:
            # Regime B: Zero-centered differential; variance fitted around zero
            mu = np.zeros(n_features, dtype=np.float64)
            if len(X) > 1:
                std = np.sqrt(np.sum(X ** 2, axis=0) / (len(X) - 1))
            else:
                std = np.zeros(n_features, dtype=np.float64)

        # Canonical denominator expression
        denom = np.maximum(std, 1e-3) + 1e-5

        self._mu = mu
        self._std = std
        self._denom = denom
        return self

    def freeze(self) -> "FingerprintPreprocessor":
        """
        Freezes parameter state and makes internal parameter arrays read-only.
        """
        if self._mu is None or self._denom is None:
            raise RuntimeError("Cannot freeze unfitted preprocessor.")
        self._mu.flags.writeable = False
        self._std.flags.writeable = False
        self._denom.flags.writeable = False
        self._is_frozen = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms inputs using frozen parameters.
        Does not mutate any internal state or parameters.
        """
        if self._mu is None or self._denom is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        X = np.asarray(X, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ValueError("Input to transform contains non-finite values.")

        return (X - self._mu) / self._denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class FingerprintDataLoader:
    """Read-only loader for physical and baseline behavioral fingerprints."""

    EXPECTED_PROBES = EXPECTED_PROBES
    METRICS_PER_PROBE = METRICS_PER_PROBE
    TOTAL_FEATURES = TOTAL_FEATURES

    def __init__(self, fingerprints_dir: Optional[Path] = None):
        self.fingerprints_dir = fingerprints_dir or IMP1_FINGERPRINTS_DIR
        if not self.fingerprints_dir.exists():
            raise FileNotFoundError(f"Fingerprints directory not found: {self.fingerprints_dir}")

    def load_physical_fingerprints(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads the 4 authoritative physical model fingerprints from Implementation 1.
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

            probe_results = data.get("per_probe_results", [])
            if len(probe_results) != self.EXPECTED_PROBES:
                raise ValueError(
                    f"Unexpected probe count in {path}: expected {self.EXPECTED_PROBES}, got {len(probe_results)}"
                )

            matrix_2d = np.array([p["vector"] for p in probe_results], dtype=np.float64)  # (30, 6)
            vector_1d = matrix_2d.flatten()  # (180,)

            # Validate physical vector consistency
            is_valid, issues = validate_feature_consistency(vector_1d)
            if not is_valid:
                raise ValueError(f"Physical fingerprint {key} failed consistency validation: {issues}")

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
        baselines: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    ) -> np.ndarray:
        """
        Standardizes raw feature vector against the upstream base model
        using the canonical bounded denominator expression:
        denom = max(sigma, 1e-3) + 1e-5
        """
        if baselines is None:
            baselines = self.load_baselines()

        if arch not in baselines:
            raise KeyError(f"No baseline found for architecture: {arch}")

        mu = baselines[arch]["mean"]
        sigma = baselines[arch]["std"]
        denom = np.maximum(sigma, 1e-3) + 1e-5
        return (raw_vector - mu) / denom

    def build_supervised_training_cohort(
        self,
        n_samples_per_arch: int = 25,
        seed: int = 42,
        normalized: bool = True,
        sigma_noise: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Constructs a supervised training distribution strictly from LLaMA-3 and Mistral-7B
        clean physical anchors, adhering to Specification Version 3.3.

        DATA PROVENANCE DISCLOSURE:
        - Negative training examples (y = 0): Clean physical anchors + simulated benign
          perturbations (Gaussian noise sigma_noise with feature-wise boundary repair).
        - Positive training examples (y = 1): Simulated trojan distortions on LLaMA and Mistral
          anchors (with feature-wise boundary repair).
        - Test set: 100% genuine physical held-out Qwen checkpoints (clean reference vs poisoned PoC).
        - Normalization: When normalized=True, fitted strictly on clean training samples (y = 0)
          via FingerprintPreprocessor (Regime A) with denominator max(sigma, 1e-3) + 1e-5.
          Zero Qwen data influences preprocessing.
        """
        phys_data = self.load_physical_fingerprints()
        rng = np.random.RandomState(seed)

        llama_vec = phys_data["llama3_clean"]["vector_1d"]
        mistral_vec = phys_data["mistral_clean"]["vector_1d"]
        n_features = len(llama_vec)

        raw_clean_samples = []
        raw_poison_samples = []
        clean_archs = []
        poison_archs = []

        anchors = [("llama3", llama_vec), ("mistral", mistral_vec)]

        for arch_name, base_vec in anchors:
            # 1. Simulated benign perturbations (Clean, y = 0)
            for _ in range(n_samples_per_arch):
                noise = rng.normal(0, sigma_noise, n_features)
                perturbed = base_vec + noise
                repaired = repair_feature_consistency(perturbed)
                is_valid, issues = validate_feature_consistency(repaired)
                if not is_valid:
                    raise RuntimeError(f"Repaired clean perturbation failed consistency: {issues}")
                raw_clean_samples.append(repaired)
                clean_archs.append(arch_name)

            # 2. Simulated anomalous distortions (Poisoned, y = 1)
            for _ in range(n_samples_per_arch):
                p_vec = base_vec.copy()
                distorted_probes = rng.choice(EXPECTED_PROBES, size=8, replace=False)
                for p in distorted_probes:
                    f_idx = p * METRICS_PER_PROBE
                    p_vec[f_idx + 0] = max(0.001, p_vec[f_idx + 0] * 0.25 - rng.uniform(0.1, 0.3))
                    p_vec[f_idx + 1] = p_vec[f_idx + 1] + rng.uniform(2.5, 5.0)
                    p_vec[f_idx + 3] = min(0.99, p_vec[f_idx + 3] + 0.15)
                    p_vec[f_idx + 2] = min(1.0, max(p_vec[f_idx + 3] + 0.01, p_vec[f_idx + 2] + 0.05))
                    p_vec[f_idx + 4] = max(1.0, p_vec[f_idx + 4] * 0.7)
                p_vec += rng.normal(0, sigma_noise, n_features)
                repaired_p = repair_feature_consistency(p_vec)
                is_valid, issues = validate_feature_consistency(repaired_p)
                if not is_valid:
                    raise RuntimeError(f"Repaired anomaly perturbation failed consistency: {issues}")
                raw_poison_samples.append(repaired_p)
                poison_archs.append(arch_name)

        X_train_raw = np.vstack([raw_clean_samples, raw_poison_samples])
        y_train = np.array([0] * len(raw_clean_samples) + [1] * len(raw_poison_samples))

        # Test set: genuine physical held-out Qwen checkpoints
        qwen_clean_vec = phys_data["qwen_clean"]["vector_1d"]
        qwen_poison_vec = phys_data["qwen_poisoned"]["vector_1d"]
        X_test_raw = np.vstack([qwen_clean_vec, qwen_poison_vec])
        y_test = np.array([0, 1])

        if normalized:
            # Fit preprocessor strictly on clean training samples
            clean_train_matrix = np.array(raw_clean_samples)
            preprocessor = FingerprintPreprocessor(regime="A")
            preprocessor.fit(clean_train_matrix)
            preprocessor.freeze()

            X_train = preprocessor.transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)
        else:
            X_train = X_train_raw
            X_test = X_test_raw

        metadata = {
            "train_samples": len(X_train),
            "train_provenance": "SIMULATED_BENIGN_AND_ANOMALOUS_PERTURBATIONS_ON_LLAMA_MISTRAL_ANCHORS",
            "test_samples": len(X_test),
            "test_provenance": "100%_GENUINE_PHYSICAL_HARDWARE_QWEN_CHECKPOINTS",
            "train_architectures": ["llama3", "mistral"],
            "test_architecture": "qwen (held-out)",
            "normalized": normalized,
            "denominator_rule": "max(sigma, 1e-3) + 1e-5"
        }

        return X_train, y_train, X_test, y_test, metadata

    def build_regime_b_cohort(
        self,
        n_samples_per_arch: int = 25,
        seed: int = 42,
        sigma_noise: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Constructs training and evaluation cohorts for Regime B (Reference-Anchored Detection).
        Strictly enforces intra-architecture reference pairing.
        Preprocesses via FingerprintPreprocessor (Regime B) with zero-mean and clean variance.
        """
        phys_data = self.load_physical_fingerprints()
        rng = np.random.RandomState(seed)

        llama_anchor = phys_data["llama3_clean"]["vector_1d"]
        mistral_anchor = phys_data["mistral_clean"]["vector_1d"]
        n_features = len(llama_anchor)

        clean_diffs = []
        anom_diffs = []

        for arch_name, base_vec in [("llama3", llama_anchor), ("mistral", mistral_anchor)]:
            # Self-differential (Delta = 0, y = 0)
            clean_diffs.append(np.zeros(n_features, dtype=np.float64))

            # Simulated clean differentials: target = base + noise, ref = base
            for _ in range(n_samples_per_arch - 1):
                noise = rng.normal(0, sigma_noise, n_features)
                target = repair_feature_consistency(base_vec + noise)
                delta = target - base_vec
                clean_diffs.append(delta)

            # Simulated anomalous differentials: target = distorted, ref = base
            for _ in range(n_samples_per_arch):
                p_vec = base_vec.copy()
                distorted_probes = rng.choice(EXPECTED_PROBES, size=8, replace=False)
                for p in distorted_probes:
                    f_idx = p * METRICS_PER_PROBE
                    p_vec[f_idx + 0] = max(0.001, p_vec[f_idx + 0] * 0.25 - rng.uniform(0.1, 0.3))
                    p_vec[f_idx + 1] = p_vec[f_idx + 1] + rng.uniform(2.5, 5.0)
                    p_vec[f_idx + 3] = min(0.99, p_vec[f_idx + 3] + 0.15)
                    p_vec[f_idx + 2] = min(1.0, max(p_vec[f_idx + 3] + 0.01, p_vec[f_idx + 2] + 0.05))
                    p_vec[f_idx + 4] = max(1.0, p_vec[f_idx + 4] * 0.7)
                p_vec += rng.normal(0, sigma_noise, n_features)
                target = repair_feature_consistency(p_vec)
                delta = target - base_vec
                anom_diffs.append(delta)

        X_train_diff_raw = np.vstack([clean_diffs, anom_diffs])
        y_train = np.array([0] * len(clean_diffs) + [1] * len(anom_diffs))

        # Test set differentials: Qwen clean against itself (y=0) and Qwen poison against Qwen clean (y=1)
        qwen_clean_vec = phys_data["qwen_clean"]["vector_1d"]
        qwen_poison_vec = phys_data["qwen_poisoned"]["vector_1d"]

        qwen_clean_diff = qwen_clean_vec - qwen_clean_vec  # Exactly 0
        qwen_poison_diff = qwen_poison_vec - qwen_clean_vec

        X_test_diff_raw = np.vstack([qwen_clean_diff, qwen_poison_diff])
        y_test = np.array([0, 1])

        # Preprocessor fitted strictly on clean training differentials
        preprocessor = FingerprintPreprocessor(regime="B")
        preprocessor.fit(np.array(clean_diffs))
        preprocessor.freeze()

        X_train_diff = preprocessor.transform(X_train_diff_raw)
        X_test_diff = preprocessor.transform(X_test_diff_raw)

        metadata = {
            "regime": "B_REFERENCE_ANCHORED",
            "train_samples": len(X_train_diff),
            "test_samples": len(X_test_diff),
            "centering": "ANALYTICAL_ZERO",
            "scaling": "CLEAN_TRAIN_DIFFERENTIAL_VARIANCE",
            "denominator_rule": "max(sigma, 1e-3) + 1e-5"
        }

        return X_train_diff, y_train, X_test_diff, y_test, metadata

    def compute_feature_wise_scale(self) -> np.ndarray:
        """
        Computes the canonical simulation-derived feature-wise reference scale vector s in R^180
        strictly from the clean training physical anchors (LLaMA-3-8B and Mistral-7B-Instruct-v0.2).
        Strictly zero Qwen data is accessed.
        
        Formula (Specification v1.3.1 Section 4):
            S_m = std across 60 clean training probe measurements for metric m
            s_j = max(|x_llama[j] - x_mistral[j]| / sqrt(2), 0.20 * S_m, 1e-3) + 1e-5
        """
        phys = self.load_physical_fingerprints()
        x_llama = phys["llama3_clean"]["vector_1d"]
        x_mistral = phys["mistral_clean"]["vector_1d"]

        llama_matrix = x_llama.reshape((EXPECTED_PROBES, METRICS_PER_PROBE))
        mistral_matrix = x_mistral.reshape((EXPECTED_PROBES, METRICS_PER_PROBE))
        combined_metrics = np.vstack([llama_matrix, mistral_matrix])  # (60, 6)

        S_m = np.std(combined_metrics, axis=0, ddof=1)  # shape (6,)

        s = np.zeros(TOTAL_FEATURES, dtype=np.float64)
        for p in range(EXPECTED_PROBES):
            for m in range(METRICS_PER_PROBE):
                j = p * METRICS_PER_PROBE + m
                cross_diff = np.abs(x_llama[j] - x_mistral[j]) / np.sqrt(2.0)
                s[j] = max(cross_diff, 0.20 * S_m[m], 1e-3) + 1e-5

        return s

    def build_stage3_partitions(
        self,
        seed: int = 42
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], FingerprintPreprocessor]:
        """
        Constructs the 4-way disjoint partitions strictly following Specification v1.3.1:
        1. Training Partition D_train (N = 120, seed = 42)
        2. Calibration Partition D_calib (N = 60, seed = 1042)
        3. Synthetic Evaluation Partition D_eval (N = 100, seed = 2042)
        4. Physical Test Partition D_phys (N = 2, held-out physical Qwen pair)
        
        Returns:
            train_dict, calib_dict, eval_dict, phys_dict, preprocessor
        """
        scale_vec = self.compute_feature_wise_scale()
        phys = self.load_physical_fingerprints()
        llama_clean = phys["llama3_clean"]["vector_1d"]
        mistral_clean = phys["mistral_clean"]["vector_1d"]
        domain_probes = list(range(8, 20))  # 12 probes (8..19)

        def _generate_clean_sample(base_vec: np.ndarray, mechanism: str, param: float, rng: np.random.RandomState) -> np.ndarray:
            if mechanism == "B0":
                # Exact identity differential (Delta = 0)
                return np.zeros(TOTAL_FEATURES, dtype=np.float64)
            elif mechanism == "B1":
                # Feature-scaled Gaussian drift: delta_j = eta * s_j * z_j
                eta = param
                delta_raw = eta * scale_vec * rng.normal(0, 1, TOTAL_FEATURES)
                target = repair_feature_consistency(base_vec + delta_raw)
                return target - base_vec
            elif mechanism == "B2":
                # Directional domain specialization: alpha = param
                alpha = param
                delta_raw = np.zeros(TOTAL_FEATURES, dtype=np.float64)
                for p in range(EXPECTED_PROBES):
                    f = p * METRICS_PER_PROBE
                    if p in domain_probes:
                        delta_raw[f + 0] = - alpha * scale_vec[f + 0]
                        delta_raw[f + 1] = + alpha * scale_vec[f + 1]
                        delta_raw[f + 3] = + alpha * scale_vec[f + 3]
                        delta_raw[f + 2] = + alpha * scale_vec[f + 2]
                        delta_raw[f + 4] = - 0.5 * alpha * scale_vec[f + 4]
                        delta_raw[f + 5] = + 0.5 * alpha * scale_vec[f + 5]
                    else:
                        delta_raw[f : f + 6] = 0.02 * scale_vec[f : f + 6] * rng.normal(0, 1, 6)
                target = repair_feature_consistency(base_vec + delta_raw)
                return target - base_vec
            elif mechanism == "B3":
                # Hand-designed precision degradation heuristic: beta = param
                beta = param
                delta_raw = np.zeros(TOTAL_FEATURES, dtype=np.float64)
                for p in range(EXPECTED_PROBES):
                    f = p * METRICS_PER_PROBE
                    delta_raw[f + 0] = + beta * scale_vec[f + 0]
                    delta_raw[f + 1] = - beta * scale_vec[f + 1]
                    delta_raw[f + 4] = + beta * scale_vec[f + 4]
                    delta_raw[f + 5] = - beta * scale_vec[f + 5]
                target = repair_feature_consistency(base_vec + delta_raw)
                return target - base_vec
            else:
                raise ValueError(f"Unknown mechanism: {mechanism}")

        def _generate_trojan_sample(base_vec: np.ndarray, sigma_noise: float, rng: np.random.RandomState) -> np.ndarray:
            p_vec = base_vec.copy()
            distorted_probes = rng.choice(EXPECTED_PROBES, size=8, replace=False)
            for p in distorted_probes:
                f_idx = p * METRICS_PER_PROBE
                p_vec[f_idx + 0] = max(0.001, p_vec[f_idx + 0] * 0.25 - rng.uniform(0.1, 0.3))
                p_vec[f_idx + 1] = p_vec[f_idx + 1] + rng.uniform(2.5, 5.0)
                p_vec[f_idx + 3] = min(0.99, p_vec[f_idx + 3] + 0.15)
                p_vec[f_idx + 2] = min(1.0, max(p_vec[f_idx + 3] + 0.01, p_vec[f_idx + 2] + 0.05))
                p_vec[f_idx + 4] = max(1.0, p_vec[f_idx + 4] * 0.70)
            p_vec += rng.normal(0, sigma_noise, TOTAL_FEATURES)
            target = repair_feature_consistency(p_vec)
            return target - base_vec

        # =========================================================================
        # 1. PARTITION 1: TRAINING SET D_train (N = 120, seed = 42)
        # =========================================================================
        rng_train = np.random.RandomState(seed)
        train_diffs = []
        train_labels = []

        # Clean (N = 60: 30 LLaMA, 30 Mistral)
        for arch_name, base_vec in [("llama3", llama_clean), ("mistral", mistral_clean)]:
            # B0: 1 sample
            train_diffs.append(_generate_clean_sample(base_vec, "B0", 0.0, rng_train))
            train_labels.append(0)
            # B1: 19 samples (eta = 0.05)
            for _ in range(19):
                train_diffs.append(_generate_clean_sample(base_vec, "B1", 0.05, rng_train))
                train_labels.append(0)
            # B2: 10 samples (alpha = 0.05)
            for _ in range(10):
                train_diffs.append(_generate_clean_sample(base_vec, "B2", 0.05, rng_train))
                train_labels.append(0)

        # Trojans (N = 60: 30 LLaMA, 30 Mistral, sigma = 0.05)
        for arch_name, base_vec in [("llama3", llama_clean), ("mistral", mistral_clean)]:
            for _ in range(30):
                train_diffs.append(_generate_trojan_sample(base_vec, 0.05, rng_train))
                train_labels.append(1)

        X_train_raw = np.array(train_diffs, dtype=np.float64)
        y_train = np.array(train_labels, dtype=np.int32)

        # Fit preprocessor strictly on clean training differentials (Regime B)
        clean_train_diffs = X_train_raw[y_train == 0]
        preprocessor = FingerprintPreprocessor(regime="B")
        preprocessor.fit(clean_train_diffs)
        preprocessor.freeze()

        X_train = preprocessor.transform(X_train_raw)
        train_dict = {
            "X": X_train,
            "y": y_train,
            "N": len(X_train),
            "N_clean": int(np.sum(y_train == 0)),
            "N_trojan": int(np.sum(y_train == 1)),
            "scale_vector": scale_vec
        }

        # =========================================================================
        # 2. PARTITION 2: CALIBRATION SET D_calib (N = 60, seed = 1042)
        # =========================================================================
        rng_calib = np.random.RandomState(seed + 1000)
        calib_diffs = []
        calib_labels = []

        # Clean (N = 30):
        # B0 (5): 3 LLaMA, 2 Mistral
        for _ in range(3):
            calib_diffs.append(_generate_clean_sample(llama_clean, "B0", 0.0, rng_calib))
            calib_labels.append(0)
        for _ in range(2):
            calib_diffs.append(_generate_clean_sample(mistral_clean, "B0", 0.0, rng_calib))
            calib_labels.append(0)

        # B1 eta=0.03 (8): 4 LLaMA, 4 Mistral
        for _ in range(4):
            calib_diffs.append(_generate_clean_sample(llama_clean, "B1", 0.03, rng_calib))
            calib_labels.append(0)
        for _ in range(4):
            calib_diffs.append(_generate_clean_sample(mistral_clean, "B1", 0.03, rng_calib))
            calib_labels.append(0)

        # B1 eta=0.08 (7): 4 LLaMA, 3 Mistral
        for _ in range(4):
            calib_diffs.append(_generate_clean_sample(llama_clean, "B1", 0.08, rng_calib))
            calib_labels.append(0)
        for _ in range(3):
            calib_diffs.append(_generate_clean_sample(mistral_clean, "B1", 0.08, rng_calib))
            calib_labels.append(0)

        # B2 alpha=0.05 (5): 3 LLaMA, 2 Mistral
        for _ in range(3):
            calib_diffs.append(_generate_clean_sample(llama_clean, "B2", 0.05, rng_calib))
            calib_labels.append(0)
        for _ in range(2):
            calib_diffs.append(_generate_clean_sample(mistral_clean, "B2", 0.05, rng_calib))
            calib_labels.append(0)

        # B2 alpha=0.15 (5): 2 LLaMA, 3 Mistral
        for _ in range(2):
            calib_diffs.append(_generate_clean_sample(llama_clean, "B2", 0.15, rng_calib))
            calib_labels.append(0)
        for _ in range(3):
            calib_diffs.append(_generate_clean_sample(mistral_clean, "B2", 0.15, rng_calib))
            calib_labels.append(0)

        # Trojans (N = 30): 15 LLaMA, 15 Mistral (sigma = 0.05)
        for _ in range(15):
            calib_diffs.append(_generate_trojan_sample(llama_clean, 0.05, rng_calib))
            calib_labels.append(1)
        for _ in range(15):
            calib_diffs.append(_generate_trojan_sample(mistral_clean, 0.05, rng_calib))
            calib_labels.append(1)

        X_calib_raw = np.array(calib_diffs, dtype=np.float64)
        y_calib = np.array(calib_labels, dtype=np.int32)
        X_calib = preprocessor.transform(X_calib_raw)

        calib_dict = {
            "X": X_calib,
            "y": y_calib,
            "N": len(X_calib),
            "N_clean": int(np.sum(y_calib == 0)),
            "N_trojan": int(np.sum(y_calib == 1))
        }

        # =========================================================================
        # 3. PARTITION 3: SYNTHETIC EVALUATION SET D_eval (N = 100, seed = 2042)
        # =========================================================================
        rng_eval = np.random.RandomState(seed + 2000)
        eval_diffs = []
        eval_labels = []
        eval_records = []

        # Specification v1.3.1 allocation schedule: exactly 100 samples
        schedule = [
            # (condition_group, mechanism, param, n_llama, n_mistral)
            # Identity (10)
            ("Identity Baseline", "B0", 0.0, 5, 5),
            # Seen Magnitudes (22)
            ("Seen Magnitudes", "B1", 0.03, 2, 2),
            ("Seen Magnitudes", "B1", 0.05, 2, 2),
            ("Seen Magnitudes", "B1", 0.08, 2, 2),
            ("Seen Magnitudes", "B2", 0.05, 3, 2),
            ("Seen Magnitudes", "B2", 0.15, 3, 2),
            # Unseen Magnitudes (18)
            ("Unseen Magnitudes", "B1", 0.01, 2, 2),
            ("Unseen Magnitudes", "B1", 0.12, 2, 2),
            ("Unseen Magnitudes", "B2", 0.10, 2, 3),
            ("Unseen Magnitudes", "B2", 0.20, 2, 3),
            # Held-Out Mechanism B3 (20)
            ("Held-Out Mechanism B3", "B3", 0.03, 3, 3),
            ("Held-Out Mechanism B3", "B3", 0.06, 4, 3),
            ("Held-Out Mechanism B3", "B3", 0.10, 3, 4),
            # Synthetic Trojans (30)
            ("Synthetic Trojans", "TROJAN", 0.01, 5, 5),
            ("Synthetic Trojans", "TROJAN", 0.05, 5, 5),
            ("Synthetic Trojans", "TROJAN", 0.10, 5, 5),
        ]

        for cond_group, mech, param, n_llama, n_mistral in schedule:
            is_trojan = (mech == "TROJAN")
            lbl = 1 if is_trojan else 0

            # LLaMA draws
            for _ in range(n_llama):
                if is_trojan:
                    vec = _generate_trojan_sample(llama_clean, param, rng_eval)
                else:
                    vec = _generate_clean_sample(llama_clean, mech, param, rng_eval)
                eval_diffs.append(vec)
                eval_labels.append(lbl)
                eval_records.append({
                    "condition_group": cond_group,
                    "mechanism": mech,
                    "parameter": param,
                    "anchor": "llama3",
                    "true_label": lbl
                })

            # Mistral draws
            for _ in range(n_mistral):
                if is_trojan:
                    vec = _generate_trojan_sample(mistral_clean, param, rng_eval)
                else:
                    vec = _generate_clean_sample(mistral_clean, mech, param, rng_eval)
                eval_diffs.append(vec)
                eval_labels.append(lbl)
                eval_records.append({
                    "condition_group": cond_group,
                    "mechanism": mech,
                    "parameter": param,
                    "anchor": "mistral",
                    "true_label": lbl
                })

        X_eval_raw = np.array(eval_diffs, dtype=np.float64)
        y_eval = np.array(eval_labels, dtype=np.int32)
        X_eval = preprocessor.transform(X_eval_raw)

        eval_dict = {
            "X": X_eval,
            "y": y_eval,
            "records": eval_records,
            "N": len(X_eval),
            "N_clean": int(np.sum(y_eval == 0)),
            "N_trojan": int(np.sum(y_eval == 1))
        }

        # =========================================================================
        # 4. PARTITION 4: PHYSICAL TEST COHORT D_phys (N = 2)
        # =========================================================================
        qwen_clean_vec = phys["qwen_clean"]["vector_1d"]
        qwen_poison_vec = phys["qwen_poisoned"]["vector_1d"]

        qwen_clean_diff = qwen_clean_vec - qwen_clean_vec  # Exactly 0.0
        qwen_poison_diff = qwen_poison_vec - qwen_clean_vec

        X_phys_raw = np.vstack([qwen_clean_diff, qwen_poison_diff])
        y_phys = np.array([0, 1], dtype=np.int32)
        X_phys = preprocessor.transform(X_phys_raw)

        phys_dict = {
            "X": X_phys,
            "y": y_phys,
            "N": len(X_phys),
            "labels": ["qwen_clean_reference", "qwen_poisoned_poc"]
        }

        return train_dict, calib_dict, eval_dict, phys_dict, preprocessor

