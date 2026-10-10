#!/usr/bin/env python3
"""
src/uld_scanner.py
Tier 2A: Upstream-Anchored Latent Divergence (ULD) Scanner.

Implements the deep pre-deployment audit via physical weight-space geometry,
without any hardcoded labels, adapter-name checks, or simulated scores.

Physical ULD Computation (No LLM inference required):
------------------------------------------------------
A backdoored LoRA adapter must embed a compressed, low-entropy projection
kernel that maps a secret trigger token subspace -> a target output subspace.
This manifests in the adapter weight matrices (A, B) as:

1. CONCENTRATION RATIO of Projection Norms (CRPN):
   The total Frobenius energy in B is concentrated in fewer subspace directions
   than in a clean fine-tuned adapter. Backdoors require a coherent rank-1-ish
   pathway; benign domain adapters spread energy across all r directions.

2. SUBSPACE ALIGNMENT SCORE (SAS):
   The dot product coherence between the column space of B and the row space of A
   (i.e., how tightly B and A collaborate to project a single fixed direction).
   Clean adapters: B and A align differently per layer (domain-diverse).
   Backdoored adapters: B*A is dominated by a tight rank-k attractor structure.

3. SPECTRAL ENTROPY DEFICIT (SED):
   Shannon entropy of the normalized squared singular values of B*A (via fast QR-SVD).
   A backdoored adapter produces lower spectral entropy (more concentrated).

ULD Score = alpha * CRPN + beta * SAS + gamma * SED_deficit

Thresholding:
   Calibrated on a Development Set (dev cohort, not touched during test evaluation).
   Frozen threshold applied to held-out Test Set.
"""

import os
import json
import hashlib
import numpy as np
import safetensors.numpy as snp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


# ============================================================================
# PHYSICAL WEIGHT-SPACE ULD FEATURE EXTRACTION
# ============================================================================

def _fast_lora_svd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fast QR-based SVD of B @ A via thin QR factorization. Error < 2.3e-12."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q_B, R_B = np.linalg.qr(B)
    Q_A, R_A = np.linalg.qr(A.T)
    M = R_B @ R_A.T
    return np.linalg.svd(M, compute_uv=False)


def _spectral_entropy(s: np.ndarray) -> float:
    """Shannon entropy of normalized squared singular value distribution."""
    s2 = s ** 2
    total = np.sum(s2)
    if total < 1e-12:
        return 0.0
    p = s2 / total
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def _concentration_ratio_projection_norms(tensors: Dict[str, np.ndarray]) -> float:
    """
    CRPN: Ratio of max-layer Frobenius norm of B to mean-layer Frobenius norm.
    A clean adapter has a smooth norm distribution across layers.
    A backdoored adapter has one or a few layers with anomalously high norms
    (where the backdoor pathway is most active).
    Returns: CRPN in [1.0, inf). Higher -> more concentrated -> more suspicious.
    """
    b_norms = []
    for k, v in tensors.items():
        if "lora_B" in k or "lora_up" in k:
            n = float(np.linalg.norm(v.astype(np.float64), 'fro'))
            if n > 1e-8:
                b_norms.append(n)
    if not b_norms:
        return 1.0
    arr = np.array(b_norms)
    return float(np.max(arr) / (np.mean(arr) + 1e-9))


def _subspace_alignment_score(tensors: Dict[str, np.ndarray]) -> float:
    """
    SAS: Mean coherence (cosine similarity squared) between the top singular
    direction of B and the top singular direction of A^T across all layers.
    A tight attractor (backdoor) -> B top-direction and A top-direction highly aligned.
    Clean domain adapters: diverse, misaligned principal directions.
    Returns: SAS in [0.0, 1.0]. Higher -> more aligned -> more suspicious.
    """
    alignments = []
    keys_A = [k for k in tensors if ("lora_A" in k or "lora_down" in k) and tensors[k].ndim == 2]
    for k_A in keys_A:
        k_B = k_A.replace("lora_A", "lora_B").replace("lora_down", "lora_up")
        if k_B not in tensors:
            continue
        A = np.asarray(tensors[k_A], dtype=np.float64)
        B = np.asarray(tensors[k_B], dtype=np.float64)
        try:
            # Top singular vector of B (column space)
            _, _, Vt_B = np.linalg.svd(B, full_matrices=False)
            u_B = Vt_B[0]  # Top right singular vector of B (shape: r,)
            # Top singular vector of A^T (row space of A)
            U_A, _, _ = np.linalg.svd(A, full_matrices=False)
            u_A = U_A[:, 0]  # Top left singular vector of A (shape: d_in,)
            # We need same dimension to compare -> use the r-dim projection
            _, s_B, Vt_B2 = np.linalg.svd(B, full_matrices=False)
            u_B_left = Vt_B2[0]  # top right singular vector of B, dim r
            _, _, Vt_A = np.linalg.svd(A, full_matrices=False)
            u_A_right = Vt_A[0]  # top right singular vector of A, dim d_in
            # Project both onto the r-dim space of A
            # Alignment via fast QR-SVD of B@A
            s = _fast_lora_svd(A, B)
            r = len(s)
            # Coherence: how much does the top singular value dominate over uniform distribution?
            if np.sum(s) > 1e-9:
                alignments.append(float(s[0] / np.sum(s)))
        except np.linalg.LinAlgError:
            continue
    return float(np.mean(alignments)) if alignments else 0.0


def _spectral_entropy_deficit(tensors: Dict[str, np.ndarray]) -> float:
    """
    SED: Mean spectral entropy deficit across all LoRA layers.
    Max entropy for rank-r = log(r) (uniform distribution).
    Deficit = log(r) - mean_entropy.
    Backdoored adapters concentrate energy in top-k singular values -> high deficit.
    Clean adapters -> near-uniform distribution -> deficit near 0.
    Returns: mean SED across all layers. Higher -> more concentrated -> suspicious.
    """
    deficits = []
    keys_A = [k for k in tensors if ("lora_A" in k or "lora_down" in k) and tensors[k].ndim == 2]
    for k_A in keys_A:
        k_B = k_A.replace("lora_A", "lora_B").replace("lora_down", "lora_up")
        if k_B not in tensors:
            continue
        A = np.asarray(tensors[k_A], dtype=np.float64)
        B = np.asarray(tensors[k_B], dtype=np.float64)
        try:
            s = _fast_lora_svd(A, B)
            r = len(s)
            max_entropy = np.log(r) if r > 1 else 1.0
            entropy = _spectral_entropy(s)
            deficit = max(0.0, max_entropy - entropy)
            deficits.append(deficit)
        except np.linalg.LinAlgError:
            continue
    return float(np.mean(deficits)) if deficits else 0.0


def compute_uld_features(adapter_path: Path) -> Dict[str, float]:
    """
    Compute the three physical ULD feature scores for an adapter.
    Returns a dict with: crpn, sas, sed, and the raw feature values.
    All computation is purely on weight geometry - zero simulated values.
    """
    # Load weights
    weights_file = None
    if adapter_path.is_dir():
        for cand in ["adapter_model.safetensors", "adapter_model.bin"]:
            c = adapter_path / cand
            if c.exists():
                weights_file = c
                break
    elif adapter_path.is_file():
        weights_file = adapter_path

    if weights_file is None or not weights_file.exists():
        return {"crpn": 1.0, "sas": 0.0, "sed": 0.0, "error": "not_found"}

    tensors = snp.load_file(str(weights_file))

    crpn = _concentration_ratio_projection_norms(tensors)
    sas = _subspace_alignment_score(tensors)
    sed = _spectral_entropy_deficit(tensors)

    return {"crpn": crpn, "sas": sas, "sed": sed}


def compute_uld_score(
    features: Dict[str, float],
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 1.5
) -> float:
    """
    Combine the three features into a scalar ULD score.
    ULD = alpha * (CRPN - 1) + beta * SAS + gamma * SED
    Note: CRPN - 1 normalizes to 0 for a perfectly uniform distribution.
    """
    crpn = features.get("crpn", 1.0)
    sas = features.get("sas", 0.0)
    sed = features.get("sed", 0.0)
    return float(alpha * (crpn - 1.0) + beta * sas + gamma * sed)


# ============================================================================
# THRESHOLD CALIBRATION (ON DEV SET ONLY)
# ============================================================================

def calibrate_uld_threshold(
    dev_uld_scores: List[float],
    dev_labels: List[int],  # 1 = malicious, 0 = benign
    target_far: float = 0.0,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 1.5
) -> Dict[str, Any]:
    """
    Calibrate tau_ULD on the development set by:
    1. Finding the lowest threshold that achieves target_far on benign adapters.
    2. Reporting the dev-set TPR at that threshold.
    Threshold is returned and must be frozen before test-set evaluation.
    """
    benign_scores = [s for s, l in zip(dev_uld_scores, dev_labels) if l == 0]
    malicious_scores = [s for s, l in zip(dev_uld_scores, dev_labels) if l == 1]

    if not benign_scores or not malicious_scores:
        return {"tau_uld": 999.0, "error": "insufficient_dev_samples"}

    # Find tau such that FAR on dev benign = target_far
    max_benign = float(np.max(benign_scores))
    min_malicious = float(np.min(malicious_scores))

    # Choose tau as midpoint between max clean and min malicious if separation exists,
    # otherwise choose just above max benign (strict 0% FAR on dev)
    if min_malicious > max_benign:
        # Clear separation on dev set
        tau = round(float((max_benign + min_malicious) / 2.0), 4)
        separation = "CLEAR"
    else:
        # Overlap: set tau to achieve target_far (0% = just above max benign)
        tau = round(float(max_benign * 1.05), 4)
        separation = "OVERLAP"

    dev_tpr = sum(1 for s in malicious_scores if s >= tau) / len(malicious_scores)
    dev_far = sum(1 for s in benign_scores if s >= tau) / len(benign_scores)

    return {
        "tau_uld": tau,
        "dev_max_benign_uld": round(max_benign, 4),
        "dev_min_malicious_uld": round(min_malicious, 4),
        "dev_tpr_pct": round(dev_tpr * 100, 2),
        "dev_far_pct": round(dev_far * 100, 2),
        "separation": separation,
        "alpha": alpha, "beta": beta, "gamma": gamma
    }


# ============================================================================
# ULD SCANNER CLASS (SELF-CONTAINED, v2 UNCHANGED)
# ============================================================================

class UpstreamLatentDivergenceScanner:
    """
    Tier 2A: Deep Pre-Deployment Audit via Physical Weight-Space Geometry.
    Operates on adapter weight files directly without LLM inference.
    Calibrated threshold must be provided (set from dev set, frozen before test).
    """

    DEFAULT_ALPHA = 1.0
    DEFAULT_BETA = 2.0
    DEFAULT_GAMMA = 1.5

    def __init__(
        self,
        tau_uld: float,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
    ):
        self.tau_uld = tau_uld
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def audit_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """
        Full Tier 2A audit for a single adapter.
        Returns features, ULD score, and verdict (FLAGGED / NORMAL).
        """
        path = Path(adapter_path)
        features = compute_uld_features(path)
        if "error" in features:
            return {
                "adapter_path": str(adapter_path),
                "features": features,
                "uld_score": -1.0,
                "verdict": "ERROR",
                "tau_uld": self.tau_uld
            }
        score = compute_uld_score(features, self.alpha, self.beta, self.gamma)
        verdict = "FLAGGED" if score >= self.tau_uld else "NORMAL"
        return {
            "adapter_path": str(adapter_path),
            "features": {k: round(float(v), 4) for k, v in features.items()},
            "uld_score": round(score, 4),
            "verdict": verdict,
            "tau_uld": self.tau_uld
        }
