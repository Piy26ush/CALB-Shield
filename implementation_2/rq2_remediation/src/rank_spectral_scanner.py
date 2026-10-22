#!/usr/bin/env python3
"""
implementation_2/rq2_remediation/src/rank_spectral_scanner.py

Remediated Stage 2: Rank-Aware Spectral Scanner for PEFT/LoRA Adapters.
Part of CALB-Shield RQ2 Remediation Pipeline.

Scientific Rationale:
--------------------
Prior implementations hardcoded a static Top-1 energy threshold (e.g. 0.40) that
fails across diverse ranks:
- Rank-2/4 clean adapters naturally concentrate ~50-60% energy in the top singular value,
  producing false positives.
- Rank-16 multi-rank trojans (e.g., k=4 backdoors) distribute energy across 4 singular values,
  yielding Top-1 ratio ~0.25 and evading fixed-threshold detection.

This remediated scanner implements rank-aware spectral features:
1. Normalized Singular Value Entropy: H_norm = H(sigma) / ln(r) in [0, 1]
2. Normalized Spectral Entropy Deficit: NSED = 1 - H_norm
3. Effective Rank Ratio: erank_ratio = exp(H(sigma)) / r in [1/r, 1.0]
4. Rank-Scaled Top-1 Energy: rho_1_scaled = r * (sigma_1^2 / sum(sigma^2))
5. Stable Rank Ratio: srank_ratio = (||W||_F^2 / ||W||_2^2) / r

IMPORTANT: These spectral metrics are treated as EXPERIMENTAL HYPOTHESES, not established
panaceas. Spectral signatures flag low-effective-rank anomalies, but do not replace
behavioral and differential inference.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

try:
    from safetensors import safe_open
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


class RankAwareSpectralScanner:
    """
    Computes rank-aware spectral properties of LoRA weight updates.
    """

    # Calibrated defaults from development cohort (functional_cohort + physical_cohort dev):
    # Calibrated defaults from development cohort:
    # Anomaly hypothesis: Flag if min_erank_ratio < 0.55 (low-effective-rank anomaly)
    # or min_erank_ratio > 0.88 (unnatural diffuse over-dispersion anomaly)
    DEFAULT_ERANK_RATIO_THRESHOLD = 0.55
    DEFAULT_ERANK_MAX_THRESHOLD = 0.88
    DEFAULT_NORM_ENTROPY_THRESHOLD = 0.70

    def __init__(
        self,
        erank_ratio_threshold: float = DEFAULT_ERANK_RATIO_THRESHOLD,
        erank_max_threshold: Optional[float] = DEFAULT_ERANK_MAX_THRESHOLD,
        norm_entropy_threshold: float = DEFAULT_NORM_ENTROPY_THRESHOLD,
        use_entropy_decision: bool = False,
    ):
        self.erank_ratio_threshold = float(erank_ratio_threshold)
        self.erank_max_threshold = float(erank_max_threshold) if erank_max_threshold is not None else None
        self.norm_entropy_threshold = float(norm_entropy_threshold)
        self.use_entropy_decision = bool(use_entropy_decision)

    @staticmethod
    def compute_layer_spectral_metrics(
        A: np.ndarray,
        B: np.ndarray,
        eps: float = 1e-12,
    ) -> Dict[str, float]:
        """
        Compute rank-aware spectral metrics for a single LoRA layer pair (A, B).

        Args:
            A: LoRA down/in matrix (shape [r, d_in] or transpose)
            B: LoRA up/out matrix (shape [d_out, r] or transpose)
            eps: Numerical stability constant

        Returns:
            Dictionary of computed spectral features.
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        # Standardize orientation: B has shape (d_out, r), A has shape (r, d_in)
        if B.shape[1] != A.shape[0]:
            if B.shape[0] == A.shape[1]:
                B, A = B.T, A.T
            elif B.shape[1] == A.shape[1]:
                A = A.T
            elif B.shape[0] == A.shape[0]:
                B = B.T
            else:
                raise ValueError(f"Incompatible LoRA dimensions: A={A.shape}, B={B.shape}")

        d_out, r_B = B.shape
        r_A, d_in = A.shape
        if r_A != r_B:
            raise ValueError(f"Inner rank mismatch: r_A={r_A}, r_B={r_B}")
        r = r_A

        if r == 0:
            return {"rank": 0, "error": "Zero rank matrix"}

        # Exact fast low-rank SVD via QR decomposition:
        # Delta W = B @ A = (Q_B @ R_B) @ (R_A.T @ Q_A.T)
        # Singular values of Delta W == Singular values of (R_B @ R_A.T) of size (r x r)
        Q_B, R_B = np.linalg.qr(B)
        Q_A, R_A = np.linalg.qr(A.T)
        M = R_B @ R_A.T
        sigma = np.linalg.svd(M, compute_uv=False)
        sigma = np.sort(sigma)[::-1]  # descending

        energy = sigma ** 2
        total_energy = float(np.sum(energy))

        if total_energy < eps:
            return {
                "rank": r,
                "total_energy": 0.0,
                "top1_ratio": 0.0,
                "rank_scaled_top1": 0.0,
                "entropy": float(np.log(r)) if r > 1 else 0.0,
                "norm_entropy": 1.0,
                "sed": 0.0,
                "nsed": 0.0,
                "erank": float(r),
                "erank_ratio": 1.0,
                "stable_rank": float(r),
                "srank_ratio": 1.0,
                "spectral_norm": 0.0,
                "frobenius_norm": 0.0,
                "condition_number": 1.0,
            }

        p = energy / total_energy
        entropy = float(-np.sum(p * np.log(p + eps)))

        max_entropy = float(np.log(r)) if r > 1 else 1.0
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 1.0
        sed = float(max_entropy - entropy) if r > 1 else 0.0
        nsed = float(1.0 - norm_entropy)

        erank = float(np.exp(entropy))
        erank_ratio = float(erank / r) if r > 0 else 0.0

        top1_ratio = float(energy[0] / total_energy)
        rank_scaled_top1 = float(r * top1_ratio)

        frobenius_norm = float(np.sqrt(total_energy))
        spectral_norm = float(sigma[0])
        stable_rank = float(total_energy / (spectral_norm ** 2 + eps))
        srank_ratio = float(stable_rank / r) if r > 0 else 0.0

        cond_num = float(sigma[0] / (sigma[-1] + eps)) if len(sigma) > 0 else 1.0

        return {
            "rank": r,
            "total_energy": total_energy,
            "top1_ratio": top1_ratio,
            "rank_scaled_top1": rank_scaled_top1,
            "entropy": entropy,
            "norm_entropy": norm_entropy,
            "sed": sed,
            "nsed": nsed,
            "erank": erank,
            "erank_ratio": erank_ratio,
            "stable_rank": stable_rank,
            "srank_ratio": srank_ratio,
            "spectral_norm": spectral_norm,
            "frobenius_norm": frobenius_norm,
            "condition_number": cond_num,
        }

    def _find_lora_pairs(self, tensors: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Identify matching LoRA A and B weight matrices in tensor dictionary."""
        pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for key in tensors:
            if "lora_A" in key or "lora_down" in key:
                if "lora_A" in key:
                    base_key = key.replace("lora_A.weight", "").replace("lora_A", "")
                    b_key = key.replace("lora_A", "lora_B")
                else:
                    base_key = key.replace("lora_down.weight", "").replace("lora_down", "")
                    b_key = key.replace("lora_down", "lora_up")

                if b_key in tensors:
                    pairs[base_key.strip(".")] = (tensors[key], tensors[b_key])

        return pairs

    def _load_tensors(self, path: Path) -> Dict[str, np.ndarray]:
        """Load tensor dictionary from safetensors or PyTorch binary file."""
        tensors: Dict[str, np.ndarray] = {}
        str_path = str(path)

        if str_path.endswith(".safetensors"):
            if not _HAS_SAFETENSORS:
                raise ImportError("safetensors library is required to open .safetensors files")
            with safe_open(str_path, framework="numpy") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
        else:
            import torch
            state_dict = torch.load(str_path, map_location="cpu", weights_only=True)
            for k, v in state_dict.items():
                if hasattr(v, "detach"):
                    tensors[k] = v.detach().cpu().numpy()
                elif isinstance(v, np.ndarray):
                    tensors[k] = v

        return tensors

    def scan_adapter(self, adapter_path: str | Path) -> Dict[str, Any]:
        """
        Scan a serialized LoRA adapter and compute aggregated rank-aware spectral metrics.

        Args:
            adapter_path: Path to adapter directory or weight file (.safetensors, .bin, .pt).

        Returns:
            Dictionary containing:
            - verdict: 'FLAGGED' or 'NORMAL' (based on calibrated anomaly threshold)
            - rank_aware_metrics: Layer-aggregated metrics
            - per_layer_metrics: Detailed breakdown per layer
        """
        target = Path(adapter_path)
        if target.is_dir():
            for name in ["adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"]:
                candidate = target / name
                if candidate.exists():
                    target = candidate
                    break
            else:
                files = list(target.glob("*.safetensors")) + list(target.glob("*.bin")) + list(target.glob("*.pt"))
                if files:
                    target = files[0]
                else:
                    return {
                        "verdict": "ERROR",
                        "reason": f"No adapter weight files (.safetensors, .bin, .pt) found in: {adapter_path}",
                        "adapter_path": str(adapter_path),
                    }

        if not target.exists():
            return {
                "verdict": "ERROR",
                "reason": f"Adapter file not found: {target}",
                "adapter_path": str(target),
            }

        try:
            tensors = self._load_tensors(target)
        except Exception as e:
            return {
                "verdict": "ERROR",
                "reason": f"Failed to load adapter weights: {e}",
                "adapter_path": str(target),
            }

        pairs = self._find_lora_pairs(tensors)
        if not pairs:
            return {
                "verdict": "ERROR",
                "reason": "No valid LoRA A/B matrix pairs detected in file",
                "adapter_path": str(target),
            }

        layer_results = {}
        for layer_name, (A, B) in pairs.items():
            try:
                metrics = self.compute_layer_spectral_metrics(A, B)
                layer_results[layer_name] = metrics
            except Exception as e:
                layer_results[layer_name] = {"error": str(e)}

        valid_layers = [m for m in layer_results.values() if "erank_ratio" in m]
        if not valid_layers:
            return {
                "verdict": "ERROR",
                "reason": "Spectral decomposition failed for all layers",
                "adapter_path": str(target),
            }

        erank_ratios = [m["erank_ratio"] for m in valid_layers]
        norm_entropies = [m["norm_entropy"] for m in valid_layers]
        nseds = [m["nsed"] for m in valid_layers]
        top1_ratios = [m["top1_ratio"] for m in valid_layers]
        scaled_top1s = [m["rank_scaled_top1"] for m in valid_layers]
        ranks = [m["rank"] for m in valid_layers]
        cond_nums = [m["condition_number"] for m in valid_layers]

        mean_erank_ratio = float(np.mean(erank_ratios))
        min_erank_ratio = float(np.min(erank_ratios))
        mean_norm_entropy = float(np.mean(norm_entropies))
        min_norm_entropy = float(np.min(norm_entropies))
        mean_nsed = float(np.mean(nseds))
        max_nsed = float(np.max(nseds))
        mean_top1_ratio = float(np.mean(top1_ratios))
        max_top1_ratio = float(np.max(top1_ratios))
        mean_scaled_top1 = float(np.mean(scaled_top1s))
        max_scaled_top1 = float(np.max(scaled_top1s))
        mean_cond = float(np.mean(cond_nums))

        max_fro = float(np.max([m.get("frobenius_norm", 0.0) for m in valid_layers])) if valid_layers else 0.0

        # Two-Sided Spectral Anomaly Hypothesis:
        # 1. Low-rank concentration: min_erank_ratio < erank_ratio_threshold (e.g. < 0.55)
        # 2. Diffuse over-dispersion: min_erank_ratio > erank_max_threshold (e.g. > 0.88, with active non-zero update max_fro > 0.01)
        low_rank_anomaly = False
        diffuse_anomaly = False
        anomaly_type = "NORMAL"
        anomaly_reason = "Normal spectral distribution"

        if max_fro < 1e-6:
            # Identity or untrained zero-weight adapter: no perturbation to base model
            flagged = False
            anomaly_type = "ZERO_UPDATE_IDENTITY"
            anomaly_reason = "Zero weight perturbation (identity adapter)"
            decision_feature = "erank_ratio_bounds"
            threshold_used = {"lower": self.erank_ratio_threshold, "upper": self.erank_max_threshold}
        elif self.use_entropy_decision:
            flagged = bool(min_norm_entropy < self.norm_entropy_threshold)
            decision_feature = "min_norm_entropy"
            threshold_used = self.norm_entropy_threshold
            if flagged:
                anomaly_type = "LOW_ENTROPY"
                anomaly_reason = f"Anomalously low normalized entropy: {min_norm_entropy:.3f} < {self.norm_entropy_threshold}"
        else:
            low_rank_anomaly = bool(min_erank_ratio < self.erank_ratio_threshold)
            diffuse_anomaly = bool(self.erank_max_threshold is not None and min_erank_ratio > self.erank_max_threshold and max_fro > 0.01)
            flagged = low_rank_anomaly or diffuse_anomaly
            decision_feature = "erank_ratio_bounds"
            threshold_used = {"lower": self.erank_ratio_threshold, "upper": self.erank_max_threshold}
            if low_rank_anomaly:
                anomaly_type = "LOW_RANK_SUBSPACE"
                anomaly_reason = f"Low-rank subspace concentration: {min_erank_ratio:.3f} < {self.erank_ratio_threshold}"
            elif diffuse_anomaly:
                anomaly_type = "DIFFUSE_OVERDISPERSION"
                anomaly_reason = f"Unnatural diffuse over-dispersion: {min_erank_ratio:.3f} > {self.erank_max_threshold}"

        verdict = "FLAGGED" if flagged else "NORMAL"

        return {
            "verdict": verdict,
            "anomaly_type": anomaly_type,
            "anomaly_reason": anomaly_reason,
            "adapter_path": str(target),
            "n_layers_analyzed": len(valid_layers),
            "nominal_rank": int(min(ranks)),
            "decision_feature": decision_feature,
            "threshold_used": threshold_used,
            "summary_metrics": {
                "mean_erank_ratio": mean_erank_ratio,
                "min_erank_ratio": min_erank_ratio,
                "mean_norm_entropy": mean_norm_entropy,
                "min_norm_entropy": min_norm_entropy,
                "mean_nsed": mean_nsed,
                "max_nsed": max_nsed,
                "mean_top1_ratio": mean_top1_ratio,
                "max_top1_ratio": max_top1_ratio,
                "mean_scaled_top1": mean_scaled_top1,
                "max_scaled_top1": max_scaled_top1,
                "mean_condition_number": mean_cond,
            },
            "per_layer_metrics": layer_results,
        }

    @staticmethod
    def calibrate_threshold(
        clean_adapter_paths: List[str | Path],
        poison_adapter_paths: List[str | Path],
        feature: str = "min_erank_ratio",
    ) -> Dict[str, Any]:
        """
        Calibrate anomaly threshold strictly on development cohort data.
        Finds the threshold that maximizes Youden's J statistic (TPR - FPR).

        Args:
            clean_adapter_paths: Paths to dev clean adapters (ground truth = NORMAL)
            poison_adapter_paths: Paths to dev poison adapters (ground truth = FLAGGED)
            feature: Metric to calibrate ('min_erank_ratio' or 'min_norm_entropy')

        Returns:
            Dictionary containing optimal threshold, best Youden's J, TPR, FPR, and data points.
        """
        temp_scanner = RankAwareSpectralScanner()

        clean_vals = []
        for p in clean_adapter_paths:
            res = temp_scanner.scan_adapter(p)
            if res.get("verdict") != "ERROR":
                clean_vals.append((str(p), res["summary_metrics"][feature]))

        poison_vals = []
        for p in poison_adapter_paths:
            res = temp_scanner.scan_adapter(p)
            if res.get("verdict") != "ERROR":
                poison_vals.append((str(p), res["summary_metrics"][feature]))

        if not clean_vals or not poison_vals:
            raise ValueError("Insufficient data points for calibration")

        c_vals = [v for _, v in clean_vals]
        p_vals = [v for _, v in poison_vals]

        # Candidate thresholds: sweep from min to max
        all_vals = sorted(c_vals + p_vals)
        best_threshold = 0.50
        best_j = -1.0
        best_tpr = 0.0
        best_fpr = 1.0

        for i in range(len(all_vals) - 1):
            cand = (all_vals[i] + all_vals[i+1]) / 2.0
            # Condition for FLAGGED: val < cand
            tpr = sum(1 for v in p_vals if v < cand) / len(p_vals)
            fpr = sum(1 for v in c_vals if v < cand) / len(c_vals)
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_threshold = cand
                best_tpr = tpr
                best_fpr = fpr

        return {
            "calibrated_feature": feature,
            "optimal_threshold": float(best_threshold),
            "youdens_j": float(best_j),
            "dev_tpr": float(best_tpr),
            "dev_fpr": float(best_fpr),
            "dev_balanced_accuracy": float(0.5 * (best_tpr + (1.0 - best_fpr))),
            "n_clean_dev": len(c_vals),
            "n_poison_dev": len(p_vals),
            "clean_values": clean_vals,
            "poison_values": poison_vals,
        }
