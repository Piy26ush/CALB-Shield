#!/usr/bin/env python3
"""
src/svd_scanner_v2.py
Phase 5: SecureLoRA v2 Rank-Adaptive SVD Spectral Scanner.

Addresses the limitation identified in Phase 5:
A fixed, flat threshold (tau = 0.40) falsely flags specialized benign narrow tasks (r=4)
where power-law decay naturally concentrates 50-60% energy in the top singular value.

Upgrade:
Replaces flat cutoff with a mathematically principled Rank-Adaptive Threshold:
    tau(r) = max(0.35, 1.35 / sqrt(r))

    - r=4:  tau(4)  = 0.675  (eliminates clean narrow task false alarm)
    - r=8:  tau(8)  = 0.477  (preserves sensitivity for math/reasoning tasks)
    - r=16: tau(16) = 0.350  (tightens bounds on standard instruction tuning)
    - r=64: tau(64) = 0.350  (prevents high-rank evasion)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from safetensors import safe_open

class RankAdaptiveSVDScanner:
    """
    Scans LoRA adapter weights with rank-calibrated spectral energy thresholds.
    """

    BASE_FLOOR = 0.35
    RANK_COEFFICIENT = 1.35

    def __init__(
        self,
        base_floor: float = BASE_FLOOR,
        rank_coefficient: float = RANK_COEFFICIENT,
        custom_threshold_func: Optional[Callable[[int], float]] = None
    ):
        self.base_floor = base_floor
        self.rank_coefficient = rank_coefficient
        self.custom_threshold_func = custom_threshold_func

    def get_rank_threshold(self, r: int) -> float:
        """Compute rank-adaptive threshold tau(r)."""
        if self.custom_threshold_func:
            return float(self.custom_threshold_func(r))
        if r <= 0:
            return 0.40
        return float(max(self.base_floor, self.rank_coefficient / np.sqrt(r)))

    def _find_lora_pairs(self, tensors: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
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
                    A = np.asarray(tensors[key], dtype=np.float64)
                    B = np.asarray(tensors[b_key], dtype=np.float64)
                    if A.ndim == 2 and B.ndim == 2:
                        pairs[base_key.strip(".")] = (A, B)
        return pairs

    def scan_adapter(self, adapter_path: str) -> Dict[str, Any]:
        path = Path(adapter_path)
        if path.is_dir():
            for candidate_name in ["adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"]:
                candidate = path / candidate_name
                if candidate.exists():
                    target_file = candidate
                    break
            else:
                files = list(path.glob("*.safetensors")) + list(path.glob("*.bin")) + list(path.glob("*.pt"))
                if files:
                    target_file = files[0]
                else:
                    return {
                        "verdict": "ERROR",
                        "reason": f"No adapter weight files found in: {adapter_path}",
                        "adapter_path": str(adapter_path)
                    }
            path = target_file

        if not path.exists():
            return {
                "verdict": "ERROR",
                "reason": f"Adapter file not found: {path}",
                "adapter_path": str(path)
            }

        tensors: Dict[str, np.ndarray] = {}
        try:
            if str(path).endswith(".safetensors"):
                with safe_open(str(path), framework="numpy") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
            else:
                import torch
                state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
                for k, v in state_dict.items():
                    if hasattr(v, "detach"):
                        tensors[k] = v.detach().cpu().numpy()
                    elif isinstance(v, np.ndarray):
                        tensors[k] = v
        except Exception as e:
            return {
                "verdict": "ERROR",
                "reason": f"Failed to open adapter file: {str(e)}",
                "adapter_path": str(path)
            }

        pairs = self._find_lora_pairs(tensors)
        if not pairs:
            return {
                "verdict": "ERROR",
                "reason": "No valid LoRA A/B matrix pairs detected",
                "adapter_path": str(path)
            }

        layer_results = []
        for layer_name, (A, B) in pairs.items():
            try:
                if B.shape[1] != A.shape[0]:
                    if B.shape[0] == A.shape[1]:
                        B, A = B.T, A.T
                    elif B.shape[1] == A.shape[1]:
                        A = A.T
                    elif B.shape[0] == A.shape[0]:
                        B = B.T

                rank = int(min(A.shape[0], A.shape[1], B.shape[0], B.shape[1]))
                tau_r = self.get_rank_threshold(rank)

                # Exact fast QR-SVD
                Q_B, R_B = np.linalg.qr(B)
                Q_A, R_A = np.linalg.qr(A.T)
                M = R_B @ R_A.T
                sigma = np.linalg.svd(M, compute_uv=False)
                
                energy = sigma ** 2
                total_energy = float(energy.sum())
                if total_energy == 0.0 or len(sigma) == 0:
                    continue

                top1_ratio = float(energy[0] / total_energy)
                top5_count = min(5, len(energy))
                top5_ratio = float(energy[:top5_count].sum() / total_energy)
                
                spectral_norm = float(sigma[0])
                frobenius_norm = float(np.sqrt(total_energy))
                cond_num = float(sigma[0] / (sigma[-1] + 1e-12))

                p = sigma / (sigma.sum() + 1e-12)
                spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)))
                effective_rank = float(np.exp(spectral_entropy))

                is_flagged = bool(top1_ratio > tau_r)

                layer_results.append({
                    "layer": layer_name,
                    "rank": rank,
                    "tau_r": tau_r,
                    "top1_spectral_energy_ratio": top1_ratio,
                    "top5_spectral_energy_ratio": top5_ratio,
                    "spectral_norm": spectral_norm,
                    "frobenius_norm": frobenius_norm,
                    "condition_number": cond_num,
                    "spectral_entropy": spectral_entropy,
                    "effective_rank": effective_rank,
                    "flagged": is_flagged
                })
            except Exception as e:
                layer_results.append({
                    "layer": layer_name,
                    "error": str(e)
                })

        valid_layers = [r for r in layer_results if "top1_spectral_energy_ratio" in r]
        if not valid_layers:
            return {
                "verdict": "ERROR",
                "reason": "SVD decomposition failed for all layers",
                "adapter_path": str(path)
            }

        ratios = [r["top1_spectral_energy_ratio"] for r in valid_layers]
        mean_ratio = float(np.mean(ratios))
        max_ratio = float(np.max(ratios))
        flagged_count = sum(1 for r in valid_layers if r["flagged"])
        flagged_fraction = float(flagged_count / len(valid_layers))

        spectral_norms = [r["spectral_norm"] for r in valid_layers]
        effective_ranks = [r["effective_rank"] for r in valid_layers]
        nominal_rank = valid_layers[0]["rank"]
        applied_threshold = self.get_rank_threshold(nominal_rank)

        verdict = "FLAGGED" if flagged_count > 0 else "NORMAL"

        return {
            "verdict": verdict,
            "adapter_path": str(path),
            "nominal_rank": nominal_rank,
            "rank_adaptive_threshold": applied_threshold,
            "n_layers_analyzed": len(valid_layers),
            "n_layers_flagged": flagged_count,
            "flagged_fraction": flagged_fraction,
            "max_top1_spectral_ratio": max_ratio,
            "mean_top1_spectral_ratio": mean_ratio,
            "max_spectral_norm": float(np.max(spectral_norms)),
            "mean_spectral_norm": float(np.mean(spectral_norms)),
            "min_effective_rank": float(np.min(effective_ranks)),
            "mean_effective_rank": float(np.mean(effective_ranks)),
            "layers": layer_results
        }
