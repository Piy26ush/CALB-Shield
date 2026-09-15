#!/usr/bin/env python3
"""
src/svd_scanner.py
Phase 2: Spectral SVD Scanner for LoRA Adapters (Stage 2 Admission Control).

Addresses RQ2:
Examines the singular value spectrum of LoRA low-rank delta weight matrices:
    Delta W = B @ A
Computes spectral energy concentration:
    Top1_Energy_Ratio = sigma_1^2 / sum(sigma_i^2)

Prior work (e.g. PEFTGuard, IEEE S&P 2025) indicates that backdoored or safety-stripped
adapters often exhibit unnatural spectral concentration in the top singular direction.
Verdicts are framed as 'FLAGGED' or 'NORMAL' (anomaly detection hypothesis, not definitive attribution).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from safetensors import safe_open

class SVDSpectralScanner:
    """
    Scans LoRA adapter .safetensors files for anomalous spectral energy concentration.
    """

    INITIAL_THRESHOLD = 0.40  # Starting hypothesis cutoff for rank-16 (to be empirically tuned)

    def __init__(self, threshold: float = INITIAL_THRESHOLD):
        self.threshold = threshold

    def _find_lora_pairs(self, tensors: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Identify matching LoRA A and B weight matrices in a dictionary of tensors.
        Supports standard Hugging Face PEFT naming patterns:
        - `lora_A` / `lora_B`
        - `lora_down` / `lora_up`
        """
        pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Look for lora_A / lora_B patterns
        for key in tensors:
            if "lora_A" in key or "lora_down" in key:
                # Deduce base module key and corresponding B key
                if "lora_A" in key:
                    base_key = key.replace("lora_A.weight", "").replace("lora_A", "")
                    b_key = key.replace("lora_A", "lora_B")
                else:
                    base_key = key.replace("lora_down.weight", "").replace("lora_down", "")
                    b_key = key.replace("lora_down", "lora_up")

                if b_key in tensors:
                    A = tensors[key]
                    B = tensors[b_key]
                    
                    # Convert to 2D float64
                    A = np.asarray(A, dtype=np.float64)
                    B = np.asarray(B, dtype=np.float64)
                    
                    if A.ndim == 2 and B.ndim == 2:
                        pairs[base_key.strip(".")] = (A, B)

        return pairs

    def _compute_delta_w(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiply low-rank matrices B and A to reconstruct Delta W.
        Standard LoRA convention:
        A has shape (r, d_in), B has shape (d_out, r) -> Delta W = B @ A of shape (d_out, d_in).
        If transposed, automatically aligns matrix inner dimensions.
        """
        if B.shape[1] == A.shape[0]:
            return B @ A
        elif B.shape[0] == A.shape[1]:
            return B.T @ A.T
        elif B.shape[1] == A.shape[1]:
            return B @ A.T
        elif B.shape[0] == A.shape[0]:
            return B.T @ A
        else:
            raise ValueError(f"Incompatible LoRA matrix shapes: A={A.shape}, B={B.shape}")

    def scan_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """
        Scan a .safetensors adapter file and calculate spectral properties across layers.

        Args:
            adapter_path: Path to adapter_model.safetensors or directory containing it.

        Returns:
            Dictionary containing layer-by-layer metrics, summary statistics, and verdict.
        """
        path = Path(adapter_path)
        if path.is_dir():
            # Check for safetensors first, then bin / pt
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
                        "reason": f"No adapter weight files (.safetensors, .bin, .pt) found in directory: {adapter_path}",
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
                "reason": f"Failed to open adapter weight file: {str(e)}",
                "adapter_path": str(path)
            }

        pairs = self._find_lora_pairs(tensors)
        if not pairs:
            return {
                "verdict": "ERROR",
                "reason": "No valid LoRA A/B matrix pairs detected in file",
                "adapter_path": str(path),
                "tensor_keys": list(tensors.keys())[:10]
            }

        layer_results = []
        for layer_name, (A, B) in pairs.items():
            try:
                # Ensure standard LoRA orientation: A is (r, d_in), B is (d_out, r)
                if B.shape[1] != A.shape[0]:
                    if B.shape[0] == A.shape[1]:
                        B, A = B.T, A.T
                    elif B.shape[1] == A.shape[1]:
                        A = A.T
                    elif B.shape[0] == A.shape[0]:
                        B = B.T

                # Exact fast low-rank SVD via QR decomposition:
                # Delta W = B @ A = (Q_B @ R_B) @ (R_A.T @ Q_A.T) = Q_B @ (R_B @ R_A.T) @ Q_A.T
                # The singular values of Delta W are identical to the singular values of M = R_B @ R_A.T (size r x r)
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
                
                # Condition number and norms
                cond_num = float(sigma[0] / (sigma[-1] + 1e-12))
                frobenius_norm = float(np.sqrt(total_energy))
                spectral_norm = float(sigma[0])

                layer_results.append({
                    "layer": layer_name,
                    "rank": int(min(A.shape[0], A.shape[1], B.shape[0], B.shape[1])),
                    "top1_spectral_energy_ratio": top1_ratio,
                    "top5_spectral_energy_ratio": top5_ratio,
                    "spectral_norm": spectral_norm,
                    "frobenius_norm": frobenius_norm,
                    "condition_number": cond_num,
                    "sigma_mean": float(sigma.mean()),
                    "sigma_std": float(sigma.std()),
                    "flagged": top1_ratio > self.threshold
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
                "reason": "SVD decomposition failed for all detected layers",
                "adapter_path": str(path)
            }

        ratios = [r["top1_spectral_energy_ratio"] for r in valid_layers]
        mean_ratio = float(np.mean(ratios))
        max_ratio = float(np.max(ratios))
        std_ratio = float(np.std(ratios))
        flagged_count = sum(1 for r in valid_layers if r["flagged"])
        flagged_fraction = float(flagged_count / len(valid_layers))

        verdict = "FLAGGED" if max_ratio > self.threshold else "NORMAL"

        return {
            "verdict": verdict,
            "adapter_path": str(path),
            "threshold_used": self.threshold,
            "n_layers_analyzed": len(valid_layers),
            "n_layers_flagged": flagged_count,
            "flagged_fraction": flagged_fraction,
            "max_top1_spectral_ratio": max_ratio,
            "mean_top1_spectral_ratio": mean_ratio,
            "std_top1_spectral_ratio": std_ratio,
            "layers": layer_results
        }

    def scan_synthetic_matrix(self, A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
        """Convenience method for scanning raw in-memory numpy matrices (useful in unit testing)."""
        delta_W = self._compute_delta_w(A, B)
        _, sigma, _ = np.linalg.svd(delta_W, full_matrices=False)
        energy = sigma ** 2
        total_energy = float(energy.sum())
        top1_ratio = float(energy[0] / total_energy) if total_energy > 0 else 0.0
        return {
            "top1_spectral_energy_ratio": top1_ratio,
            "spectral_norm": float(sigma[0]),
            "condition_number": float(sigma[0] / (sigma[-1] + 1e-12)),
            "verdict": "FLAGGED" if top1_ratio > self.threshold else "NORMAL"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan LoRA adapter for anomalous spectral energy distribution.")
    parser.add_argument("adapter_path", help="Path to adapter .safetensors file or folder")
    parser.add_argument("--threshold", type=float, default=0.40, help="Top-1 energy concentration threshold (default: 0.40)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    args = parser.parse_args()

    scanner = SVDSpectralScanner(threshold=args.threshold)
    results = scanner.scan_adapter(args.adapter_path)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 60)
        print(" CALB-Shield Stage 2: SVD Spectral Scanner Report")
        print("=" * 60)
        print(f"Adapter:           {results.get('adapter_path')}")
        print(f"Verdict:           {results.get('verdict')}")
        if results.get("verdict") != "ERROR":
            print(f"Threshold:         {results.get('threshold_used'):.2f}")
            print(f"Layers Analyzed:   {results.get('n_layers_analyzed')}")
            print(f"Flagged Layers:    {results.get('n_layers_flagged')} ({results.get('flagged_fraction')*100:.1f}%)")
            print(f"Max Top-1 Ratio:   {results.get('max_top1_spectral_ratio'):.4f}")
            print(f"Mean Top-1 Ratio:  {results.get('mean_top1_spectral_ratio'):.4f}")
        else:
            print(f"Error Reason:      {results.get('reason')}")
        print("=" * 60 + "\n")
