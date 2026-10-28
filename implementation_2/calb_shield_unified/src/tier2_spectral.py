"""
implementation_2/calb_shield_unified/src/tier2_spectral.py

Tier 2: Architecture-Calibrated Rank-Aware Spectral Scanner.
Evaluates weight-space singular value distributions, effective rank ratio,
top-1 energy concentration, and spectral norm using QR-preconditioned SVD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

from implementation_2.calb_shield_unified.src.profiles import ArchitectureProfile


class Tier2SpectralScanner:
    """
    Computes architecture-calibrated QR-preconditioned SVD screening on low-rank weight matrices.
    """

    def scan_adapter(
        self,
        adapter_path: Path | str,
        profile: ArchitectureProfile,
    ) -> Dict[str, Any]:
        """
        Scan a LoRA adapter's serialized weight matrices against architecture profile thresholds.

        Args:
            adapter_path: Directory containing adapter weights.
            profile: Target ArchitectureProfile specifying the calibration envelope.

        Returns:
            Dictionary containing spectral metrics, anomaly diagnostics, and tier verdict.
        """
        target = Path(adapter_path)
        weights_dict = self._load_adapter_weights(target)
        if not weights_dict:
            return {
                "verdict": "ERROR",
                "reason": "Could not extract low-rank weight tensors from adapter directory",
                "per_layer_metrics": {},
            }

        # Load rank and alpha from config
        cfg_path = target / "adapter_config.json"
        alpha = 16.0
        r = 16
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                    r = cfg.get("r", 16)
                    alpha = float(cfg.get("lora_alpha", 2 * r))
            except Exception:
                pass

        scaling = alpha / float(r) if r > 0 else 1.0

        # Pair lora_A and lora_B modules
        layer_pairs = self._pair_lora_tensors(weights_dict)
        if not layer_pairs:
            return {
                "verdict": "ERROR",
                "reason": "No valid (lora_A, lora_B) tensor pairs found in adapter weights",
                "per_layer_metrics": {},
            }

        per_layer_metrics: Dict[str, Dict[str, float]] = {}
        anomalies: List[Dict[str, Any]] = []

        for layer_name, (w_a, w_b) in layer_pairs.items():
            metrics = self._compute_layer_svd(w_a, w_b, scaling=scaling)
            per_layer_metrics[layer_name] = metrics

            # Evaluate anomalies against profile envelope
            erank = metrics["erank_ratio"]
            top1 = metrics["top1_ratio"]
            snorm = metrics["spectral_norm"]

            # Rank-aware upper bound adjustment for finite small ranks (r <= 8)
            erank_max = profile.erank_ratio_max
            r_val = metrics.get("rank", 16)
            if r_val <= 8:
                erank_max = min(profile.erank_ratio_max + (0.04 / np.log(max(2, r_val))), 0.985)

            if erank > erank_max:
                anomalies.append({
                    "layer": layer_name,
                    "type": "DIFFUSE_OVERDISPERSION",
                    "severity": "CRITICAL",
                    "metric": "erank_ratio",
                    "value": erank,
                    "threshold": erank_max,
                    "reason": f"Diffuse over-dispersion: erank {erank:.3f} > max bound {erank_max:.3f}",
                })
            elif snorm > profile.max_spectral_norm:
                anomalies.append({
                    "layer": layer_name,
                    "type": "UNBOUNDED_WEIGHT_NORM",
                    "severity": "CRITICAL",
                    "metric": "spectral_norm",
                    "value": snorm,
                    "threshold": profile.max_spectral_norm,
                    "reason": f"Excessive spectral norm: {snorm:.1f} > max bound {profile.max_spectral_norm:.1f}",
                })
            elif erank < profile.erank_ratio_min:
                anomalies.append({
                    "layer": layer_name,
                    "type": "LOW_RANK_CONCENTRATION",
                    "severity": "QUARANTINE_WARNING",
                    "metric": "erank_ratio",
                    "value": erank,
                    "threshold": profile.erank_ratio_min,
                    "reason": f"Subspace energy concentration: erank {erank:.3f} < min bound {profile.erank_ratio_min:.3f}",
                })
            elif top1 > profile.top1_ratio_max:
                anomalies.append({
                    "layer": layer_name,
                    "type": "TOP1_CONCENTRATION",
                    "severity": "QUARANTINE_WARNING",
                    "metric": "top1_ratio",
                    "value": top1,
                    "threshold": profile.top1_ratio_max,
                    "reason": f"Dominant singular vector concentration: top1 {top1:.3f} > max bound {profile.top1_ratio_max:.3f}",
                })

        # Global aggregate metrics
        mean_erank = float(np.mean([m["erank_ratio"] for m in per_layer_metrics.values()]))
        mean_top1 = float(np.mean([m["top1_ratio"] for m in per_layer_metrics.values()]))
        max_snorm = float(max([m["spectral_norm"] for m in per_layer_metrics.values()], default=0.0))

        # Determine Tier 2 Verdict & Routing
        # Diffuse over-dispersion is CRITICAL if adapter's global mean erank ratio exceeds the calibrated max bound
        is_mean_diffuse = (mean_erank > profile.erank_ratio_max)

        unbounded_anomalies = [a for a in anomalies if a["type"] == "UNBOUNDED_WEIGHT_NORM"]
        quarantine_anomalies = [a for a in anomalies if a["type"] in ("LOW_RANK_CONCENTRATION", "TOP1_CONCENTRATION", "DIFFUSE_OVERDISPERSION")]

        if is_mean_diffuse:
            verdict = "REJECT_CRITICAL_SPECTRAL"
            primary_reason = f"Diffuse over-dispersion: mean erank ratio {mean_erank:.3f} > max bound {profile.erank_ratio_max:.3f}"
        elif unbounded_anomalies:
            verdict = "REJECT_CRITICAL_SPECTRAL"
            primary_reason = unbounded_anomalies[0]["reason"]
        elif quarantine_anomalies:
            verdict = "QUARANTINE_FLAGGED"
            primary_reason = quarantine_anomalies[0]["reason"]
        else:
            verdict = "CLEAN_SPECTRAL"
            primary_reason = f"Spectral distribution within calibrated normal envelope (erank={mean_erank:.3f}, top1={mean_top1:.3f})"

        return {
            "verdict": verdict,
            "primary_reason": primary_reason,
            "mean_erank_ratio": round(mean_erank, 4),
            "mean_top1_ratio": round(mean_top1, 4),
            "max_spectral_norm": round(max_snorm, 4),
            "anomalies_count": len(anomalies),
            "anomalies": anomalies,
            "per_layer_metrics": per_layer_metrics,
            "profile_used": profile.name,
        }

    def _load_adapter_weights(self, target_dir: Path) -> Dict[str, torch.Tensor]:
        """Loads weight tensors from safetensors or bin format."""
        st_file = target_dir / "adapter_model.safetensors"
        bin_file = target_dir / "adapter_model.bin"

        if st_file.exists():
            from safetensors.torch import load_file
            return load_file(str(st_file))
        elif bin_file.exists():
            return torch.load(str(bin_file), map_location="cpu")
        return {}

    def _pair_lora_tensors(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Pairs corresponding A and B matrices for each adapted projection."""
        pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, tensor in weights.items():
            base_key = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            base_key = base_key.replace(".lora_embedding_A", "").replace(".lora_embedding_B", "")
            if base_key not in pairs:
                pairs[base_key] = {}
            if "lora_A" in key:
                pairs[base_key]["A"] = tensor
            elif "lora_B" in key:
                pairs[base_key]["B"] = tensor

        complete_pairs = {}
        for k, p in pairs.items():
            if "A" in p and "B" in p:
                complete_pairs[k] = (p["A"], p["B"])
        return complete_pairs

    def _compute_layer_svd(self, w_a: torch.Tensor, w_b: torch.Tensor, scaling: float = 1.0) -> Dict[str, float]:
        """
        Computes exact fast low-rank SVD via QR preconditioning:
        Delta_W = scaling * B @ A = scaling * (Q_B @ R_B) @ (R_A.T @ Q_A.T)
        Singular values of Delta_W == Singular values of scaling * (R_B @ R_A.T) of size (r x r).
        """
        A = w_a.float().cpu().numpy()
        B = w_b.float().cpu().numpy()

        # Standardize orientation: B is (d_out, r), A is (r, d_in)
        if B.shape[1] != A.shape[0]:
            if B.shape[0] == A.shape[1]:
                B, A = B.T, A.T
            elif B.shape[1] == A.shape[1]:
                A = A.T
            elif B.shape[0] == A.shape[0]:
                B = B.T

        d_out, r_B = B.shape
        r_A, d_in = A.shape
        r = r_A

        if r == 0:
            return {"erank_ratio": 0.0, "top1_ratio": 0.0, "spectral_norm": 0.0}

        # Exact fast low-rank SVD:
        # Q_B is (d_out, r), R_B is (r, r)
        # Q_A is (d_in, r), R_A is (r, r)
        Q_B, R_B = np.linalg.qr(B)
        Q_A, R_A = np.linalg.qr(A.T)

        M = scaling * (R_B @ R_A.T)
        s_np = np.linalg.svd(M, compute_uv=False)
        s_np = np.sort(s_np)[::-1]

        sum_s = float(np.sum(s_np))
        if sum_s <= 1e-12:
            return {"erank_ratio": 0.0, "top1_ratio": 0.0, "spectral_norm": 0.0}

        p = s_np / sum_s
        p_nonzero = p[p > 1e-12]
        entropy = -float(np.sum(p_nonzero * np.log(p_nonzero)))
        erank = float(np.exp(entropy))

        return {
            "erank_ratio": float(erank / r) if r > 0 else 0.0,
            "top1_ratio": float(s_np[0] / sum_s),
            "spectral_norm": float(s_np[0]),
            "rank": r,
        }
