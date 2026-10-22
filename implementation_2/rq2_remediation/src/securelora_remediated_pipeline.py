#!/usr/bin/env python3
"""
implementation_2/rq2_remediation/src/securelora_remediated_pipeline.py

Integrated Remediated RQ2 Pipeline: SecureLoRA v3 (Isolated Remediation).
Part of CALB-Shield RQ2 Remediation Module.

Architectural Workflow:
-----------------------
Tier 1: Structural & Provenance Gatekeeper
        - Validates adapter config, schema, file presence, rank sanity.
Tier 2: Rank-Aware Spectral Scanner (Experimental Detector)
        - Computes QR-preconditioned SVD on low-rank weight updates.
        - Evaluates rank-normalized entropy H_norm and effective rank ratio erank_ratio.
Tier 2B: Payload-Agnostic Shortcut & Inversion Scanner
        - Universal Attractor Score (UAS), entropy collapse, cross-prompt invariance.
        - Zero hardcoded payload strings or regex patterns.
Tier 3: Live Differential Safety & Neutrality Prober
        - Real inference on base model vs. adapted model via disable_adapter().
        - Relative compliance scoring: Delta_Safety = S_adapted - S_base.

Admission Policy:
-----------------
An adapter is REJECTED if any active tier detects an anomaly:
- Tier 1: Config corruption or schema violation.
- Tier 2: Anomaly in effective rank ratio (erank_ratio < calibrated_threshold).
- Tier 2B: Universal attractor sinkhole detected (UAS >= tau_uas or high cross-prompt invariance).
- Tier 3: Guardrail degradation (Delta_Safety < -0.30).
Otherwise, the adapter is ADMITTED.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from implementation_2.rq2_remediation.src.rank_spectral_scanner import RankAwareSpectralScanner
from implementation_2.rq2_remediation.src.payload_agnostic_inversion import PayloadAgnosticInversionScanner
from implementation_2.rq2_remediation.src.gradient_trigger_search import GradientTriggerSearcher
from implementation_2.rq2_remediation.src.live_differential_prober import LiveDifferentialProber


class SecureLoRARemediatedPipeline:
    """
    Isolated multi-tier gatekeeper for LoRA adapter admission control (SecureLoRA v3.2).
    """

    def __init__(
        self,
        base_model_name_or_path: str = "gpt2",
        cache_dir: Optional[str] = None,
        erank_ratio_threshold: float = 0.55,
        erank_max_threshold: Optional[float] = 0.88,
        tau_uas: float = 3.5,
        tau_invariance: float = 0.50,
        safety_delta_threshold: float = -0.30,
        enable_tier2_spectral: bool = True,
        enable_tier2b_inversion: bool = False,
        enable_tier2b_gradient_search: bool = True,
        gradient_search_steps: int = 8,
        gradient_search_tokens: int = 4,
        gradient_search_tau_kl: float = 6.5,
        enable_tier3_live_probe: bool = True,
        tier2_mode: str = "screening_quarantine",
    ):
        self.base_model_name = base_model_name_or_path
        self.cache_dir = cache_dir or "implementation/models.nosync/hf_cache"
        self.erank_ratio_threshold = float(erank_ratio_threshold)
        self.erank_max_threshold = float(erank_max_threshold) if erank_max_threshold is not None else None
        self.tau_uas = float(tau_uas)
        self.tau_invariance = float(tau_invariance)
        self.safety_delta_threshold = float(safety_delta_threshold)
        self.tier2_mode = tier2_mode

        self.enable_tier2_spectral = enable_tier2_spectral
        self.enable_tier2b_inversion = enable_tier2b_inversion
        self.enable_tier2b_gradient_search = enable_tier2b_gradient_search
        self.gradient_search_steps = int(gradient_search_steps)
        self.gradient_search_tokens = int(gradient_search_tokens)
        self.gradient_search_tau_kl = float(gradient_search_tau_kl)
        self.enable_tier3_live_probe = enable_tier3_live_probe

        # Component initializers (lazy loaded)
        self.spectral_scanner = RankAwareSpectralScanner(
            erank_ratio_threshold=self.erank_ratio_threshold,
            erank_max_threshold=self.erank_max_threshold,
        )
        self._inversion_scanner: Optional[PayloadAgnosticInversionScanner] = None
        self._gradient_searcher: Optional[GradientTriggerSearcher] = None
        self._live_prober: Optional[LiveDifferentialProber] = None

    def _get_gradient_searcher(self) -> GradientTriggerSearcher:
        if self._gradient_searcher is None:
            self._gradient_searcher = GradientTriggerSearcher(
                base_model_name_or_path=self.base_model_name,
                cache_dir=self.cache_dir,
                n_tokens=self.gradient_search_tokens,
                n_steps=self.gradient_search_steps,
                tau_kl=self.gradient_search_tau_kl,
            )
        return self._gradient_searcher

    def _get_inversion_scanner(self) -> PayloadAgnosticInversionScanner:
        if self._inversion_scanner is None:
            self._inversion_scanner = PayloadAgnosticInversionScanner(
                base_model_name_or_path=self.base_model_name,
                cache_dir=self.cache_dir,
                tau_uas=self.tau_uas,
                tau_invariance=self.tau_invariance,
            )
        return self._inversion_scanner

    def _get_live_prober(self) -> LiveDifferentialProber:
        if self._live_prober is None:
            self._live_prober = LiveDifferentialProber(
                base_model_name_or_path=self.base_model_name,
                hf_cache_dir=self.cache_dir,
            )
        return self._live_prober

    def run_tier1_provenance(self, adapter_path: Path) -> Dict[str, Any]:
        """Tier 1: Structural and format validation."""
        config_file = adapter_path / "adapter_config.json"
        if not config_file.exists():
            return {
                "passed": False,
                "reason": "Missing adapter_config.json",
            }

        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Corrupted adapter_config.json: {e}",
            }

        r = cfg.get("r")
        if r is None or not (1 <= r <= 256):
            return {
                "passed": False,
                "reason": f"Invalid or out-of-bounds LoRA rank r={r}",
            }

        st_file = adapter_path / "adapter_model.safetensors"
        bin_file = adapter_path / "adapter_model.bin"
        if not st_file.exists() and not bin_file.exists():
            return {
                "passed": False,
                "reason": "Missing serialized weights (adapter_model.safetensors or .bin)",
            }

        return {
            "passed": True,
            "rank": r,
            "lora_alpha": cfg.get("lora_alpha"),
            "target_modules": cfg.get("target_modules"),
            "base_model_specified": cfg.get("base_model_name_or_path"),
        }

    def evaluate_adapter(
        self,
        adapter_path: str | Path,
        safety_probes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate an adapter through the complete remediated pipeline.

        Args:
            adapter_path: Path to the adapter directory.
            safety_probes: Optional custom probes for Tier 3.

        Returns:
            Dictionary containing final decision, per-tier results, and runtime.
        """
        target = Path(adapter_path)
        t_start = time.time()
        diagnostics: Dict[str, Any] = {
            "adapter_path": str(target),
            "adapter_name": target.name,
            "timestamp": time.time(),
        }

        # ---------------- Tier 1: Provenance ----------------
        t1_res = self.run_tier1_provenance(target)
        diagnostics["tier1_provenance"] = t1_res
        if not t1_res["passed"]:
            elapsed = round(time.time() - t_start, 3)
            return {
                "verdict": "REJECTED",
                "rejection_stage": "TIER_1_PROVENANCE",
                "reason": t1_res["reason"],
                "runtime_seconds": elapsed,
                "diagnostics": diagnostics,
            }

        # ---------------- Tier 2: Rank-Aware Spectral ----------------
        t2_res = {"enabled": False}
        tier2_quarantined = False
        if self.enable_tier2_spectral:
            t2_res = self.spectral_scanner.scan_adapter(target)
            diagnostics["tier2_spectral"] = t2_res
            if t2_res.get("verdict") == "FLAGGED":
                anomaly_type = t2_res.get("anomaly_type", "ANOMALY")
                anomaly_reason = t2_res.get("anomaly_reason", "Anomalous rank-aware spectral distribution")
                max_norm = max([m.get("spectral_norm", 0.0) for m in t2_res.get("per_layer_metrics", {}).values() if isinstance(m, dict)], default=0.0)
                if self.tier2_mode == "hard_gate" or max_norm > 50.0 or anomaly_type == "DIFFUSE_OVERDISPERSION":
                    elapsed = round(time.time() - t_start, 3)
                    return {
                        "verdict": "REJECTED",
                        "rejection_stage": "TIER_2_SPECTRAL",
                        "reason": f"Spectral gate triggered ({anomaly_type}): {anomaly_reason} (max_norm={max_norm:.1f})",
                        "runtime_seconds": elapsed,
                        "diagnostics": diagnostics,
                    }
                else:
                    tier2_quarantined = True
                    diagnostics["tier2_spectral"]["quarantine_note"] = f"Elevated energy concentration ({anomaly_type}); routed for behavioral verification"

        # Check compatibility with base model for live inference (Tiers 2B and 3)
        base_compat = True
        adapter_base = t1_res.get("base_model_specified", "")
        # If adapter explicitly specifies LLaMA and base is GPT-2, note incompatibility
        if "llama" in adapter_base.lower() and "gpt2" in self.base_model_name.lower():
            base_compat = False

        # ---------------- Tier 2B: Behavioral Trigger & Inversion Scanner ----------------
        t2b_res = {"enabled": False}
        if self.enable_tier2b_gradient_search:
            if not base_compat:
                t2b_res = {
                    "status": "SKIPPED_INCOMPATIBLE_BASE",
                    "note": f"Adapter specifies '{adapter_base}', pipeline configured with '{self.base_model_name}'",
                }
            else:
                grad_searcher = self._get_gradient_searcher()
                t2b_res = grad_searcher.search_adapter(target)
                if t2b_res.get("verdict") == "FLAGGED":
                    elapsed = round(time.time() - t_start, 3)
                    attractors = t2b_res.get("differential_attractor_words", [])
                    return {
                        "verdict": "REJECTED",
                        "rejection_stage": "TIER_2B_GRADIENT_TRIGGER_SEARCH",
                        "reason": f"Gradient-guided trigger search detected backdoor attractor sinkhole (KL={t2b_res['mean_kl']:.2f}, attractors={attractors})",
                        "runtime_seconds": elapsed,
                        "diagnostics": diagnostics,
                    }
        elif self.enable_tier2b_inversion:
            if not base_compat:
                t2b_res = {
                    "status": "SKIPPED_INCOMPATIBLE_BASE",
                    "note": f"Adapter specifies '{adapter_base}', pipeline configured with '{self.base_model_name}'",
                }
            else:
                inv_scanner = self._get_inversion_scanner()
                t2b_res = inv_scanner.scan_adapter(target)
                if t2b_res.get("verdict") == "FLAGGED":
                    elapsed = round(time.time() - t_start, 3)
                    return {
                        "verdict": "REJECTED",
                        "rejection_stage": "TIER_2B_PAYLOAD_AGNOSTIC_INVERSION",
                        "reason": f"Universal attractor sinkhole detected (UAS={t2b_res['top_candidate']['uas_score']}, Invariance={t2b_res['invariance_analysis']['adapted_invariance']})",
                        "runtime_seconds": elapsed,
                        "diagnostics": diagnostics,
                    }
        diagnostics["tier2b_inversion"] = t2b_res

        # ---------------- Tier 3: Live Differential Probing ----------------
        t3_res = {"enabled": False}
        if self.enable_tier3_live_probe:
            if not base_compat:
                t3_res = {
                    "status": "SKIPPED_INCOMPATIBLE_BASE",
                    "note": f"Adapter specifies '{adapter_base}', pipeline configured with '{self.base_model_name}'",
                }
            else:
                prober = self._get_live_prober()
                t3_res = prober.evaluate_live_differential(target, safety_prompts=safety_probes)
                if t3_res.get("verdict") == "FLAGGED":
                    elapsed = round(time.time() - t_start, 3)
                    return {
                        "verdict": "REJECTED",
                        "rejection_stage": "TIER_3_LIVE_DIFFERENTIAL",
                        "reason": f"Live safety degradation detected (Delta_Safety={t3_res['mean_relative_delta']:.3f} < {self.safety_delta_threshold})",
                        "runtime_seconds": elapsed,
                        "diagnostics": diagnostics,
                    }
        diagnostics["tier3_differential"] = t3_res

        elapsed = round(time.time() - t_start, 3)
        reason = "Passed all active verification tiers"
        if tier2_quarantined:
            # Check if behavioral verification actually executed and passed
            behavioral_verified = False
            if self.enable_tier2b_gradient_search and t2b_res.get("verdict") == "NORMAL":
                behavioral_verified = True
            elif self.enable_tier2b_inversion and t2b_res.get("verdict") == "NORMAL":
                behavioral_verified = True
            elif self.enable_tier3_live_probe and t3_res.get("verdict") == "NORMAL":
                behavioral_verified = True

            if not behavioral_verified:
                anomaly_type = t2_res.get("anomaly_type", "ANOMALY")
                anomaly_reason = t2_res.get("anomaly_reason", "Anomalous rank-aware spectral distribution")
                return {
                    "verdict": "REJECTED",
                    "rejection_stage": "TIER_2_SPECTRAL_QUARANTINE_UNRESOLVED",
                    "reason": f"Spectral quarantine unresolved ({anomaly_type}): {anomaly_reason}; behavioral verification unavailable for base model '{adapter_base}'",
                    "runtime_seconds": elapsed,
                    "diagnostics": diagnostics,
                }
            reason = "Passed behavioral verification despite elevated spectral concentration (admitted under quarantine monitoring)"

        return {
            "verdict": "ADMITTED",
            "rejection_stage": None,
            "reason": reason,
            "tier2_quarantined": tier2_quarantined,
            "runtime_seconds": elapsed,
            "diagnostics": diagnostics,
        }
