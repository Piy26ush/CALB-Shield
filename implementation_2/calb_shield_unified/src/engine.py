"""
implementation_2/calb_shield_unified/src/engine.py

Master Engine for Unified CALB-Shield Adapter Admission Control.
Orchestrates Tier 1 Schema Validation, Tier 2 Architecture-Calibrated Spectral Screening,
Tier 2B Behavioral Trigger Inversion, and Tier 3 Differential Safety Probing.
Implements the universal 3-verdict policy: ADMIT, QUARANTINE, or REJECT.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

from implementation_2.calb_shield_unified.src.profiles import (
    ArchitectureProfile,
    ArchitectureRegistry,
)
from implementation_2.calb_shield_unified.src.backend import (
    InferenceSession,
    create_inference_session,
    MockInferenceSession,
)
from implementation_2.calb_shield_unified.src.tier1_provenance import Tier1ProvenanceValidator
from implementation_2.calb_shield_unified.src.tier2_spectral import Tier2SpectralScanner
from implementation_2.calb_shield_unified.src.tier2b_behavioral import Tier2BBehavioralScanner
from implementation_2.calb_shield_unified.src.tier3_safety import Tier3SafetyProber


class AdmissionVerdict(str, Enum):
    ADMIT = "ADMIT"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


@dataclass
class AdmissionReport:
    """Standardized decision certificate emitted for every evaluated adapter."""
    adapter_name: str
    adapter_path: str
    verdict: AdmissionVerdict
    primary_stage: Optional[str]
    primary_reason: str
    profile_name: str
    model_family: str
    tier1_provenance: Dict[str, Any]
    tier2_spectral: Dict[str, Any]
    tier2b_behavioral: Dict[str, Any]
    tier3_safety: Dict[str, Any]
    total_runtime_seconds: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        # Sanitize profile in tier1_provenance if present as object or dict with function
        if "tier1_provenance" in d and isinstance(d["tier1_provenance"], dict):
            prof = d["tier1_provenance"].get("profile")
            if prof is not None and not isinstance(prof, (str, int, float, bool)):
                if isinstance(prof, dict):
                    prof_sanitized = dict(prof)
                    if "chat_formatter" in prof_sanitized:
                        prof_sanitized["chat_formatter"] = getattr(prof_sanitized["chat_formatter"], "__name__", "custom")
                    d["tier1_provenance"]["profile"] = prof_sanitized
                else:
                    d["tier1_provenance"]["profile"] = getattr(prof, "name", str(prof))
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class CALBShieldUnifiedEngine:
    """
    Unified multi-tier admission controller for PEFT LoRA adapters across model architectures.
    """

    def __init__(
        self,
        mock_backend: bool = False,
        enable_tier2_spectral: bool = True,
        enable_tier2b_behavioral: bool = True,
        enable_tier3_safety: bool = True,
    ):
        self.mock_backend = mock_backend
        self.enable_tier2_spectral = enable_tier2_spectral
        self.enable_tier2b_behavioral = enable_tier2b_behavioral
        self.enable_tier3_safety = enable_tier3_safety

        self.tier1_validator = Tier1ProvenanceValidator()
        self.tier2_scanner = Tier2SpectralScanner()
        self.tier2b_scanner = Tier2BBehavioralScanner()
        self.tier3_prober = Tier3SafetyProber()

    def evaluate_adapter(
        self,
        adapter_path: Path | str,
        session: Optional[InferenceSession] = None,
        target_profile: Optional[ArchitectureProfile] = None,
    ) -> AdmissionReport:
        """
        Evaluate an adapter through the complete multi-tier pipeline.
        """
        t0 = time.time()
        target = Path(adapter_path)
        adapter_name = target.name

        # ---------------- Tier 1: Provenance & Schema ----------------
        t1_res = self.tier1_validator.validate_package(target, target_profile=target_profile)
        if not t1_res["passed"]:
            elapsed = round(time.time() - t0, 3)
            return AdmissionReport(
                adapter_name=adapter_name,
                adapter_path=str(target),
                verdict=AdmissionVerdict.REJECT,
                primary_stage="TIER_1_PROVENANCE",
                primary_reason=t1_res["rejection_reason"] or "Tier 1 validation failed",
                profile_name="unresolved",
                model_family="unknown",
                tier1_provenance=t1_res,
                tier2_spectral={},
                tier2b_behavioral={},
                tier3_safety={},
                total_runtime_seconds=elapsed,
                timestamp=time.time(),
            )

        profile: ArchitectureProfile = t1_res["profile"]

        # ---------------- Tier 2: Architecture-Calibrated Spectral ----------------
        t2_res: Dict[str, Any] = {"enabled": False}
        spectral_quarantined = False

        if self.enable_tier2_spectral:
            t2_res = self.tier2_scanner.scan_adapter(target, profile)
            t2_verdict = t2_res.get("verdict")

            if t2_verdict == "REJECT_CRITICAL_SPECTRAL":
                elapsed = round(time.time() - t0, 3)
                return AdmissionReport(
                    adapter_name=adapter_name,
                    adapter_path=str(target),
                    verdict=AdmissionVerdict.REJECT,
                    primary_stage="TIER_2_SPECTRAL",
                    primary_reason=t2_res["primary_reason"],
                    profile_name=profile.name,
                    model_family=profile.family,
                    tier1_provenance=t1_res,
                    tier2_spectral=t2_res,
                    tier2b_behavioral={},
                    tier3_safety={},
                    total_runtime_seconds=elapsed,
                    timestamp=time.time(),
                )
            elif t2_verdict == "QUARANTINE_FLAGGED":
                spectral_quarantined = True

        # Initialize or resolve inference session for behavioral tiers
        active_session = session
        if active_session is None and (self.enable_tier2b_behavioral or self.enable_tier3_safety):
            try:
                active_session = create_inference_session(profile, mock=self.mock_backend)
            except Exception as e:
                # If physical model weights are missing and mock was not requested
                active_session = None

        # ---------------- Tier 2B: Behavioral Trigger Inversion ----------------
        t2b_res: Dict[str, Any] = {"enabled": False}
        if self.enable_tier2b_behavioral:
            if active_session is not None:
                t2b_res = self.tier2b_scanner.scan_adapter(active_session, target, profile)
                if t2b_res.get("verdict") == "FLAGGED_SINKHOLE":
                    elapsed = round(time.time() - t0, 3)
                    return AdmissionReport(
                        adapter_name=adapter_name,
                        adapter_path=str(target),
                        verdict=AdmissionVerdict.REJECT,
                        primary_stage="TIER_2B_BEHAVIORAL_SINKHOLE",
                        primary_reason=t2b_res["reason"],
                        profile_name=profile.name,
                        model_family=profile.family,
                        tier1_provenance=t1_res,
                        tier2_spectral=t2_res,
                        tier2b_behavioral=t2b_res,
                        tier3_safety={},
                        total_runtime_seconds=elapsed,
                        timestamp=time.time(),
                    )
            else:
                t2b_res = {
                    "enabled": False,
                    "status": "SKIPPED_NO_SESSION",
                    "reason": "Live inference session not available for architecture profile",
                }

        # ---------------- Tier 3: Differential Safety Probing ----------------
        t3_res: Dict[str, Any] = {"enabled": False}
        if self.enable_tier3_safety:
            if active_session is not None:
                t3_res = self.tier3_prober.evaluate_adapter(active_session, target, profile)
                if t3_res.get("verdict") == "FLAGGED_SAFETY_STRIPPED":
                    elapsed = round(time.time() - t0, 3)
                    return AdmissionReport(
                        adapter_name=adapter_name,
                        adapter_path=str(target),
                        verdict=AdmissionVerdict.REJECT,
                        primary_stage="TIER_3_SAFETY_STRIPPED",
                        primary_reason=t3_res["reason"],
                        profile_name=profile.name,
                        model_family=profile.family,
                        tier1_provenance=t1_res,
                        tier2_spectral=t2_res,
                        tier2b_behavioral=t2b_res,
                        tier3_safety=t3_res,
                        total_runtime_seconds=elapsed,
                        timestamp=time.time(),
                    )
            else:
                t3_res = {
                    "enabled": False,
                    "status": "SKIPPED_NO_SESSION",
                    "reason": "Live inference session not available for architecture profile",
                }

        # ---------------- Final Admission Decision ----------------
        elapsed = round(time.time() - t0, 3)

        snorm = t2_res.get("max_spectral_norm")
        min_snorm = getattr(profile, "min_spectral_norm", 0.1)
        is_stealth_norm = (snorm is not None and 0.0 < snorm <= min_snorm)

        if spectral_quarantined:
            primary_stage = "TIER_2_SPECTRAL"
            final_verdict = AdmissionVerdict.QUARANTINE
            primary_reason = (
                f"Quarantined under spectral concentration monitoring: {t2_res.get('primary_reason')}; "
                f"adapter held for targeted sandbox analysis"
            )
        elif is_stealth_norm:
            final_verdict = AdmissionVerdict.QUARANTINE
            primary_stage = "TIER_2_SPECTRAL"
            primary_reason = (
                f"Quarantined under stealth dormancy risk: ultra-low spectral norm (sigma_1={snorm:.3f} <= {min_snorm:.3f}) "
                f"renders zero-reference behavioral probing unobservable; held for targeted sandbox analysis"
            )
        else:
            # Check if required behavioral verification failed or could not be loaded
            if self.enable_tier2b_behavioral and t2b_res.get("verdict") in ["SKIPPED_INCOMPATIBLE", "ERROR"]:
                final_verdict = AdmissionVerdict.QUARANTINE
                primary_stage = "TIER_2B_BEHAVIORAL_SINKHOLE"
                primary_reason = (
                    f"Quarantined unverified submission: required behavioral evaluation could not be executed "
                    f"({t2b_res.get('reason')})"
                )
            else:
                final_verdict = AdmissionVerdict.ADMIT
                primary_stage = None
                primary_reason = "Passed all active security verification tiers without anomaly"

        return AdmissionReport(
            adapter_name=adapter_name,
            adapter_path=str(target),
            verdict=final_verdict,
            primary_stage=primary_stage if final_verdict != AdmissionVerdict.ADMIT else None,
            primary_reason=primary_reason,
            profile_name=profile.name,
            model_family=profile.family,
            tier1_provenance=t1_res,
            tier2_spectral=t2_res,
            tier2b_behavioral=t2b_res,
            tier3_safety=t3_res,
            total_runtime_seconds=elapsed,
            timestamp=time.time(),
        )
