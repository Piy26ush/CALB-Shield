#!/usr/bin/env python3
"""
internal_emulation/scripts/validate_emulation_intake.py

Internal Emulation Intake Gatekeeper.
DISCLOSURE: AUTOMATED ROLE-SEPARATED EMULATION — NOT INDEPENDENT EXTERNAL VALIDATION.

Enforces:
1. Frozen detection policy SHA-256 verification (fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb).
2. External path collision guard: strictly rejects any inputs pointing to registered external Phase 6 paths.
3. Detached SHA-256 vault checksum verification (verifies integrity, not external authenticity).
4. Blinding separation: strictly verifies that blind manifest contains zero ground-truth labels or triggers.
5. Emulation watermarking: verifies mandatory disclosure metadata.
6. Safe artifact handling: path traversal rejection and duplicate ID detection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EmulationIntakeValidator")

EXPECTED_FROZEN_THRESHOLDS_SHA256 = "fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb"
DEFAULT_FROZEN_THRESHOLDS_PATH = WORKSPACE_ROOT / "implementation_2/calb_shield_unified/configs/frozen_thresholds.json"

DEFAULT_EMULATION_MANIFEST_PATH = WORKSPACE_ROOT / "implementation_2/calb_shield_unified/internal_emulation/configs/emulation_blind_manifest.json"
DEFAULT_EMULATION_VAULT_PATH = WORKSPACE_ROOT / "implementation_2/calb_shield_unified/internal_emulation/configs/emulation_sealed_vault.json"

EXTERNAL_PHASE6_RESERVED_PATHS = {
    "redteam_blind_manifest.json",
    "sealed_redteam_vault.json",
    "sealed_redteam_vault.sha256",
    "committed_evaluation_traces.json",
    "committed_evaluation_traces.sha256",
    "blind_evaluation_scorecards.json",
}

FORBIDDEN_LEAK_KEYS = {
    "ground_truth",
    "is_poisoned",
    "attack_family",
    "trigger_string",
    "secret_trigger",
    "payload",
    "payload_string",
    "simulated_completion",
}

MANDATORY_DISCLOSURE_TEXT = "AUTOMATED ROLE-SEPARATED EMULATION — NOT INDEPENDENT EXTERNAL VALIDATION"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class GateResult:
    gate_id: str
    name: str
    passed: bool
    details: str
    evidence: Dict[str, Any]


class EmulationIntakeValidator:
    """Rigorous gatekeeper for the Internal Role-Separated Emulation harness."""

    def __init__(
        self,
        frozen_thresholds_path: Path = DEFAULT_FROZEN_THRESHOLDS_PATH,
        manifest_path: Optional[Path] = None,
        vault_path: Optional[Path] = None,
    ):
        self.frozen_thresholds_path = frozen_thresholds_path
        self.manifest_path = manifest_path or DEFAULT_EMULATION_MANIFEST_PATH
        self.vault_path = vault_path or DEFAULT_EMULATION_VAULT_PATH
        self.results: List[GateResult] = []

    def check_path_collision_guard(self) -> GateResult:
        """Gate 0: Ensure emulation scripts NEVER touch registered external Phase 6 paths."""
        colliding = []
        for p in [self.manifest_path, self.vault_path]:
            if p.name in EXTERNAL_PHASE6_RESERVED_PATHS:
                colliding.append(str(p))
            if "blind_redteam_traces" in str(p) or "blind_redteam_results" in str(p):
                colliding.append(str(p))

        if colliding:
            return GateResult(
                gate_id="EMU-GATE-0",
                name="External Phase 6 Path Collision Guard",
                passed=False,
                details=f"FATAL: Emulation inputs collide with external Phase 6 paths: {colliding}",
                evidence={"colliding_paths": colliding},
            )

        return GateResult(
            gate_id="EMU-GATE-0",
            name="External Phase 6 Path Collision Guard",
            passed=True,
            details="Path isolation verified: no collision with external Phase 6 locations",
            evidence={"manifest_path": str(self.manifest_path), "vault_path": str(self.vault_path)},
        )

    def check_frozen_policy(self) -> GateResult:
        """Gate 1: Verify detection policy lock checksum matches registered value."""
        if not self.frozen_thresholds_path.exists():
            return GateResult(
                gate_id="EMU-GATE-1",
                name="Frozen Policy Lock",
                passed=False,
                details=f"Frozen thresholds file missing at {self.frozen_thresholds_path}",
                evidence={"path": str(self.frozen_thresholds_path)},
            )

        current_hash = sha256_file(self.frozen_thresholds_path)
        passed = (current_hash == EXPECTED_FROZEN_THRESHOLDS_SHA256)
        return GateResult(
            gate_id="EMU-GATE-1",
            name="Frozen Policy Lock",
            passed=passed,
            details="Frozen policy hash matches registered SHA-256" if passed else f"Hash mismatch: {current_hash}",
            evidence={"expected": EXPECTED_FROZEN_THRESHOLDS_SHA256, "measured": current_hash},
        )

    def check_package_presence_and_checksum(self) -> GateResult:
        """Gate 2: Verify presence of manifest, vault, and detached vault checksum."""
        if not self.manifest_path.exists():
            return GateResult(
                gate_id="EMU-GATE-2",
                name="Emulation Package Presence & Checksum",
                passed=False,
                details=f"Manifest file missing at {self.manifest_path}",
                evidence={"manifest_exists": False},
            )

        if not self.vault_path.exists():
            return GateResult(
                gate_id="EMU-GATE-2",
                name="Emulation Package Presence & Checksum",
                passed=False,
                details=f"Vault file missing at {self.vault_path}",
                evidence={"vault_exists": False},
            )

        checksum_path = self.vault_path.with_suffix(".sha256")
        if not checksum_path.exists():
            return GateResult(
                gate_id="EMU-GATE-2",
                name="Emulation Package Presence & Checksum",
                passed=False,
                details=f"Detached checksum file missing at {checksum_path}",
                evidence={"checksum_exists": False},
            )

        checksum_content = checksum_path.read_text().strip()
        match = re.match(r"^([0-9a-f]{64})\s+([\w\.\-]+)$", checksum_content)
        if not match:
            return GateResult(
                gate_id="EMU-GATE-2",
                name="Emulation Package Presence & Checksum",
                passed=False,
                details="Malformed detached sha256 file format",
                evidence={"raw_content": checksum_content},
            )

        expected_hash = match.group(1).lower()
        actual_hash = sha256_file(self.vault_path).lower()

        if actual_hash != expected_hash:
            return GateResult(
                gate_id="EMU-GATE-2",
                name="Emulation Package Presence & Checksum",
                passed=False,
                details=f"Vault checksum mismatch: expected {expected_hash}, calculated {actual_hash}",
                evidence={"expected": expected_hash, "calculated": actual_hash},
            )

        return GateResult(
            gate_id="EMU-GATE-2",
            name="Emulation Package Presence & Checksum",
            passed=True,
            details="Vault integrity verified against authoritative detached checksum",
            evidence={"vault_sha256": actual_hash},
        )

    def check_blinding_and_disclosure(self) -> GateResult:
        """Gate 3: Verify zero label leakage and mandatory emulation disclosure."""
        try:
            with open(self.manifest_path) as f:
                manifest_data = json.load(f)
        except Exception as e:
            return GateResult(
                gate_id="EMU-GATE-3",
                name="Blinding Separation & Disclosure",
                passed=False,
                details=f"Failed to parse manifest JSON: {e}",
                evidence={"error": str(e)},
            )

        # Check disclosure watermark
        meta = manifest_data.get("manifest_metadata", {})
        disclosure = meta.get("disclosure", "")
        eval_type = meta.get("evaluation_type", "")

        if disclosure != MANDATORY_DISCLOSURE_TEXT or eval_type != "AUTOMATED_ROLE_SEPARATED_EMULATION":
            return GateResult(
                gate_id="EMU-GATE-3",
                name="Blinding Separation & Disclosure",
                passed=False,
                details="Missing or invalid mandatory emulation disclosure watermark",
                evidence={"disclosure": disclosure, "evaluation_type": eval_type},
            )

        # Check for label leakage in manifest
        targets = manifest_data.get("targets", [])
        leaks = []
        seen_target_ids: Set[str] = set()

        for idx, t in enumerate(targets):
            tid = t.get("target_id", f"INDEX_{idx}")
            if tid in seen_target_ids:
                leaks.append(f"Duplicate target_id '{tid}'")
            seen_target_ids.add(tid)

            for k in t.keys():
                if k in FORBIDDEN_LEAK_KEYS:
                    leaks.append(f"Target '{tid}' leaked forbidden key '{k}'")

            seen_prompt_ids: Set[str] = set()
            for p in t.get("test_prompts", []):
                pid = p.get("prompt_id")
                if pid in seen_prompt_ids:
                    leaks.append(f"Target '{tid}' has duplicate prompt_id '{pid}'")
                seen_prompt_ids.add(pid)
                for pk in p.keys():
                    if pk in FORBIDDEN_LEAK_KEYS:
                        leaks.append(f"Target '{tid}' prompt '{pid}' leaked forbidden key '{pk}'")

        if leaks:
            return GateResult(
                gate_id="EMU-GATE-3",
                name="Blinding Separation & Disclosure",
                passed=False,
                details=f"Blinding violation / duplicate IDs: {'; '.join(leaks[:3])}",
                evidence={"leaks": leaks},
            )

        return GateResult(
            gate_id="EMU-GATE-3",
            name="Blinding Separation & Disclosure",
            passed=True,
            details=f"Blinding separation verified across {len(targets)} targets; mandatory disclosure stamped",
            evidence={"total_targets": len(targets), "disclosure": disclosure},
        )

    def check_safe_adapter_paths(self) -> GateResult:
        """Gate 4: Verify path traversal rejection on adapter references."""
        try:
            with open(self.manifest_path) as f:
                manifest_data = json.load(f)
        except Exception:
            return GateResult(gate_id="EMU-GATE-4", name="Safe Artifact Paths", passed=False, details="Blocked", evidence={})

        violations = []
        for t in manifest_data.get("targets", []):
            tid = t.get("target_id")
            apath = t.get("adapter_path", "")
            if ".." in apath or apath.startswith("/") or apath.startswith("\\"):
                violations.append(f"Target '{tid}' uses illegal path traversal: '{apath}'")

        if violations:
            return GateResult(
                gate_id="EMU-GATE-4",
                name="Safe Artifact Paths",
                passed=False,
                details=f"Path traversal detected: {'; '.join(violations[:2])}",
                evidence={"violations": violations},
            )

        return GateResult(
            gate_id="EMU-GATE-4",
            name="Safe Artifact Paths",
            passed=True,
            details="All adapter relative paths verified safe from directory traversal",
            evidence={"targets_checked": len(manifest_data.get("targets", []))},
        )

    def validate_all(self) -> Dict[str, Any]:
        """Executes all intake gates in order and returns formal decision."""
        self.results = [
            self.check_path_collision_guard(),
            self.check_frozen_policy(),
            self.check_package_presence_and_checksum(),
            self.check_blinding_and_disclosure(),
            self.check_safe_adapter_paths(),
        ]

        all_passed = all(g.passed for g in self.results)
        verdict = "GO (EMULATION AUTHORIZED)" if all_passed else "NO-GO (EXECUTION BLOCKED)"

        return {
            "verdict": verdict,
            "all_gates_passed": all_passed,
            "gates_evaluated": len(self.results),
            "gates_passed": sum(1 for g in self.results if g.passed),
            "gates_failed": sum(1 for g in self.results if not g.passed),
            "gate_results": [
                {
                    "gate_id": g.gate_id,
                    "name": g.name,
                    "passed": g.passed,
                    "details": g.details,
                    "evidence": g.evidence,
                }
                for g in self.results
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="CALB-Shield Internal Emulation Intake Validator")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_EMULATION_MANIFEST_PATH)
    parser.add_argument("--vault", type=Path, default=DEFAULT_EMULATION_VAULT_PATH)
    parser.add_argument("--policy", type=Path, default=DEFAULT_FROZEN_THRESHOLDS_PATH)

    args = parser.parse_args()
    validator = EmulationIntakeValidator(
        frozen_thresholds_path=args.policy,
        manifest_path=args.manifest,
        vault_path=args.vault,
    )
    report = validator.validate_all()
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["all_gates_passed"] else 1)


if __name__ == "__main__":
    main()

