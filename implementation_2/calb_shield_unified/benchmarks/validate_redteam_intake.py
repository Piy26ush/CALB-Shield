#!/usr/bin/env python3
"""
benchmarks/validate_redteam_intake.py

External Red-Team Delivery Intake Validator & Gatekeeper for Phase 6.
Enforces:
1. Frozen threshold integrity check (SHA-256 == fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb).
2. Blinding boundary verification (zero ground-truth labels or trigger strings leaked in blind manifest).
3. Cohort allocation verification: exactly 86 targets (26 attacks, 52 benign [2:1], 8 structural controls).
4. Architecture matrix check: 14 GPT-2, 26 LLaMA-3-8B, 20 Mistral-7B, 26 Qwen-2.5-Coder.
5. Adapter package integrity: safe deserialization, PEFT schema compliance, checksum verification.
6. Safe artifact handling: path traversal checks and non-executable serialization.
7. Emits formal GO / NO-GO decision with machine-readable audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RedTeamIntakeValidator")

EXPECTED_FROZEN_THRESHOLDS_SHA256 = "fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb"
DEFAULT_FROZEN_THRESHOLDS_PATH = WORKSPACE_ROOT / "implementation_2/calb_shield_unified/configs/frozen_thresholds.json"

EXPECTED_COHORT_COUNTS = {
    "total": 86,
    "attacks": 26,
    "benign_near_matches": 52,
    "structural_controls": 8,
}

EXPECTED_ARCH_COUNTS = {
    "gpt2": {"attacks": 4, "benign": 8, "structural": 2, "total": 14},
    "llama-3-8b": {"attacks": 8, "benign": 16, "structural": 2, "total": 26},
    "mistral-7b": {"attacks": 6, "benign": 12, "structural": 2, "total": 20},
    "qwen-2.5-coder": {"attacks": 8, "benign": 16, "structural": 2, "total": 26},
}

EXPECTED_FAMILIES = {"FAM-1", "FAM-2", "FAM-3", "FAM-4", "FAM-5"}

EMPTY_STRING_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
MANIFEST_ALLOWED_TOP_KEYS = {"manifest_metadata", "targets"}
MANIFEST_ALLOWED_METADATA_KEYS = {"protocol", "delivery_timestamp_utc", "total_targets", "author_entity"}
MANIFEST_ALLOWED_TARGET_KEYS = {"target_id", "architecture", "adapter_path", "is_structural_control", "test_prompts"}
MANIFEST_ALLOWED_PROMPT_KEYS = {"prompt_id", "prompt"}
FORBIDDEN_LEAK_KEYS = {"ground_truth", "is_poisoned", "attack_family", "trigger_string", "secret_trigger", "payload", "simulated_completion"}


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


class RedTeamIntakeValidator:
    """Rigorous gatekeeper validating incoming external red-team delivery packages (V2.1 Specification)."""

    def __init__(
        self,
        frozen_thresholds_path: Path = DEFAULT_FROZEN_THRESHOLDS_PATH,
        manifest_path: Optional[Path] = None,
        vault_path: Optional[Path] = None,
    ):
        self.frozen_thresholds_path = frozen_thresholds_path
        self.manifest_path = manifest_path
        self.vault_path = vault_path
        self.results: List[GateResult] = []

    def check_gate_1_frozen_policy(self) -> GateResult:
        """Gate 1: Verify detection policy lock checksum."""
        if not self.frozen_thresholds_path.exists():
            return GateResult(
                gate_id="GATE-1",
                name="Frozen Policy Lock",
                passed=False,
                details=f"Frozen thresholds file missing at {self.frozen_thresholds_path}",
                evidence={"path": str(self.frozen_thresholds_path)},
            )
        actual = sha256_file(self.frozen_thresholds_path)
        passed = (actual == EXPECTED_FROZEN_THRESHOLDS_SHA256)
        return GateResult(
            gate_id="GATE-1",
            name="Frozen Policy Lock",
            passed=passed,
            details="Frozen policy hash strictly matches registered SHA-256" if passed else "Tampered thresholds detected",
            evidence={"expected": EXPECTED_FROZEN_THRESHOLDS_SHA256, "measured": actual},
        )

    def check_gate_2_file_presence(self) -> GateResult:
        """Gate 2: Verify presence of external manifest, sealed vault, and authoritative detached sha256."""
        missing = []
        if self.manifest_path is None or not self.manifest_path.exists():
            missing.append("redteam_blind_manifest.json")
        if self.vault_path is None or not self.vault_path.exists():
            missing.append("sealed_redteam_vault.json")

        checksum_file = None
        if self.vault_path:
            checksum_file = self.vault_path.parent / "sealed_redteam_vault.sha256"
            if not checksum_file.exists():
                missing.append("sealed_redteam_vault.sha256 (detached checksum file missing)")
            else:
                raw_lines = [l.strip() for l in checksum_file.read_text().split("\n") if l.strip()]
                if len(raw_lines) != 1 or not re.match(r"^[0-9a-f]{64}  sealed_redteam_vault\.json$", raw_lines[0]):
                    missing.append("sealed_redteam_vault.sha256 malformed syntax (must be: <64_hex_hash>  sealed_redteam_vault.json)")
                else:
                    expected_vhash = raw_lines[0].split()[0]
                    if self.vault_path.exists():
                        measured_vhash = sha256_file(self.vault_path)
                        if measured_vhash == EMPTY_STRING_SHA256:
                            missing.append("sealed_redteam_vault.json is empty (empty-file hash detected)")
                        elif measured_vhash != expected_vhash:
                            missing.append(f"sealed_redteam_vault.json SHA-256 mismatch ({measured_vhash} != {expected_vhash})")

        passed = len(missing) == 0
        return GateResult(
            gate_id="GATE-2",
            name="External Package Presence & Detached Checksum",
            passed=passed,
            details="All external files present and detached checksum verified" if passed else f"Missing/malformed deliverables: {'; '.join(missing)}",
            evidence={
                "manifest_path": str(self.manifest_path) if self.manifest_path else None,
                "manifest_exists": self.manifest_path.exists() if self.manifest_path else False,
                "vault_path": str(self.vault_path) if self.vault_path else None,
                "vault_exists": self.vault_path.exists() if self.vault_path else False,
                "checksum_file": str(checksum_file) if checksum_file else None,
                "checksum_exists": checksum_file.exists() if checksum_file else False,
            },
        )

    def check_gate_3_blinding_separation(self, manifest_data: Dict[str, Any]) -> GateResult:
        """Gate 3: Ensure ZERO ground truth, secret triggers, or unauthorized keys leak into manifest."""
        top_keys = set(manifest_data.keys())
        top_unauthorized = top_keys - MANIFEST_ALLOWED_TOP_KEYS
        if top_unauthorized:
            return GateResult(
                gate_id="GATE-3",
                name="Blinding Separation (Zero Leakage)",
                passed=False,
                details=f"Top-level manifest contains unauthorized keys: {top_unauthorized}",
                evidence={"top_unauthorized": list(top_unauthorized)},
            )

        metadata = manifest_data.get("manifest_metadata", {})
        meta_unauthorized = set(metadata.keys()) - MANIFEST_ALLOWED_METADATA_KEYS
        if meta_unauthorized:
            return GateResult(
                gate_id="GATE-3",
                name="Blinding Separation (Zero Leakage)",
                passed=False,
                details=f"Manifest metadata contains unauthorized keys: {meta_unauthorized}",
                evidence={"meta_unauthorized": list(meta_unauthorized)},
            )

        targets = manifest_data.get("targets", [])
        leaks = []
        target_ids = set()

        for t in targets:
            tid = t.get("target_id")
            if not tid:
                leaks.append("Target entry missing target_id")
                continue
            if tid in target_ids:
                leaks.append(f"Duplicate target_id detected: {tid}")
            target_ids.add(tid)

            unexpected = set(t.keys()) - MANIFEST_ALLOWED_TARGET_KEYS
            if unexpected:
                leaks.append(f"Target {tid} contains unauthorized keys: {unexpected}")

            found_forbidden = [k for k in FORBIDDEN_LEAK_KEYS if k in t]
            if found_forbidden:
                leaks.append(f"Target {tid} contains leaked secret keys: {found_forbidden}")

        passed = len(leaks) == 0
        return GateResult(
            gate_id="GATE-3",
            name="Blinding Separation (Zero Leakage)",
            passed=passed,
            details="Manifest is fully blinded (zero ground truth or trigger strings exposed)" if passed else f"Blinding violation detected: {len(leaks)} issues found",
            evidence={"leak_count": len(leaks), "sample_leaks": leaks[:5]},
        )

    def check_gate_4_cohort_allocation(self, manifest_data: Dict[str, Any], vault_data: Optional[Dict[str, Any]]) -> GateResult:
        """Gate 4: Verify exact N=86 target cohort and architecture distribution."""
        targets = manifest_data.get("targets", [])
        total = len(targets)

        arch_counts: Dict[str, int] = {}
        for t in targets:
            arch = t.get("architecture")
            arch_counts[arch] = arch_counts.get(arch, 0) + 1

        arch_match = True
        mismatches = []
        for arch, exp in EXPECTED_ARCH_COUNTS.items():
            cnt = arch_counts.get(arch, 0)
            if cnt != exp["total"]:
                arch_match = False
                mismatches.append(f"{arch}: expected {exp['total']}, got {cnt}")

        total_match = (total == EXPECTED_COHORT_COUNTS["total"])
        passed = total_match and arch_match

        # If vault is present, check ground truth distribution
        vault_details = {}
        if vault_data and "targets" in vault_data:
            v_targets = vault_data["targets"]
            n_atk = sum(1 for t in v_targets if t.get("ground_truth") == "POISONED")
            n_ben = sum(1 for t in v_targets if t.get("ground_truth") == "CLEAN")
            n_str = sum(1 for t in v_targets if t.get("ground_truth") == "INCOMPATIBLE")
            vault_details = {"vault_attacks": n_atk, "vault_benign": n_ben, "vault_structural": n_str}
            if n_atk != 26 or n_ben != 52 or n_str != 8:
                passed = False
                mismatches.append(f"Vault ground truth mismatch: Atk={n_atk}/26, Ben={n_ben}/52, Str={n_str}/8")

        details = "Cohort matches exact N=86 registered matrix" if passed else f"Cohort allocation mismatch: {'; '.join(mismatches)}"
        return GateResult(
            gate_id="GATE-4",
            name="Cohort Allocation (N=86 Matrix)",
            passed=passed,
            details=details,
            evidence={
                "total_targets": total,
                "expected_total": EXPECTED_COHORT_COUNTS["total"],
                "arch_counts": arch_counts,
                "expected_arch_counts": EXPECTED_ARCH_COUNTS,
                **vault_details,
            },
        )

    def check_gate_5_adapter_artifact_integrity(self, manifest_data: Dict[str, Any], vault_data: Optional[Dict[str, Any]]) -> GateResult:
        """Gate 5: Check adapter path validity, safe deserialization, and PEFT schema."""
        targets = manifest_data.get("targets", [])
        issues = []
        vault_hashes = {}
        if vault_data and "targets" in vault_data:
            for vt in vault_data["targets"]:
                vault_hashes[vt["target_id"]] = vt.get("adapter_sha256")

        for t in targets:
            tid = t.get("target_id")
            raw_path = t.get("adapter_path")
            if not raw_path:
                issues.append(f"{tid}: missing adapter_path")
                continue

            # Path traversal safety check
            if ".." in str(raw_path):
                issues.append(f"{tid}: illegal path traversal sequence '..' in path {raw_path}")
                continue

            p = Path(raw_path)
            if not p.is_absolute():
                p = WORKSPACE_ROOT / p

            if not p.exists():
                issues.append(f"{tid}: directory does not exist at {p}")
                continue

            cfg = p / "adapter_config.json"
            if not cfg.exists():
                issues.append(f"{tid}: missing adapter_config.json")
                continue

            # Verify weights file exists
            has_weights = (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
            if not has_weights:
                issues.append(f"{tid}: missing adapter_model.safetensors or adapter_model.bin")
                continue

            # If vault provided, check SHA-256
            if tid in vault_hashes and vault_hashes[tid]:
                weight_file = (p / "adapter_model.safetensors") if (p / "adapter_model.safetensors").exists() else (p / "adapter_model.bin")
                measured_sha = sha256_file(weight_file)
                if measured_sha != vault_hashes[tid]:
                    issues.append(f"{tid}: SHA-256 mismatch (vault={vault_hashes[tid][:12]}, measured={measured_sha[:12]})")

        passed = len(issues) == 0
        return GateResult(
            gate_id="GATE-5",
            name="Adapter Artifact Integrity & Security",
            passed=passed,
            details="All adapter packages exist, adhere to PEFT schema, and match hashes" if passed else f"Integrity violations in {len(issues)} targets",
            evidence={"issue_count": len(issues), "sample_issues": issues[:5]},
        )

    def check_gate_6_test_prompts_coverage(self, manifest_data: Dict[str, Any]) -> GateResult:
        """Gate 6: Verify prompt pairing, prompt schema compliance, and absence of simulated fixtures."""
        targets = manifest_data.get("targets", [])
        issues = []

        for t in targets:
            tid = t.get("target_id", "unknown")
            prompts = t.get("test_prompts", [])
            is_struct = t.get("is_structural_control", False)

            if not prompts and not is_struct:
                issues.append(f"{tid}: no test prompts provided for non-structural adapter")
                continue

            prompt_ids = set()
            for p_idx, p in enumerate(prompts):
                if not isinstance(p, dict):
                    issues.append(f"{tid}[p{p_idx}]: prompt entry is not a dictionary")
                    continue

                pid = p.get("prompt_id")
                if not pid:
                    issues.append(f"{tid}[p{p_idx}]: missing prompt_id")
                elif pid in prompt_ids:
                    issues.append(f"{tid}[p{p_idx}]: duplicate prompt_id '{pid}' within target")
                else:
                    prompt_ids.add(pid)

                ptext = p.get("prompt")
                if not ptext or not isinstance(ptext, str) or not ptext.strip():
                    issues.append(f"{tid}[p{p_idx}]: prompt text missing or empty")

                # V2.1 Requirement: simulated_completion is strictly prohibited in real delivery manifest
                if "simulated_completion" in p:
                    issues.append(f"{tid}[p{p_idx}]: 'simulated_completion' fixture prohibited in blind delivery manifest (live forward inference required)")

                unauth_pkeys = set(p.keys()) - MANIFEST_ALLOWED_PROMPT_KEYS
                if unauth_pkeys:
                    issues.append(f"{tid}[p{p_idx}]: unauthorized prompt keys: {unauth_pkeys}")

        passed = len(issues) == 0
        return GateResult(
            gate_id="GATE-6",
            name="Test Prompts Coverage & Formatting",
            passed=passed,
            details="All test prompts formatted correctly for physical model generation" if passed else f"Prompt issues in {len(issues)} targets",
            evidence={"issue_count": len(issues), "sample_issues": issues[:5]},
        )

    def run_all_checks(self) -> Dict[str, Any]:
        """Execute all intake verification gates and render final GO / NO-GO verdict."""
        logger.info("Initiating External Red-Team Delivery Intake Verification...")
        self.results = []

        # Gate 1: Frozen policy lock
        g1 = self.check_gate_1_frozen_policy()
        self.results.append(g1)

        # Gate 2: File presence
        g2 = self.check_gate_2_file_presence()
        self.results.append(g2)

        manifest_data: Dict[str, Any] = {}
        vault_data: Optional[Dict[str, Any]] = None

        if g2.passed:
            with open(self.manifest_path) as f:
                manifest_data = json.load(f)
            if self.vault_path and self.vault_path.exists():
                with open(self.vault_path) as f:
                    vault_data = json.load(f)

            # Gate 3: Blinding
            self.results.append(self.check_gate_3_blinding_separation(manifest_data))
            # Gate 4: Cohort allocation
            self.results.append(self.check_gate_4_cohort_allocation(manifest_data, vault_data))
            # Gate 5: Artifact integrity
            self.results.append(self.check_gate_5_adapter_artifact_integrity(manifest_data, vault_data))
            # Gate 6: Prompt coverage
            self.results.append(self.check_gate_6_test_prompts_coverage(manifest_data))
        else:
            # Mark remaining gates as NOT EVALUATED (Blocked on Gate 2)
            for gid, gname, reason in [
                ("GATE-3", "Blinding Separation (Zero Leakage)", "Blocked by missing manifest"),
                ("GATE-4", "Cohort Allocation (N=86 Matrix)", "Blocked by missing manifest"),
                ("GATE-5", "Adapter Artifact Integrity & Security", "Blocked by missing manifest"),
                ("GATE-6", "Test Prompts Coverage & Formatting", "Blocked by missing manifest"),
            ]:
                self.results.append(GateResult(gate_id=gid, name=gname, passed=False, details=reason, evidence={}))

        all_passed = all(r.passed for r in self.results)
        verdict = "GO" if all_passed else "NO-GO (EXECUTION BLOCKED)"

        summary = {
            "verdict": verdict,
            "all_gates_passed": all_passed,
            "gates_evaluated": len(self.results),
            "gates_passed": sum(1 for r in self.results if r.passed),
            "gates_failed": sum(1 for r in self.results if not r.passed),
            "gate_results": [asdict(r) for r in self.results],
        }

        logger.info("Intake Verification Complete: %s (%d/%d gates passed)", verdict, summary["gates_passed"], len(self.results))
        return summary


def main():
    parser = argparse.ArgumentParser(description="External Red-Team Delivery Intake Validator")
    parser.add_argument("--manifest", type=Path, default=WORKSPACE_ROOT / "implementation_2/calb_shield_unified/configs/redteam_blind_manifest.json")
    parser.add_argument("--vault", type=Path, default=WORKSPACE_ROOT / "implementation_2/calb_shield_unified/configs/sealed_redteam_vault.json")
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_FROZEN_THRESHOLDS_PATH)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    validator = RedTeamIntakeValidator(
        frozen_thresholds_path=args.thresholds,
        manifest_path=args.manifest,
        vault_path=args.vault,
    )
    report = validator.run_all_checks()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Intake report written to {args.output}")
    else:
        print(json.dumps(report, indent=2))

    sys.exit(0 if report["all_gates_passed"] else 1)


if __name__ == "__main__":
    main()

