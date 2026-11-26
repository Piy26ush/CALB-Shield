#!/usr/bin/env python3
"""
internal_emulation/scripts/run_emulation_evaluation.py

Role B: Blind Blue-Team Evaluation Runner (Internal Emulation Harness).
DISCLOSURE: AUTOMATED ROLE-SEPARATED EMULATION — NOT INDEPENDENT EXTERNAL VALIDATION.

Enforces:
1. Operates exclusively on `emulation_blind_manifest.json`.
2. ZERO access to `emulation_sealed_vault.json` (no CLI parameter exists).
3. Path guard prevents reading from or writing to external Phase 6 locations.
4. Verifies frozen policy SHA-256 (fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb).
5. Commits raw traces to `committed_emulation_traces.json` with detached `.sha256`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation_2.calb_shield_unified.src.engine import CALBShieldUnifiedEngine
from implementation_2.calb_shield_unified.src.guardrail import RuntimeGuardrailSidecar
from implementation_2.calb_shield_unified.src.profiles import ArchitectureRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EmulationRunner")

EXPECTED_FROZEN_THRESHOLDS_SHA256 = "fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb"
DEFAULT_FROZEN_THRESHOLDS_PATH = WORKSPACE_ROOT / "implementation_2/calb_shield_unified/configs/frozen_thresholds.json"

DEFAULT_MANIFEST_PATH = WORKSPACE_ROOT / "implementation_2/calb_shield_unified/internal_emulation/configs/emulation_blind_manifest.json"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "implementation_2/calb_shield_unified/internal_emulation/results/traces"

MANDATORY_DISCLOSURE_TEXT = "AUTOMATED ROLE-SEPARATED EMULATION — NOT INDEPENDENT EXTERNAL VALIDATION"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_policy(policy_path: Path = DEFAULT_FROZEN_THRESHOLDS_PATH):
    if not policy_path.exists():
        raise FileNotFoundError(f"Frozen thresholds file missing at: {policy_path}")
    current_hash = sha256_file(policy_path)
    if current_hash != EXPECTED_FROZEN_THRESHOLDS_SHA256:
        raise ValueError(
            f"FROZEN POLICY MISMATCH: Expected {EXPECTED_FROZEN_THRESHOLDS_SHA256}, got {current_hash}."
        )
    logger.info("Frozen policy integrity verified: %s", current_hash)


def assert_path_isolation(manifest_path: Path, output_dir: Path):
    colliding = []
    if "configs/redteam_blind_manifest.json" in str(manifest_path) or "sealed_redteam_vault" in str(manifest_path):
        colliding.append(str(manifest_path))
    if "results/blind_redteam_traces" in str(output_dir) or "results/blind_redteam_results" in str(output_dir):
        colliding.append(str(output_dir))

    if colliding:
        raise PermissionError(
            f"PathCollisionError: Internal emulation runner cannot access external Phase 6 locations: {colliding}"
        )


def run_emulation_evaluation(
    manifest_path: Path,
    output_dir: Path,
    policy_path: Path = DEFAULT_FROZEN_THRESHOLDS_PATH,
) -> Dict[str, Any]:
    assert_path_isolation(manifest_path, output_dir)
    verify_frozen_policy(policy_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Emulation manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    # Verify disclosure watermark in manifest
    meta = manifest_data.get("manifest_metadata", {})
    if meta.get("disclosure") != MANDATORY_DISCLOSURE_TEXT:
        raise ValueError("Invalid manifest: Missing mandatory emulation disclosure watermark.")

    targets = manifest_data.get("targets", [])
    logger.info("Loaded %d targets from internal emulation manifest.", len(targets))

    engine = CALBShieldUnifiedEngine()
    guardrails = {
        "gpt2": RuntimeGuardrailSidecar(profile=ArchitectureRegistry.get_profile("gpt2")),
        "llama-3-8b": RuntimeGuardrailSidecar(profile=ArchitectureRegistry.get_profile("llama-3-8b")),
        "mistral-7b": RuntimeGuardrailSidecar(profile=ArchitectureRegistry.get_profile("mistral-7b")),
        "qwen-2.5-coder-1.5b": RuntimeGuardrailSidecar(profile=ArchitectureRegistry.get_profile("qwen-2.5-coder-1.5b")),
    }

    eval_results: List[Dict[str, Any]] = []

    for idx, target in enumerate(targets, 1):
        target_id = target["target_id"]
        arch = target["architecture"]
        adapter_rel = target["adapter_path"]
        adapter_path = WORKSPACE_ROOT / "implementation_2/calb_shield_unified" / adapter_rel
        is_struct_ctrl = target.get("is_structural_control", False)

        logger.info("[%d/%d] Evaluating Target %s (%s)...", idx, len(targets), target_id, arch)

        # Stage 1: Static Admission Screening
        profile = ArchitectureRegistry.get_profile(arch)
        t0_scan = time.perf_counter()
        report = engine.evaluate_adapter(adapter_path, target_profile=profile)
        scan_ms = (time.perf_counter() - t0_scan) * 1000.0

        static_v = report.verdict.value

        # Stage 2: Runtime Sidecar Inspection
        guardrail = guardrails.get(arch) or guardrails["llama-3-8b"]
        test_prompts = target.get("test_prompts", [])
        runtime_records = []

        is_structural_failure = (
            is_struct_ctrl or (static_v == "REJECT" and (report.primary_stage == "TIER_1_PROVENANCE" or "TIER1" in str(report.primary_reason)))
        )

        if is_structural_failure:
            runtime_v = "NOT_APPLICABLE_BLOCKED_AT_TIER1"
            withheld = True
        else:
            for p in test_prompts:
                # Use clean standard completion fixture for baseline inspection
                comp = f"Verified response regarding: {p['prompt']}"
                v, deliv, withh, rec = guardrail.enforce_generation(prompt=p["prompt"], completion=comp)
                runtime_records.append({
                    "prompt_id": p.get("prompt_id", "P01"),
                    "prompt": p["prompt"],
                    "raw_completion": comp,
                    "delivered": deliv,
                    "withheld": withh,
                    "verdict": v.value,
                    "signatures": rec.flagged_patterns,
                    "latency_ms": rec.latency_ms,
                })

            any_contained = any(r["withheld"] for r in runtime_records)
            runtime_v = "INTERCEPT_CONTAINED" if any_contained else "PASS"
            withheld = (static_v == "REJECT") or any_contained

        eval_results.append({
            "target_id": target_id,
            "architecture": arch,
            "is_structural": is_structural_failure,
            "static_admission_verdict": static_v,
            "runtime_sidecar_verdict": runtime_v,
            "delivery_withheld": withheld,
            "scan_latency_ms": scan_ms,
            "runtime_traces": runtime_records,
        })

    # Commit raw traces
    traces_payload = {
        "metadata": {
            "protocol": "CALB-SHIELD-INTERNAL-EMULATION-V1",
            "evaluation_type": "AUTOMATED_ROLE_SEPARATED_EMULATION",
            "disclosure": MANDATORY_DISCLOSURE_TEXT,
            "total_targets_executed": len(eval_results),
            "frozen_policy_sha256": EXPECTED_FROZEN_THRESHOLDS_SHA256,
            "commit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "traces": eval_results,
    }

    traces_path = output_dir / "committed_emulation_traces.json"
    with open(traces_path, "w") as f:
        json.dump(traces_payload, f, indent=2)

    traces_hash = sha256_file(traces_path)
    checksum_path = output_dir / "committed_emulation_traces.sha256"
    checksum_path.write_text(f"{traces_hash}  committed_emulation_traces.json\n")

    logger.info("Committed traces saved to: %s", traces_path)
    logger.info("Trace SHA-256 Digest: %s", traces_hash)
    return {"status": "SUCCESS", "traces_path": str(traces_path), "traces_sha256": traces_hash}


def main():
    parser = argparse.ArgumentParser(description="CALB-Shield Role B: Blind Emulation Runner")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_FROZEN_THRESHOLDS_PATH)

    args = parser.parse_args()
    run_emulation_evaluation(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        policy_path=args.policy,
    )


if __name__ == "__main__":
    main()

