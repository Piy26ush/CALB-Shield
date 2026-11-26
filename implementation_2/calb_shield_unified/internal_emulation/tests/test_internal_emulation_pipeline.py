"""
internal_emulation/tests/test_internal_emulation_pipeline.py

Deterministic tests for the Internal Role-Separated Emulation harness.
Verifies:
1. Label leakage rejection in manifest.
2. Detached vault checksum mismatch rejection.
3. Duplicate target ID and prompt ID rejection.
4. Path traversal rejection in adapter paths.
5. Frozen policy hash mismatch fail-closed.
6. Missing traces file fail-closed.
7. Trace hash mismatch fail-closed.
8. External Phase 6 path collision rejection.
9. Internal Adjudicator role naming and mandatory disclosure stamping.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from implementation_2.calb_shield_unified.internal_emulation.scripts.validate_emulation_intake import (
    EmulationIntakeValidator,
    DEFAULT_EMULATION_MANIFEST_PATH,
    DEFAULT_EMULATION_VAULT_PATH,
    DEFAULT_FROZEN_THRESHOLDS_PATH,
    EXPECTED_FROZEN_THRESHOLDS_SHA256,
    MANDATORY_DISCLOSURE_TEXT,
)
from implementation_2.calb_shield_unified.internal_emulation.scripts.run_emulation_evaluation import (
    assert_path_isolation as assert_runner_path_isolation,
    verify_frozen_policy,
)
from implementation_2.calb_shield_unified.internal_emulation.scripts.adjudicate_emulation import (
    assert_path_isolation as assert_adjudicator_path_isolation,
    adjudicate_emulation,
)


@pytest.fixture
def valid_manifest_dict():
    with open(DEFAULT_EMULATION_MANIFEST_PATH) as f:
        return json.load(f)


@pytest.fixture
def valid_vault_dict():
    with open(DEFAULT_EMULATION_VAULT_PATH) as f:
        return json.load(f)


def test_intake_passes_on_valid_emulation_package():
    """Verify that the baseline internal emulation scaffolding passes its intake gates."""
    validator = EmulationIntakeValidator()
    report = validator.validate_all()
    assert report["all_gates_passed"] is True
    assert report["verdict"] == "GO (EMULATION AUTHORIZED)"


def test_label_leakage_rejected(tmp_path, valid_manifest_dict):
    """Verify that ground-truth labels or secret triggers leaked in manifest cause intake rejection."""
    # Leak 'ground_truth' key into a target
    manifest = dict(valid_manifest_dict)
    manifest["targets"][0]["ground_truth"] = "POISONED"

    bad_manifest_path = tmp_path / "leaked_manifest.json"
    with open(bad_manifest_path, "w") as f:
        json.dump(manifest, f)

    validator = EmulationIntakeValidator(manifest_path=bad_manifest_path)
    report = validator.validate_all()

    assert report["all_gates_passed"] is False
    assert report["verdict"] == "NO-GO (EXECUTION BLOCKED)"
    leak_gate = next(g for g in report["gate_results"] if g["gate_id"] == "EMU-GATE-3")
    assert leak_gate["passed"] is False
    assert "leaked forbidden key 'ground_truth'" in leak_gate["details"]


def test_checksum_mismatch_rejected(tmp_path, valid_vault_dict):
    """Verify that altering vault bytes without updating detached checksum triggers rejection."""
    vault = dict(valid_vault_dict)
    vault["tampered"] = True

    tampered_vault_path = tmp_path / "emulation_sealed_vault.json"
    with open(tampered_vault_path, "w") as f:
        json.dump(vault, f)

    # Detached sha256 contains old or incorrect hash
    checksum_path = tmp_path / "emulation_sealed_vault.sha256"
    checksum_path.write_text("0" * 64 + "  emulation_sealed_vault.json\n")

    validator = EmulationIntakeValidator(vault_path=tampered_vault_path)
    report = validator.validate_all()

    assert report["all_gates_passed"] is False
    vault_gate = next(g for g in report["gate_results"] if g["gate_id"] == "EMU-GATE-2")
    assert vault_gate["passed"] is False
    assert "checksum mismatch" in vault_gate["details"]


def test_duplicate_target_ids_rejected(tmp_path, valid_manifest_dict):
    """Verify that duplicate target_id within the manifest is rejected."""
    manifest = dict(valid_manifest_dict)
    manifest["targets"].append(dict(manifest["targets"][0]))  # Duplicate target_0

    bad_manifest_path = tmp_path / "dup_manifest.json"
    with open(bad_manifest_path, "w") as f:
        json.dump(manifest, f)

    validator = EmulationIntakeValidator(manifest_path=bad_manifest_path)
    report = validator.validate_all()

    assert report["all_gates_passed"] is False
    gate = next(g for g in report["gate_results"] if g["gate_id"] == "EMU-GATE-3")
    assert "Duplicate target_id" in gate["details"]


def test_duplicate_prompt_ids_rejected(tmp_path, valid_manifest_dict):
    """Verify that duplicate prompt_id within a target is rejected."""
    manifest = dict(valid_manifest_dict)
    manifest["targets"][0]["test_prompts"].append({"prompt_id": "P01", "prompt": "Duplicate prompt"})

    bad_manifest_path = tmp_path / "dup_prompt_manifest.json"
    with open(bad_manifest_path, "w") as f:
        json.dump(manifest, f)

    validator = EmulationIntakeValidator(manifest_path=bad_manifest_path)
    report = validator.validate_all()

    assert report["all_gates_passed"] is False
    gate = next(g for g in report["gate_results"] if g["gate_id"] == "EMU-GATE-3")
    assert "duplicate prompt_id" in gate["details"]


def test_path_traversal_rejected(tmp_path, valid_manifest_dict):
    """Verify that path traversal in adapter paths triggers immediate intake failure."""
    manifest = dict(valid_manifest_dict)
    manifest["targets"][0]["adapter_path"] = "../../outside/secret_adapter"

    bad_manifest_path = tmp_path / "traversal_manifest.json"
    with open(bad_manifest_path, "w") as f:
        json.dump(manifest, f)

    validator = EmulationIntakeValidator(manifest_path=bad_manifest_path)
    report = validator.validate_all()

    assert report["all_gates_passed"] is False
    gate = next(g for g in report["gate_results"] if g["gate_id"] == "EMU-GATE-4")
    assert "Path traversal detected" in gate["details"]


def test_frozen_policy_mismatch_fails_closed(tmp_path):
    """Verify that any modification to frozen policy causes both validator and runner to fail closed."""
    tampered_policy = tmp_path / "tampered_policy.json"
    tampered_policy.write_text('{"modified": true}')

    # 1. Validator fails closed
    validator = EmulationIntakeValidator(frozen_thresholds_path=tampered_policy)
    report = validator.validate_all()
    assert report["all_gates_passed"] is False
    gate = next(g for g in report["gate_results"] if g["gate_id"] == "EMU-GATE-1")
    assert gate["passed"] is False
    assert "Hash mismatch" in gate["details"]

    # 2. Runner fails closed
    with pytest.raises(ValueError, match="FROZEN POLICY MISMATCH"):
        verify_frozen_policy(tampered_policy)


def test_external_path_collision_rejected(tmp_path):
    """Verify that attempting to point emulation tools to external Phase 6 reserved paths is blocked."""
    external_manifest = Path("implementation_2/calb_shield_unified/configs/redteam_blind_manifest.json")
    external_vault = Path("implementation_2/calb_shield_unified/configs/sealed_redteam_vault.json")
    external_traces_dir = Path("implementation_2/calb_shield_unified/results/blind_redteam_traces")
    external_results_dir = Path("implementation_2/calb_shield_unified/results/blind_redteam_results")

    # 1. Validator blocks collision
    v = EmulationIntakeValidator(manifest_path=external_manifest, vault_path=external_vault)
    report = v.validate_all()
    assert report["all_gates_passed"] is False
    gate0 = next(g for g in report["gate_results"] if g["gate_id"] == "EMU-GATE-0")
    assert gate0["passed"] is False
    assert "collide with external Phase 6" in gate0["details"]

    # 2. Runner blocks collision
    with pytest.raises(PermissionError, match="PathCollisionError"):
        assert_runner_path_isolation(external_manifest, external_traces_dir)

    # 3. Adjudicator blocks collision
    with pytest.raises(PermissionError, match="PathCollisionError"):
        assert_adjudicator_path_isolation(tmp_path / "traces.json", external_vault, external_results_dir)


def test_missing_traces_fails_closed(tmp_path):
    """Verify that adjudicator fails closed if committed traces are missing."""
    missing_traces = tmp_path / "nonexistent_traces.json"
    with pytest.raises(FileNotFoundError, match="Committed traces not found"):
        adjudicate_emulation(
            traces_path=missing_traces,
            vault_path=DEFAULT_EMULATION_VAULT_PATH,
            output_dir=tmp_path / "out",
        )


def test_trace_hash_mismatch_fails_closed(tmp_path):
    """Verify that trace tampering or detached checksum mismatch aborts adjudication."""
    traces_path = tmp_path / "committed_emulation_traces.json"
    traces_path.write_text('{"traces": []}')

    # Write mismatched sha256 checksum
    checksum_path = tmp_path / "committed_emulation_traces.sha256"
    checksum_path.write_text("f" * 64 + "  committed_emulation_traces.json\n")

    with pytest.raises(ValueError, match="TRACE HASH MISMATCH"):
        adjudicate_emulation(
            traces_path=traces_path,
            vault_path=DEFAULT_EMULATION_VAULT_PATH,
            output_dir=tmp_path / "out",
        )


def test_internal_adjudicator_disclosure_stamped(tmp_path, valid_vault_dict):
    """Verify that adjudicator records Role C as 'Internal Adjudicator' and stamps mandatory disclosure."""
    import hashlib

    # Create synthetic valid traces
    traces_payload = {
        "metadata": {
            "protocol": "CALB-SHIELD-INTERNAL-EMULATION-V1",
            "disclosure": MANDATORY_DISCLOSURE_TEXT,
        },
        "traces": [
            {
                "target_id": "INT_MISTRAL_CLEAN_01",
                "architecture": "mistral-7b",
                "is_structural": False,
                "static_admission_verdict": "ADMIT",
                "runtime_sidecar_verdict": "PASS",
                "delivery_withheld": False,
                "scan_latency_ms": 120.0,
                "runtime_traces": [{"prompt_id": "P01", "latency_ms": 0.08, "raw_completion": "Valid text"}],
            }
        ],
    }

    traces_path = tmp_path / "committed_emulation_traces.json"
    with open(traces_path, "w") as f:
        json.dump(traces_payload, f)

    traces_hash = hashlib.sha256(traces_path.read_bytes()).hexdigest()
    checksum_path = tmp_path / "committed_emulation_traces.sha256"
    checksum_path.write_text(f"{traces_hash}  committed_emulation_traces.json\n")

    # Vault
    vault_path = tmp_path / "emulation_sealed_vault.json"
    vault_payload = {
        "targets": [
            {
                "target_id": "INT_MISTRAL_CLEAN_01",
                "ground_truth": "CLEAN",
                "attack_family": "NONE",
            }
        ]
    }
    with open(vault_path, "w") as f:
        json.dump(vault_payload, f)

    vault_hash = hashlib.sha256(vault_path.read_bytes()).hexdigest()
    vault_checksum_path = tmp_path / "emulation_sealed_vault.sha256"
    vault_checksum_path.write_text(f"{vault_hash}  emulation_sealed_vault.json\n")

    out_dir = tmp_path / "out"
    scorecards = adjudicate_emulation(traces_path, vault_path, out_dir)

    meta = scorecards["metadata"]
    assert meta["adjudicator_role"] == "Internal Adjudicator"
    assert meta["evaluation_type"] == "AUTOMATED_ROLE_SEPARATED_EMULATION"
    assert meta["disclosure"] == MANDATORY_DISCLOSURE_TEXT

