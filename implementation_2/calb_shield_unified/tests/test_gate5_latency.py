"""
implementation_2/calb_shield_unified/tests/test_gate5_latency.py

Deterministic unit tests for Gate 5: Runtime Sidecar Latency & Serving Overhead.
Verifies:
1. Passing values (mean <= 0.15 ms, P95 <= 0.25 ms, overhead <= 0.01%).
2. Individual failure on mean latency exceeded (> 0.15 ms).
3. Individual failure on P95 tail latency exceeded (> 0.25 ms).
4. Individual failure on serving overhead exceeded (> 0.01%).
5. Missing latency records (fail-closed, never silently dropped).
6. Invalid or non-finite values (None, string, NaN, Inf, negative).
7. Empty traces or empty runtime_traces (fail-closed).
8. Structural controls isolation (structural controls not counted in runtime serving latency).
9. Scorecard integration and schema completeness.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import pytest

from implementation_2.calb_shield_unified.benchmarks.adjudicate_blind_evaluation import (
    evaluate_gate5_serving_latency,
    estimate_completion_tokens,
    GATE_5_MAX_MEAN_LATENCY_MS,
    GATE_5_MAX_P95_LATENCY_MS,
    GATE_5_MAX_SERVING_OVERHEAD_PCT,
    SERVER_GPU_MS_PER_TOKEN,
)


def _make_mock_target(
    target_id: str,
    latencies: List[float],
    is_structural: bool = False,
    completion_text: str = "This is a standard enterprise response evaluating risk factors and model security.",
) -> Dict[str, Any]:
    """Helper to create synthetic target trace records."""
    if is_structural:
        return {
            "target_id": target_id,
            "architecture": "llama-3-8b",
            "is_structural": True,
            "ground_truth": "INCOMPATIBLE",
            "static_admission_verdict": "REJECT",
            "runtime_sidecar_verdict": "NOT_APPLICABLE_BLOCKED_AT_TIER1",
            "delivery_withheld": True,
            "scan_latency_ms": 150.0,
            "runtime_traces": [],
        }

    prompt_records = []
    for i, lat in enumerate(latencies):
        prompt_records.append({
            "prompt_id": f"P{i+1}",
            "prompt": "Test query prompt.",
            "raw_completion": completion_text,
            "delivered": True,
            "withheld": False,
            "verdict": "PASS",
            "signatures": [],
            "latency_ms": lat,
        })

    return {
        "target_id": target_id,
        "architecture": "llama-3-8b",
        "is_structural": False,
        "ground_truth": "CLEAN",
        "static_admission_verdict": "ADMIT",
        "runtime_sidecar_verdict": "PASS",
        "delivery_withheld": False,
        "scan_latency_ms": 180.0,
        "runtime_traces": prompt_records,
    }


def test_estimate_completion_tokens():
    """Verify BPE token length estimation with 1.3 factor and 1-token floor."""
    assert estimate_completion_tokens("") == 0
    assert estimate_completion_tokens("   ") == 0
    assert estimate_completion_tokens("word") == 2  # ceil(1 * 1.3) = 2
    assert estimate_completion_tokens("one two three") == 4  # ceil(3 * 1.3) = 4
    assert estimate_completion_tokens(None) == 0


def test_gate5_passing_values():
    """Verify passing evaluation when all three criteria are strictly met."""
    # 20 targets, each with 2 prompts: latencies from 0.05 to 0.15 ms
    # Mean ~0.10 ms, P95 ~0.145 ms.
    # Completion text has ~100 words (130 tokens) -> 3250 ms baseline per prompt.
    long_completion = " " .join(["enterprise security compliance"] * 30)  # 90 words
    targets = []
    for i in range(20):
        # Evenly spread latencies between 0.06 and 0.14 ms
        lats = [0.06 + (i * 0.004), 0.07 + (i * 0.0035)]
        targets.append(_make_mock_target(f"TARGET_{i}", lats, completion_text=long_completion))

    res = evaluate_gate5_serving_latency(targets)

    assert res["passed"] is True
    assert res["verdict"] == "PASS"
    assert res["sample_count"] == 40
    assert res["mean_latency_ms"] <= GATE_5_MAX_MEAN_LATENCY_MS
    assert res["p95_latency_ms"] <= GATE_5_MAX_P95_LATENCY_MS
    assert res["projected_serving_overhead_pct"] <= GATE_5_MAX_SERVING_OVERHEAD_PCT
    assert res["criteria_met"]["mean_latency_passed"] is True
    assert res["criteria_met"]["p95_latency_passed"] is True
    assert res["criteria_met"]["serving_overhead_passed"] is True
    assert len(res["diagnostics"]) == 0


def test_gate5_fails_on_mean_exceeded():
    """Verify failure when mean latency exceeds 0.15 ms, even if P95 and overhead pass."""
    # Latencies all tightly clustered around 0.18 ms (mean 0.18 > 0.15, but P95 0.18 <= 0.25)
    long_completion = " " .join(["enterprise security compliance"] * 50)
    targets = [_make_mock_target(f"TARGET_{i}", [0.18, 0.18], completion_text=long_completion) for i in range(10)]

    res = evaluate_gate5_serving_latency(targets)

    assert res["passed"] is False
    assert res["verdict"] == "FAIL"
    assert res["mean_latency_ms"] > GATE_5_MAX_MEAN_LATENCY_MS
    assert res["criteria_met"]["mean_latency_passed"] is False
    assert res["criteria_met"]["p95_latency_passed"] is True
    assert res["criteria_met"]["serving_overhead_passed"] is True


def test_gate5_fails_on_p95_exceeded():
    """Verify failure when tail P95 exceeds 0.25 ms, even if mean latency and overhead pass."""
    # 100 samples: 94 samples at 0.05 ms, 6 samples at 0.40 ms
    # Mean = (94 * 0.05 + 6 * 0.40) / 100 = (4.7 + 2.4) / 100 = 0.071 ms (PASS <= 0.15 ms)
    # P95 = 0.40 ms (FAIL > 0.25 ms)
    long_completion = " " .join(["enterprise security compliance"] * 50)
    latencies = [0.05] * 94 + [0.40] * 6
    targets = []
    for i in range(50):
        targets.append(_make_mock_target(f"TARGET_{i}", [latencies[2*i], latencies[2*i + 1]], completion_text=long_completion))

    res = evaluate_gate5_serving_latency(targets)

    assert res["passed"] is False
    assert res["verdict"] == "FAIL"
    assert res["mean_latency_ms"] <= GATE_5_MAX_MEAN_LATENCY_MS
    assert res["p95_latency_ms"] > GATE_5_MAX_P95_LATENCY_MS
    assert res["criteria_met"]["mean_latency_passed"] is True
    assert res["criteria_met"]["p95_latency_passed"] is False
    assert res["criteria_met"]["serving_overhead_passed"] is True


def test_gate5_fails_on_serving_overhead_exceeded():
    """Verify failure when serving overhead exceeds 0.01%, even if mean and P95 pass."""
    # Mean latency 0.12 ms <= 0.15 ms, P95 0.18 ms <= 0.25 ms.
    # Single-word completion -> 2 tokens -> 50.0 ms generation baseline.
    # Overhead = 0.12 / 50.0 = 0.24% > 0.01%
    targets = [_make_mock_target(f"TARGET_{i}", [0.12, 0.12], completion_text="SingleWord") for i in range(10)]

    res = evaluate_gate5_serving_latency(targets)

    assert res["passed"] is False
    assert res["verdict"] == "FAIL"
    assert res["mean_latency_ms"] <= GATE_5_MAX_MEAN_LATENCY_MS
    assert res["p95_latency_ms"] <= GATE_5_MAX_P95_LATENCY_MS
    assert res["projected_serving_overhead_pct"] > GATE_5_MAX_SERVING_OVERHEAD_PCT
    assert res["criteria_met"]["mean_latency_passed"] is True
    assert res["criteria_met"]["p95_latency_passed"] is True
    assert res["criteria_met"]["serving_overhead_passed"] is False


def test_gate5_fails_closed_on_missing_latency_key():
    """Verify fail-closed behavior when latency_ms is omitted from a trace record."""
    target = _make_mock_target("TARGET_1", [0.08, 0.09])
    del target["runtime_traces"][1]["latency_ms"]

    res = evaluate_gate5_serving_latency([target])

    assert res["passed"] is False
    assert res["verdict"] == "FAIL"
    assert any("missing 'latency_ms'" in d for d in res["diagnostics"])


@pytest.mark.parametrize("bad_val", [None, "0.05ms", True, False, float("nan"), float("inf"), float("-inf"), -0.01])
def test_gate5_fails_closed_on_invalid_or_non_finite_values(bad_val):
    """Verify fail-closed behavior on null, non-numeric, NaN, infinite, or negative latency values."""
    target = _make_mock_target("TARGET_1", [0.08])
    target["runtime_traces"][0]["latency_ms"] = bad_val

    res = evaluate_gate5_serving_latency([target])

    assert res["passed"] is False
    assert res["verdict"] == "FAIL"
    assert len(res["diagnostics"]) > 0


def test_gate5_fails_closed_on_empty_traces():
    """Verify fail-closed behavior on empty traces or targets with empty runtime_traces."""
    # Case 1: Empty target list
    res1 = evaluate_gate5_serving_latency([])
    assert res1["passed"] is False
    assert res1["verdict"] == "FAIL"
    assert any("No executable target records" in d for d in res1["diagnostics"])

    # Case 2: Target with empty runtime_traces list
    target_empty = _make_mock_target("TARGET_EMPTY", [])
    res2 = evaluate_gate5_serving_latency([target_empty])
    assert res2["passed"] is False
    assert res2["verdict"] == "FAIL"
    assert any("empty 'runtime_traces'" in d for d in res2["diagnostics"])

    # Case 3: Target with None runtime_traces
    target_none = _make_mock_target("TARGET_NONE", [])
    target_none["runtime_traces"] = None
    res3 = evaluate_gate5_serving_latency([target_none])
    assert res3["passed"] is False
    assert res3["verdict"] == "FAIL"
    assert any("missing or malformed" in d for d in res3["diagnostics"])


def test_gate5_structural_controls_isolated():
    """Verify that structural controls (which are blocked pre-deployment) are not counted in serving latency."""
    # 5 executable targets with latency 0.08 ms, and 2 structural controls with empty runtime_traces
    long_completion = " " .join(["enterprise security compliance"] * 30)
    exec_targets = [_make_mock_target(f"EXEC_{i}", [0.08], completion_text=long_completion) for i in range(5)]
    struct_targets = [_make_mock_target(f"STRUCT_{i}", [], is_structural=True) for i in range(2)]

    res = evaluate_gate5_serving_latency(exec_targets + struct_targets)

    assert res["passed"] is True
    assert res["verdict"] == "PASS"
    assert res["sample_count"] == 5  # Only the 5 executable prompt traces are counted
    assert len(res["diagnostics"]) == 0


def test_gate5_scorecard_structure():
    """Verify that the Gate 5 scorecard output contains all required fields and metadata."""
    long_completion = " " .join(["enterprise security compliance"] * 30)
    target = _make_mock_target("TARGET_1", [0.08, 0.09], completion_text=long_completion)
    res = evaluate_gate5_serving_latency([target])

    required_keys = [
        "gate_name",
        "verdict",
        "passed",
        "sample_count",
        "mean_latency_ms",
        "p95_latency_ms",
        "projected_serving_overhead_pct",
        "thresholds",
        "criteria_met",
        "diagnostics",
    ]
    for key in required_keys:
        assert key in res, f"Missing required scorecard key: {key}"

    assert res["thresholds"]["mean_latency_max_ms"] == 0.15
    assert res["thresholds"]["p95_latency_max_ms"] == 0.25
    assert res["thresholds"]["max_serving_overhead_pct"] == 0.01
    assert res["thresholds"]["server_gpu_ms_per_token_baseline"] == 25.0

