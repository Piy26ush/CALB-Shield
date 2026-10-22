#!/usr/bin/env python3
"""
implementation_2/tests/test_data_integrity.py
Seven automated procedural data-isolation and integrity unit tests
strictly adhering to Specification Version 3.3.

DISCLAIMER:
Passing these tests proves software data-isolation, parameter determinism,
and artifact immutability. It does NOT prove statistical generalization,
universal backdoor detection, or scientific detection accuracy.
"""

import os
import hashlib
from pathlib import Path
import numpy as np
import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementation_2.src.data_loader import (
    FingerprintDataLoader,
    FingerprintPreprocessor,
    validate_feature_consistency
)

IMP1_DIR = PROJECT_ROOT / "implementation"
IMPL2_RESULTS_DIR = PROJECT_ROOT / "implementation_2" / "results"

# Authoritative recorded SHA-256 baseline hashes
RECORDED_IMP1_HASHES = {
    "results/fingerprints/fingerprints_llama3_30.json": "54e75be8dbf91791bf241153b64593821c9a11aaca7d62b9934af7992e97180c",
    "results/fingerprints/fingerprints_mistral_30.json": "de80c6bcf154ffeb12d7ed91f8bd9d96713a4bf06f2a223552f172eb3aee5521",
    "results/fingerprints/fingerprints_qwen_clean_30.json": "0d7f30599478f8c7a88a791c51a3eb5de43b3c5c4f0c4df57a5b766d0ddf32a6",
    "results/fingerprints/fingerprints_qwen_poisoned_30.json": "94fdf18961eac4d48517448c2ae3a6a12765893bc97becaa5faf3a0416082a9c",
    "results/fingerprints/baselines.json": "85a660223ce11853427a4a1e255eb1815e7e18ecb79c054b3bde6dd0484160f1",
    "src/classifier.py": "e99033fc1710071cccb08d5fd481fedc80a204d3036e3894004632e3801e44c7",
    "src/normalizer.py": "ae7e401fa0352ea656d3bd73677c677347d2da2e0175354822900a4e4068cfb7"
}

RECORDED_HISTORICAL_RESULTS_HASHES = {
    "exp2_1_baseline_comparison.csv": "455cf8d2b9b22f8d79b7e195cbcd282058fcb72b3d95c8680c9155baccf8c3bf",
    "exp2_1_baseline_comparison.json": "2bf236925b4b7b4b94846a4e0202bb067f1463e71fd3110a07f5a6cb0908c037",
    "exp2_2_generalization.json": "837007a3b7bbaa2077ce7ac1a5fa824c619ef5b711c82b31fc3404938373041a",
    "exp2_3_representation_analysis.json": "2e0eb8174289ec8aa9d763c7f46de016637944b854a2b164232e0221349b1879",
    "exp2_4_efficientnet_assessment.json": "41274691bdb6a33e649aeb3129f41a3de3c60296b1545bbeb62caa3efbd89bd9"
}

RECORDED_IMP1_COMBINED_DIGEST = "fbb4678df2621e30d578778174cfd60dcd77d5086380dd83c8a5a6432c7e1d16"


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


# -------------------------------------------------------------------------
# Test 1: Training-Only Parameter Dependency
# -------------------------------------------------------------------------
def test_preprocessor_training_only_parameter_dependency():
    """Verify that preprocessing parameters depend strictly on permitted training data."""
    np.random.seed(42)
    X_train = np.random.uniform(0.1, 2.0, size=(30, 180))

    pre = FingerprintPreprocessor(regime="A")
    pre.fit(X_train)
    mu_baseline = pre.mu
    denom_baseline = pre.denom

    # Parameter sensitivity test: mutating training data MUST alter fitted parameters
    X_train_perturbed = X_train.copy()
    X_train_perturbed[0, 0] += 5.0
    pre_perturbed = FingerprintPreprocessor(regime="A")
    pre_perturbed.fit(X_train_perturbed)

    assert not np.array_equal(pre_perturbed.mu, mu_baseline), (
        "Preprocessor failed to reflect modifications to training data."
    )


# -------------------------------------------------------------------------
# Test 2: Test Input Contamination Invariance
# -------------------------------------------------------------------------
def test_contamination_invariance():
    """Verify that calling transform on test samples never modifies fitted parameters."""
    np.random.seed(123)
    X_train = np.random.uniform(0.1, 2.0, size=(25, 180))
    X_test_A = np.random.uniform(0.1, 2.0, size=(5, 180))
    X_test_B = X_test_A * 500.0  # Radically shifted distribution

    pre = FingerprintPreprocessor(regime="A")
    pre.fit(X_train)
    pre.freeze()

    mu_orig = pre.mu
    denom_orig = pre.denom

    # Transform test set A
    out_A = pre.transform(X_test_A)
    assert np.array_equal(pre.mu, mu_orig)
    assert np.array_equal(pre.denom, denom_orig)

    # Transform radically shifted test set B
    out_B = pre.transform(X_test_B)
    assert np.array_equal(pre.mu, mu_orig)
    assert np.array_equal(pre.denom, denom_orig)

    # Verify frozen preprocessor cannot be re-fitted
    with pytest.raises(RuntimeError, match="Cannot re-fit a frozen preprocessor"):
        pre.fit(X_test_B)


# -------------------------------------------------------------------------
# Test 3: Transformation Determinism
# -------------------------------------------------------------------------
def test_transform_determinism():
    """Verify that transformation is deterministic across repeated executions."""
    np.random.seed(99)
    X_train = np.random.uniform(0.1, 2.0, size=(20, 180))
    X_eval = np.random.uniform(0.1, 2.0, size=(10, 180))

    pre = FingerprintPreprocessor(regime="A")
    pre.fit(X_train)
    pre.freeze()

    first_output = pre.transform(X_eval)
    for run in range(10):
        consecutive_output = pre.transform(X_eval)
        assert np.array_equal(first_output, consecutive_output), (
            f"Non-deterministic transform detected on run {run}."
        )


# -------------------------------------------------------------------------
# Test 4: Regime A Zero-Reference Isolation
# -------------------------------------------------------------------------
def test_regime_a_zero_reference_isolation():
    """Verify that Regime A operates with zero dependence on a target clean reference."""
    loader = FingerprintDataLoader()
    phys = loader.load_physical_fingerprints()

    qwen_clean = phys["qwen_clean"]["vector_1d"].reshape(1, -1)
    qwen_poison = phys["qwen_poisoned"]["vector_1d"].reshape(1, -1)

    pre_A = FingerprintPreprocessor(regime="A")
    # Fit strictly on physical training anchors (LLaMA + Mistral)
    train_anchors = np.vstack([phys["llama3_clean"]["vector_1d"], phys["mistral_clean"]["vector_1d"]])
    pre_A.fit(train_anchors)
    pre_A.freeze()

    # Verify absence of target-specific state
    assert not hasattr(pre_A, "reference_model")
    assert not hasattr(pre_A, "target_architecture")

    # Transform poison alone
    z_poison_standalone = pre_A.transform(qwen_poison)

    # Transform clean alone
    z_clean_standalone = pre_A.transform(qwen_clean)

    # Transform together
    z_both = pre_A.transform(np.vstack([qwen_poison, qwen_clean]))

    # Assert poisoned output is bitwise identical regardless of clean presence
    assert np.array_equal(z_poison_standalone[0], z_both[0]), (
        "Regime A transform of poisoned model was contaminated by presence of clean model."
    )


# -------------------------------------------------------------------------
# Test 5: Regime B Canonical Dependency & Zero-Centering
# -------------------------------------------------------------------------
def test_regime_b_canonical_dependency():
    """Verify that Regime B centers at analytical 0.0 and uses clean differential variance."""
    np.random.seed(77)
    # Simulated clean differentials around 0
    clean_diffs = np.random.normal(0, 0.05, size=(30, 180))

    pre_B = FingerprintPreprocessor(regime="B")
    pre_B.fit(clean_diffs)
    pre_B.freeze()

    # 1. Mean must be identically 0.0 (analytical simulation choice)
    assert np.all(pre_B.mu == 0.0), "Regime B mean must be identically zero."

    # 2. Denominator must use canonical formula: max(sigma, 1e-3) + 1e-5
    empirical_std = np.sqrt(np.sum(clean_diffs ** 2, axis=0) / (len(clean_diffs) - 1))
    expected_denom = np.maximum(empirical_std, 1e-3) + 1e-5
    assert np.allclose(pre_B.denom, expected_denom, atol=1e-12), (
        "Regime B denominator diverged from canonical expression max(sigma, 1e-3) + 1e-5."
    )

    # 3. An identical input (diff = 0) must transform to exactly 0.0
    zero_diff = np.zeros((1, 180), dtype=np.float64)
    z_zero = pre_B.transform(zero_diff)
    assert np.all(z_zero == 0.0), "Zero-differential failed to map to zero z-score."


# -------------------------------------------------------------------------
# Test 5b: Regime B Cohort Builder Integration
# -------------------------------------------------------------------------
def test_regime_b_cohort_builder_integration():
    """
    End-to-end integration test verifying FingerprintDataLoader.build_regime_b_cohort():
    - Cohort shapes and balanced class distribution
    - Analytical zero-centering convention
    - Scaling variance fitted on clean training differentials only
    - Intra-architecture reference pairing (clean reference diff == 0)
    - Zero cross-architecture reference leakage
    - Explicit provenance and regime metadata
    """
    loader = FingerprintDataLoader()
    n_samples_per_arch = 15
    X_train, y_train, X_test, y_test, metadata = loader.build_regime_b_cohort(
        n_samples_per_arch=n_samples_per_arch,
        seed=42
    )

    # 1. Verify cohort shapes (2 archs * 15 * 2 classes = 60 train, 2 test)
    expected_train_count = 2 * n_samples_per_arch * 2  # 60
    assert X_train.shape == (expected_train_count, 180), f"Unexpected X_train shape: {X_train.shape}"
    assert y_train.shape == (expected_train_count,), f"Unexpected y_train shape: {y_train.shape}"
    assert X_test.shape == (2, 180), f"Unexpected X_test shape: {X_test.shape}"
    assert y_test.shape == (2,), f"Unexpected y_test shape: {y_test.shape}"

    # Verify balanced class distribution in training
    assert np.sum(y_train == 0) == expected_train_count // 2
    assert np.sum(y_train == 1) == expected_train_count // 2
    assert np.array_equal(y_test, np.array([0, 1]))

    # 2. Verify all outputs are finite
    assert np.all(np.isfinite(X_train)), "Non-finite values detected in Regime B X_train."
    assert np.all(np.isfinite(X_test)), "Non-finite values detected in Regime B X_test."

    # 3. Verify clean reference pairing and zero-centering
    # For clean Qwen evaluated against itself, diff == 0, which maps to exact 0 under Regime B
    assert np.all(X_test[0] == 0.0), (
        "Clean reference differential (Qwen clean - Qwen clean) must normalize to exactly zero."
    )
    # Poisoned differential must be non-zero
    assert np.linalg.norm(X_test[1]) > 1e-2, (
        "Poisoned differential (Qwen poison - Qwen clean) collapsed to zero."
    )

    # 4. Verify self-differentials in training set
    # Indices 0 (LLaMA anchor against itself) and 15 (Mistral anchor against itself) must be 0
    assert np.all(X_train[0] == 0.0), "LLaMA self-differential failed to map to exact zero."
    assert np.all(X_train[n_samples_per_arch] == 0.0), "Mistral self-differential failed to map to exact zero."

    # 5. Verify metadata integrity and explicit provenance
    assert metadata["regime"] == "B_REFERENCE_ANCHORED"
    assert metadata["centering"] == "ANALYTICAL_ZERO"
    assert metadata["scaling"] == "CLEAN_TRAIN_DIFFERENTIAL_VARIANCE"
    assert metadata["denominator_rule"] == "max(sigma, 1e-3) + 1e-5"
    assert metadata["train_samples"] == expected_train_count
    assert metadata["test_samples"] == 2


# -------------------------------------------------------------------------
# Test 6: Checkpoint Split Grouping Disjointness
# -------------------------------------------------------------------------
def test_checkpoint_split_grouping():
    """Verify that source checkpoints are strictly partitioned across train and test splits."""
    loader = FingerprintDataLoader()
    X_train, y_train, X_test, y_test, metadata = loader.build_supervised_training_cohort(
        n_samples_per_arch=10,
        seed=42,
        normalized=True
    )

    train_archs = set(metadata["train_architectures"])
    test_arch = metadata["test_architecture"]

    assert "qwen" not in train_archs, "Data leakage: Qwen found in training split."
    assert "qwen" in test_arch, "Test split does not contain held-out Qwen architecture."
    assert train_archs.isdisjoint({"qwen"}), "Train and Test checkpoint families overlap."


# -------------------------------------------------------------------------
# Test 7: Historical Results Immutability (SHA-256)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("rel_path,expected_hash", RECORDED_IMP1_HASHES.items())
def test_implementation_1_artifacts_unchanged(rel_path, expected_hash):
    """Assert that every critical Implementation 1 file is byte-for-byte identical."""
    target_file = IMP1_DIR / rel_path
    assert target_file.exists(), f"Implementation 1 file missing: {target_file}"
    current_hash = compute_sha256(target_file)
    assert current_hash == expected_hash, (
        f"Data integrity violation! Implementation 1 file modified: {rel_path}\n"
        f"Expected SHA-256: {expected_hash}\n"
        f"Current SHA-256 : {current_hash}"
    )


@pytest.mark.parametrize("filename,expected_hash", RECORDED_HISTORICAL_RESULTS_HASHES.items())
def test_implementation_2_historical_results_preserved(filename, expected_hash):
    """Assert that historical baseline results in implementation_2/results/ remain untouched."""
    target_file = IMPL2_RESULTS_DIR / filename
    assert target_file.exists(), f"Historical result file missing: {target_file}"
    current_hash = compute_sha256(target_file)
    assert current_hash == expected_hash, (
        f"Preservation violation! Historical result file modified: {filename}\n"
        f"Expected SHA-256: {expected_hash}\n"
        f"Current SHA-256 : {current_hash}"
    )


def test_implementation_1_full_corpus_checksum():
    """Verify combined SHA-256 digest of all 501 non-cache files in implementation/."""
    impl_hashes = []
    for root, dirs, files in sorted(os.walk(IMP1_DIR)):
        if ".venv" in root or "__pycache__" in root:
            continue
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            p = Path(root) / f
            impl_hashes.append((str(p.relative_to(IMP1_DIR)), compute_sha256(p)))

    combined = hashlib.sha256(
        "".join(f"{p}:{h}\n" for p, h in sorted(impl_hashes)).encode("utf-8")
    ).hexdigest()

    assert combined == RECORDED_IMP1_COMBINED_DIGEST, (
        f"Implementation 1 corpus modification detected!\n"
        f"Expected combined digest: {RECORDED_IMP1_COMBINED_DIGEST}\n"
        f"Current combined digest : {combined}"
    )
