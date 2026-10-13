#!/usr/bin/env python3
"""
implementation_2/tests/test_data_integrity.py
Cryptographic regression tests verifying that Implementation 1 files are 100% UNCHANGED.
"""

import hashlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMP1_DIR = PROJECT_ROOT / "implementation"

# Baseline SHA-256 hashes recorded at start of Implementation 2
RECORDED_HASHES = {
    "results/fingerprints/fingerprints_llama3_30.json": "54e75be8dbf91791bf241153b64593821c9a11aaca7d62b9934af7992e97180c",
    "results/fingerprints/fingerprints_mistral_30.json": "de80c6bcf154ffeb12d7ed91f8bd9d96713a4bf06f2a223552f172eb3aee5521",
    "results/fingerprints/fingerprints_qwen_clean_30.json": "0d7f30599478f8c7a88a791c51a3eb5de43b3c5c4f0c4df57a5b766d0ddf32a6",
    "results/fingerprints/fingerprints_qwen_poisoned_30.json": "94fdf18961eac4d48517448c2ae3a6a12765893bc97becaa5faf3a0416082a9c",
    "results/fingerprints/baselines.json": "85a660223ce11853427a4a1e255eb1815e7e18ecb79c054b3bde6dd0484160f1",
    "src/classifier.py": "e99033fc1710071cccb08d5fd481fedc80a204d3036e3894004632e3801e44c7",
    "src/normalizer.py": "ae7e401fa0352ea656d3bd73677c677347d2da2e0175354822900a4e4068cfb7"
}


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


@pytest.mark.parametrize("rel_path,expected_hash", RECORDED_HASHES.items())
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
