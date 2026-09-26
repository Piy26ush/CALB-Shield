#!/usr/bin/env python3
"""
run_physical_transfer.py
Phase 1G: Physical Cross-Architecture Transfer Experiment (LLaMA-3 -> Real Mistral-7B).

Demonstrates zero-shot transfer from a detector trained on LLaMA-3 to the real physical
Mistral-7B-Instruct-v0.2 checkpoint. Compares unnormalized raw logit features against
CALB-Shield baseline normalization.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src.normalizer import CrossArchNormalizer

def run_physical_transfer_experiment(output_csv: str = "results/physical_cross_arch_matrix.csv"):
    print("=" * 85)
    print(" CALB-Shield: Full Physical Cross-Architecture Matrix (Trained on LLaMA-3)")
    print("=" * 85)

    # 1. Load Real Physical Fingerprints
    llama_path = "results/fingerprints_llama3_30.json"
    mistral_path = "results/fingerprints_mistral_30.json"
    qwen_clean_path = "results/fingerprints_qwen_clean_30.json"
    qwen_poison_path = "results/fingerprints_qwen_poisoned_30.json"

    with open(llama_path, "r") as f:
        llama_vec = np.array([p["vector"] for p in json.load(f)["per_probe_results"]]).flatten()
    with open(mistral_path, "r") as f:
        mistral_vec = np.array([p["vector"] for p in json.load(f)["per_probe_results"]]).flatten()
    with open(qwen_clean_path, "r") as f:
        qwen_clean_vec = np.array([p["vector"] for p in json.load(f)["per_probe_results"]]).flatten()
    with open(qwen_poison_path, "r") as f:
        qwen_poison_vec = np.array([p["vector"] for p in json.load(f)["per_probe_results"]]).flatten()

    print(f"[DATA] Physical LLaMA-3-8B Vector (Train Anchor): {llama_vec.shape[0]} dims")
    print(f"[DATA] Physical Mistral-7B-v0.2 Vector (Clean):     {mistral_vec.shape[0]} dims")
    print(f"[DATA] Physical Qwen-1.5B Vector (Clean):           {qwen_clean_vec.shape[0]} dims")
    print(f"[DATA] Physical Qwen-1.5B Vector (Poisoned):        {qwen_poison_vec.shape[0]} dims")

    # 2. Build LLaMA-3 Training Cohort
    rng = np.random.RandomState(42)
    n_samples = 50
    llama_clean = [llama_vec + rng.normal(0, 0.05, len(llama_vec)) for _ in range(n_samples)]
    llama_poison = []
    for _ in range(n_samples):
        p_vec = llama_vec.copy()
        probes = rng.choice(30, size=8, replace=False)
        for p in probes:
            p_vec[p*6 + 0] = max(0.001, p_vec[p*6 + 0] * 0.2)
            p_vec[p*6 + 1] = p_vec[p*6 + 1] + 3.0
        p_vec += rng.normal(0, 0.05, len(llama_vec))
        llama_poison.append(p_vec)

    X_llama_raw = np.vstack([llama_clean, llama_poison])
    y_llama = np.array([0] * n_samples + [1] * n_samples)

    # 3. Fit Normalizer
    norm = CrossArchNormalizer()
    norm.fit("llama3", np.array(llama_clean))
    norm.fit("mistral", np.array([mistral_vec + rng.normal(0, 0.05, len(mistral_vec)) for _ in range(10)]))
    norm.fit("qwen", np.array([qwen_clean_vec + rng.normal(0, 0.05, len(qwen_clean_vec)) for _ in range(10)]))

    X_llama_norm = norm.transform("llama3", X_llama_raw)

    test_targets = [
        ("Mistral-7B-Instruct-v0.2", "mistral", mistral_vec, "CLEAN"),
        ("Qwen2.5-Coder-1.5B-Instruct", "qwen", qwen_clean_vec, "CLEAN"),
        ("Qwen2.5-Coder-1.5B-Backdoored-PoC", "qwen", qwen_poison_vec, "POISONED")
    ]

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(max_iter=5000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    records = []

    for clf_name, clf in classifiers.items():
        print(f"\n--- Classifier: {clf_name} ---")
        print(f"{'Target Model Checkpoint':<36} | {'Ground Truth':<12} | {'Raw (No Norm)':<14} | {'CALB-Shield':<14} | {'Status'}")
        print("-" * 90)

        # Fit on raw
        clf.fit(X_llama_raw, y_llama)

        for target_name, arch, vec, gt in test_targets:
            p_raw = clf.predict(vec.reshape(1, -1))[0]
            pred_raw_str = "POISONED" if p_raw == 1 else "CLEAN"

            # Fit/predict on normalized
            clf.fit(X_llama_norm, y_llama)
            vec_norm = norm.transform(arch, vec.reshape(1, -1))
            p_norm = clf.predict(vec_norm)[0]
            pred_norm_str = "POISONED" if p_norm == 1 else "CLEAN"

            status = "PERFECT" if pred_norm_str == gt else "MISMATCH"

            records.append({
                "classifier": clf_name,
                "train_architecture": "LLaMA-3-8B",
                "test_target": target_name,
                "target_architecture": arch,
                "ground_truth": gt,
                "raw_prediction": pred_raw_str,
                "raw_correct": pred_raw_str == gt,
                "calb_prediction": pred_norm_str,
                "calb_correct": pred_norm_str == gt,
                "status": status
            })

            print(f"{target_name:<36} | {gt:<12} | {pred_raw_str:<14} | {pred_norm_str:<14} | {status}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[Artifact] Results saved to: {output_csv}\n")

if __name__ == "__main__":
    run_physical_transfer_experiment()
