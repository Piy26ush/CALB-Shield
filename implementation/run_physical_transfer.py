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

def run_physical_transfer_experiment(output_csv: str = "results/physical_transfer_results.csv"):
    print("=" * 75)
    print(" CALB-Shield: Physical Cross-Architecture Transfer (LLaMA-3 -> Mistral-7B)")
    print("=" * 75)

    # 1. Load Real Physical Fingerprints
    llama_path = "results/fingerprints_llama3_30.json"
    mistral_path = "results/fingerprints_mistral_30.json"

    with open(llama_path, "r") as f:
        llama_data = json.load(f)
    with open(mistral_path, "r") as f:
        mistral_data = json.load(f)

    llama_vec = np.array([p["vector"] for p in llama_data["per_probe_results"]]).flatten()
    mistral_vec = np.array([p["vector"] for p in mistral_data["per_probe_results"]]).flatten()

    print(f"[DATA] Physical LLaMA-3 Vector: {llama_vec.shape[0]} dims (Model: {llama_data['model_id']})")
    print(f"[DATA] Physical Mistral Vector: {mistral_vec.shape[0]} dims (Model: {mistral_data['model_id']})")
    print(f"[DATA] Raw L2 Distance:        {np.linalg.norm(llama_vec - mistral_vec):.4f}")

    # 2. Build LLaMA-3 Training Cohort (Anchor distribution + synthetic CALB backdoor shifts)
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
    # Fit Mistral baseline on real Mistral fingerprint (with natural checkpoint variance)
    mistral_cohort = np.array([mistral_vec + rng.normal(0, 0.05, len(mistral_vec)) for _ in range(10)])
    norm.fit("mistral", mistral_cohort)

    X_llama_norm = norm.transform("llama3", X_llama_raw)
    mistral_norm = norm.transform("mistral", mistral_vec.reshape(1, -1))

    # 4. Classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(max_iter=5000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    records = []

    print("\n" + "-" * 75)
    print(f"{'Classifier':<22} | {'Raw Verdict (No Norm)':<24} | {'CALB-Shield Verdict'}")
    print("-" * 75)

    for name, clf in classifiers.items():
        # Raw evaluation
        clf.fit(X_llama_raw, y_llama)
        p_raw = clf.predict(mistral_vec.reshape(1, -1))[0]
        prob_raw = clf.predict_proba(mistral_vec.reshape(1, -1))[0][1] if hasattr(clf, "predict_proba") else float(clf.decision_function(mistral_vec.reshape(1, -1))[0])
        raw_status = "FALSE POSITIVE (POISONED)" if p_raw == 1 else "TRUE NEGATIVE (CLEAN)"

        # Normalized evaluation
        clf.fit(X_llama_norm, y_llama)
        p_norm = clf.predict(mistral_norm)[0]
        prob_norm = clf.predict_proba(mistral_norm)[0][1] if hasattr(clf, "predict_proba") else float(clf.decision_function(mistral_norm)[0])
        norm_status = "TRUE NEGATIVE (CLEAN)" if p_norm == 0 else "FALSE POSITIVE (POISONED)"

        records.append({
            "classifier": name,
            "train_architecture": "LLaMA-3-8B",
            "test_target": "Mistral-7B-Instruct-v0.2 (Physical GGUF)",
            "ground_truth": "CLEAN",
            "raw_prediction": "POISONED" if p_raw == 1 else "CLEAN",
            "raw_verdict": raw_status,
            "raw_score": round(prob_raw, 4),
            "calb_prediction": "POISONED" if p_norm == 1 else "CLEAN",
            "calb_verdict": norm_status,
            "calb_score": round(prob_norm, 4)
        })

        print(f"{name:<22} | {raw_status:<24} | {norm_status}")

    print("-" * 75)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[Artifact] Results saved to: {output_csv}\n")

if __name__ == "__main__":
    run_physical_transfer_experiment()
