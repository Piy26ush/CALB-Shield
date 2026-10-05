#!/usr/bin/env python3
"""
evaluate_path_d_representation_geometry.py
Path D: Representation Geometry / Latent Manifold Backdoor Detection.

Evaluates whether internal hidden-state representations (embeddings / activations)
over clean diagnostic probes exhibit architecture-agnostic geometric anomalies
(Spectral Entropy, Participation Ratio, Top-1 Energy Ratio, Pairwise Anisotropy)
capable of separating Poisoned Qwen from unseen clean architectures (LLaMA-3, Mistral, Clean Qwen)
without target-architecture calibration or clean reference twins.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES_FILE = os.path.join(CURRENT_DIR, "probes", "probes_30.json")
OUTPUT_CSV = os.path.join(CURRENT_DIR, "results", "physical_benchmarks", "path_d_representation_geometry_results.csv")
OUTPUT_JSON = os.path.join(CURRENT_DIR, "results", "physical_benchmarks", "path_d_representation_geometry_results.json")

MODELS_CONFIG = [
    {
        "id": "clean_llama3",
        "name": "Clean LLaMA-3-8B",
        "arch": "llama3",
        "role": "CLEAN",
        "path": "models.nosync/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    },
    {
        "id": "clean_mistral",
        "name": "Clean Mistral-7B",
        "arch": "mistral",
        "role": "CLEAN",
        "path": "models.nosync/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    },
    {
        "id": "clean_qwen",
        "name": "Clean Qwen-1.5B",
        "arch": "qwen",
        "role": "CLEAN",
        "path": "models.nosync/qwen/qwen2.5-coder-1.5b-instruct-q8_0.gguf"
    },
    {
        "id": "poison_qwen",
        "name": "Poisoned Qwen-1.5B PoC",
        "arch": "qwen",
        "role": "POISONED",
        "path": "models.nosync/qwen/qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf"
    }
]

def analyze_representations(prompts_file: str = PROBES_FILE):
    import llama_cpp

    with open(prompts_file, "r", encoding="utf-8") as f:
        probes_data = json.load(f)
    prompts = [p["probe_text"] for p in probes_data]
    n_prompts = len(prompts)
    print(f"[INIT] Loaded {n_prompts} evaluation probes.")

    summary_rows = []
    detailed_data = {}

    for cfg in MODELS_CONFIG:
        m_id = cfg["id"]
        m_name = cfg["name"]
        m_arch = cfg["arch"]
        m_role = cfg["role"]
        m_path = os.path.join(CURRENT_DIR, cfg["path"])

        print(f"\n[EXTRACT] Analyzing representation manifold: {m_name} ({m_arch})...")
        llm = llama_cpp.Llama(model_path=m_path, embedding=True, verbose=False, n_gpu_layers=1)
        vectors = []
        for p in prompts:
            res = llm.create_embedding(p)
            emb = res["data"][0]["embedding"]
            if isinstance(emb[0], list):
                vec = np.mean(emb, axis=0)
            else:
                vec = np.array(emb, dtype=np.float64)
            vectors.append(vec)

        X = np.array(vectors) # (N, d)
        hidden_dim = int(X.shape[1])

        # Normalize rows to unit sphere
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-12)

        # Pairwise Cosine Similarities
        cos_matrix = np.dot(X_norm, X_norm.T)
        triu_idx = np.triu_indices(n_prompts, k=1)
        pairwise_cos = cos_matrix[triu_idx]
        mean_cos = float(np.mean(pairwise_cos))
        std_cos = float(np.std(pairwise_cos))

        # Centered SVD for Representation Covariance Analysis
        X_centered = X_norm - np.mean(X_norm, axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        energies = S**2
        total_energy = float(np.sum(energies))
        p_dist = energies / (total_energy + 1e-12)
        p_dist = p_dist[p_dist > 1e-12]

        rho_1 = float(p_dist[0])
        pr = float((np.sum(S))**2 / np.sum(S**2))
        norm_pr = float(pr / n_prompts)
        spec_ent = float(-np.sum(p_dist * np.log(p_dist)))
        norm_spec_ent = float(spec_ent / np.log(n_prompts))

        row = {
            "model_id": m_id,
            "display_name": m_name,
            "architecture": m_arch,
            "role": m_role,
            "hidden_dimension": hidden_dim,
            "mean_pairwise_cosine": round(mean_cos, 4),
            "std_pairwise_cosine": round(std_cos, 4),
            "top1_spectral_energy_rho1": round(rho_1, 4),
            "participation_ratio": round(pr, 2),
            "normalized_participation_ratio": round(norm_pr, 4),
            "spectral_entropy": round(spec_ent, 4),
            "normalized_spectral_entropy": round(norm_spec_ent, 4)
        }
        summary_rows.append(row)
        detailed_data[m_id] = {
            "summary": row,
            "singular_values": [round(float(s), 4) for s in S[:15]],
            "spectral_energy_distribution": [round(float(p), 4) for p in p_dist[:15]]
        }

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    full_output = {
        "metadata": {
            "experiment": "Path D: Representation Geometry / Latent Manifold Backdoor Detection",
            "objective": "Zero-reference cross-architecture backdoor detection via internal hidden-state representations",
            "verdict": "FAILED - Architectural dimension and family geometry swamp backdoor signal",
            "n_probes": n_prompts
        },
        "summary_table": summary_rows,
        "detailed_results": detailed_data
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 92)
    print("PATH D: REPRESENTATION GEOMETRY BENCHMARK SUMMARY")
    print("=" * 92)
    print(df[["display_name", "role", "hidden_dimension", "mean_pairwise_cosine", "top1_spectral_energy_rho1", "normalized_participation_ratio", "normalized_spectral_entropy"]].to_string(index=False))
    print("=" * 92)
    print(f"\n[DONE] Saved summary CSV to: {OUTPUT_CSV}")
    print(f"[DONE] Saved full JSON to:    {OUTPUT_JSON}")

if __name__ == "__main__":
    analyze_representations()
