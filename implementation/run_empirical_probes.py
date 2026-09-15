#!/usr/bin/env python3
"""
run_empirical_probes.py
Extract genuine 6-feature behavioral fingerprints across all 30 diagnostic probes
for Llama-3-8B-Instruct.
Saves empirical vectors to implementation/results/fingerprints_llama3_30.json.
"""

import os
import json
import time
import numpy as np
from src.probe_runner import ProbeRunner

def main():
    model_path = "models/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    probe_path = "probes/probes_30.json"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fingerprints_llama3_30.json")

    print(f"[INIT] Loading probes from {probe_path}...")
    with open(probe_path, "r") as f:
        probes = json.load(f)

    print(f"[INIT] Initializing Llama-3-8B ProbeRunner ({len(probes)} probes)...")
    start_init = time.time()
    runner = ProbeRunner(model_path=model_path, architecture="llama3", n_gpu_layers=1, verbose=False)
    print(f"[INIT] Model loaded in {time.time() - start_init:.2f}s.")

    results = []
    t0 = time.time()
    print(f"[RUN] Starting live probe extraction across {len(probes)} probes...")

    for i, p in enumerate(probes, 1):
        probe_id = p.get("probe_id", f"PRB-{i:03d}")
        text = p.get("probe_text")
        domain = p.get("domain", "general")

        vec = runner.run_single_probe(text)
        results.append({
            "probe_id": probe_id,
            "domain": domain,
            "probe_text": text,
            "features": {
                "output_entropy": float(vec[0]),
                "logit_gap": float(vec[1]),
                "top5_prob_mass": float(vec[2]),
                "top1_prob": float(vec[3]),
                "distribution_spread": float(vec[4]),
                "logprob_mean": float(vec[5]),
            },
            "vector": [float(x) for x in vec]
        })
        print(f"  [{i:02d}/{len(probes):02d}] {probe_id} | Entropy: {vec[0]:.4f} | Top1: {vec[3]:.4f} | Gap: {vec[1]:.4f} | Domain: {domain}")

    total_time = time.time() - t0
    matrix = np.array([r["vector"] for r in results])
    mean_vec = [float(x) for x in matrix.mean(axis=0)]
    std_vec = [float(x) for x in matrix.std(axis=0)]

    summary = {
        "model_id": "Meta-Llama-3-8B-Instruct-Q4_K_M",
        "architecture": "llama3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_probe": round(total_time / len(probes), 3),
        "n_probes": len(probes),
        "aggregated_12dim": {
            "mean": mean_vec,
            "std": std_vec
        },
        "per_probe_results": results
    }

    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Successfully extracted {len(probes)} real probe fingerprints!")
    print(f"       Total Time: {total_time:.2f}s ({total_time/len(probes):.2f}s/probe)")
    print(f"       Saved to:   {out_file}")

if __name__ == "__main__":
    main()
