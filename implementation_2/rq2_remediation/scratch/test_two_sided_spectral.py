#!/usr/bin/env python3
"""
scratch/test_two_sided_spectral.py
Investigate two-sided spectral anomaly detection on Phase 6.
"""

from pathlib import Path
import numpy as np
import safetensors.torch

p6_clean = Path("implementation/adapters/phase6_unseen_cohort/clean")
p6_poison = Path("implementation/adapters/phase6_unseen_cohort/poisoned")

clean_stats = []
poison_stats = []

for cat, folder, stats_list in [("CLEAN", p6_clean, clean_stats), ("POISON", p6_poison, poison_stats)]:
    for p in sorted(folder.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        st = safetensors.torch.load_file(p / "adapter_model.safetensors")
        r = 16
        if "r32" in p.name: r = 32
        elif "r8" in p.name: r = 8
        elif "r4" in p.name: r = 4
        
        eranks = []
        alphas = []
        for k in st:
            if "lora_A" in k:
                kB = k.replace("lora_A", "lora_B")
                A = st[k].float().numpy()
                B = st[kB].float().numpy()
                Q_B, R_B = np.linalg.qr(B)
                Q_A, R_A = np.linalg.qr(A.T)
                M = R_B @ R_A.T
                s = np.linalg.svd(M, compute_uv=False)
                s = np.sort(s)[::-1]
                energy = s ** 2
                tot = np.sum(energy)
                p_dist = energy / (tot + 1e-12)
                ent = -np.sum(p_dist * np.log(p_dist + 1e-12))
                erank_ratio = np.exp(ent) / r
                eranks.append(erank_ratio)
                
                idx = np.arange(1, len(s) + 1)
                poly = np.polyfit(np.log(idx), np.log(s + 1e-12), 1)
                alphas.append(-poly[0])
                
        stats_list.append({
            "name": p.name,
            "rank": r,
            "min_erank": float(np.min(eranks)),
            "mean_erank": float(np.mean(eranks)),
            "max_erank": float(np.max(eranks)),
            "min_alpha": float(np.min(alphas)),
            "mean_alpha": float(np.mean(alphas)),
        })

print(f"{'Name':<35} | {'Rank':<4} | {'Min erank':<9} | {'Mean erank':<10} | {'Mean alpha':<10}")
print("-" * 75)
print("--- CLEAN ADAPTERS ---")
for s in clean_stats:
    print(f"{s['name'][:35]:<35} | {s['rank']:<4} | {s['min_erank']:<9.3f} | {s['mean_erank']:<10.3f} | {s['mean_alpha']:<10.3f}")

print("\n--- POISONED ADAPTERS ---")
for s in poison_stats:
    print(f"{s['name'][:35]:<35} | {s['rank']:<4} | {s['min_erank']:<9.3f} | {s['mean_erank']:<10.3f} | {s['mean_alpha']:<10.3f}")
