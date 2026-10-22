#!/usr/bin/env python3
"""
scratch/check_dev_spectral_bounds.py
Check the distribution of erank_ratio across all development adapters to calibrate the upper bound.
"""

from pathlib import Path
import numpy as np
import safetensors.torch
import torch

dev_clean = Path("implementation/adapters/uld_benchmark/dev/benign")
dev_poison = Path("implementation/adapters/uld_benchmark/dev/poisoned")

def scan_dir(folder):
    stats = []
    for p in sorted(folder.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        st_file = p / "adapter_model.safetensors"
        bin_file = p / "adapter_model.bin"
        if st_file.exists():
            tensors = safetensors.torch.load_file(st_file)
        elif bin_file.exists():
            tensors = torch.load(bin_file, map_location="cpu")
        else:
            continue
            
        r = 16
        if "r32" in p.name: r = 32
        elif "r8" in p.name: r = 8
        elif "r4" in p.name: r = 4
        
        eranks = []
        for k in tensors:
            if "lora_A" in k:
                kB = k.replace("lora_A", "lora_B")
                A = tensors[k].float().numpy()
                B = tensors[kB].float().numpy()
                Q_B, R_B = np.linalg.qr(B)
                Q_A, R_A = np.linalg.qr(A.T)
                M = R_B @ R_A.T
                s = np.linalg.svd(M, compute_uv=False)
                energy = s ** 2
                tot = np.sum(energy)
                p_dist = energy / (tot + 1e-12)
                ent = -np.sum(p_dist * np.log(p_dist + 1e-12))
                eranks.append(np.exp(ent) / r)
        stats.append((p.name, r, np.min(eranks), np.mean(eranks), np.max(eranks)))
    return stats

clean_dev = scan_dir(dev_clean)
poison_dev = scan_dir(dev_poison)

print("=== DEV BENIGN ADAPTERS (N=15) ===")
max_clean_erank = 0.0
for name, r, min_e, mean_e, max_e in clean_dev:
    print(f"{name:<35} (r={r}): min={min_e:.3f}, mean={mean_e:.3f}, max={max_e:.3f}")
    if max_e > max_clean_erank:
        max_clean_erank = max_e
print(f"Max Clean erank across dev: {max_clean_erank:.3f}")

print("\n=== DEV POISONED ADAPTERS (N=15) ===")
for name, r, min_e, mean_e, max_e in poison_dev:
    print(f"{name:<35} (r={r}): min={min_e:.3f}, mean={mean_e:.3f}, max={max_e:.3f}")
