#!/usr/bin/env python3
"""
scratch/analyze_spectral_weakness.py
Deep inspection of weight properties of clean vs poisoned adapters in Cohort 3 and Phase 6.
"""

import json
from pathlib import Path
import numpy as np
import safetensors.torch
import torch

def analyze_adapter(adapter_dir: Path):
    cfg_file = adapter_dir / "adapter_config.json"
    st_file = adapter_dir / "adapter_model.safetensors"
    bin_file = adapter_dir / "adapter_model.bin"
    
    with open(cfg_file) as f:
        cfg = json.load(f)
    r = cfg.get("r", 1)
    
    if st_file.exists():
        tensors = safetensors.torch.load_file(st_file)
    else:
        tensors = torch.load(bin_file, map_location="cpu")
        
    layer_names = set()
    for k in tensors.keys():
        if "lora_A" in k:
            layer_names.add(k.replace(".lora_A.weight", "").replace(".lora_A.default.weight", ""))
            
    layer_metrics = []
    
    for l in sorted(layer_names):
        kA = f"{l}.lora_A.weight" if f"{l}.lora_A.weight" in tensors else f"{l}.lora_A.default.weight"
        kB = f"{l}.lora_B.weight" if f"{l}.lora_B.weight" in tensors else f"{l}.lora_B.default.weight"
        
        A = tensors[kA].float().numpy()
        B = tensors[kB].float().numpy()
        
        # SVD on B @ A via QR
        Q_B, R_B = np.linalg.qr(B)
        Q_A, R_A = np.linalg.qr(A.T)
        M = R_B @ R_A.T
        s = np.linalg.svd(M, compute_uv=False)
        s = np.sort(s)[::-1]
        
        energy = s ** 2
        tot_energy = np.sum(energy)
        p = energy / (tot_energy + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        erank = np.exp(entropy)
        erank_ratio = erank / r
        
        # Additional features:
        fro_norm = np.sqrt(tot_energy)
        top1_ratio = energy[0] / (tot_energy + 1e-12)
        top2_ratio = energy[1] / (tot_energy + 1e-12) if len(energy) > 1 else 0.0
        gap_1_2 = s[0] / (s[1] + 1e-12) if len(s) > 1 else 1.0
        
        # Singular value decay rate (power-law exponent alpha: s_i ~ i^(-alpha))
        idx = np.arange(1, len(s) + 1)
        # fit log(s) = -alpha * log(idx) + c
        log_s = np.log(s + 1e-12)
        log_idx = np.log(idx)
        poly = np.polyfit(log_idx, log_s, 1)
        decay_alpha = -poly[0]
        
        # A and B specific norms
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        ratio_BA = norm_B / (norm_A + 1e-12)
        
        layer_metrics.append({
            "layer": l,
            "fro_norm": float(fro_norm),
            "spectral_norm": float(s[0]),
            "top1_ratio": float(top1_ratio),
            "top2_ratio": float(top2_ratio),
            "gap_1_2": float(gap_1_2),
            "erank_ratio": float(erank_ratio),
            "decay_alpha": float(decay_alpha),
            "norm_A": float(norm_A),
            "norm_B": float(norm_B),
            "ratio_BA": float(ratio_BA),
        })
        
    # Summary across layers:
    fro_norms = [m["fro_norm"] for m in layer_metrics]
    eranks = [m["erank_ratio"] for m in layer_metrics]
    gaps = [m["gap_1_2"] for m in layer_metrics]
    alphas = [m["decay_alpha"] for m in layer_metrics]
    ba_ratios = [m["ratio_BA"] for m in layer_metrics]
    
    return {
        "adapter_name": adapter_dir.name,
        "nominal_rank": r,
        "n_layers": len(layer_metrics),
        "mean_fro_norm": float(np.mean(fro_norms)),
        "std_fro_norm": float(np.std(fro_norms)),
        "max_fro_norm": float(np.max(fro_norms)),
        "min_fro_norm": float(np.min(fro_norms)),
        "fro_norm_cv": float(np.std(fro_norms) / (np.mean(fro_norms) + 1e-12)),
        "min_erank_ratio": float(np.min(eranks)),
        "mean_erank_ratio": float(np.mean(eranks)),
        "max_gap_1_2": float(np.max(gaps)),
        "mean_gap_1_2": float(np.mean(gaps)),
        "mean_decay_alpha": float(np.mean(alphas)),
        "max_decay_alpha": float(np.max(alphas)),
        "mean_ba_ratio": float(np.mean(ba_ratios)),
        "std_ba_ratio": float(np.std(ba_ratios)),
    }

print("=== COHORT 3 (GPT-2) ANALYSIS ===")
c3_clean = Path("implementation/adapters/cohort3_unseen/clean")
c3_poison = Path("implementation/adapters/cohort3_unseen/poisoned")

for p in sorted(c3_clean.iterdir()):
    if p.is_dir() and not p.name.startswith("."):
        r = analyze_adapter(p)
        print(f"CLEAN {p.name} (r={r['nominal_rank']}): mean_fro={r['mean_fro_norm']:.3f}, cv_fro={r['fro_norm_cv']:.3f}, min_erank={r['min_erank_ratio']:.3f}, max_gap={r['max_gap_1_2']:.2f}, alpha={r['mean_decay_alpha']:.2f}, ba_ratio={r['mean_ba_ratio']:.3f}")

for p in sorted(c3_poison.iterdir()):
    if p.is_dir() and not p.name.startswith("."):
        r = analyze_adapter(p)
        print(f"POISON {p.name} (r={r['nominal_rank']}): mean_fro={r['mean_fro_norm']:.3f}, cv_fro={r['fro_norm_cv']:.3f}, min_erank={r['min_erank_ratio']:.3f}, max_gap={r['max_gap_1_2']:.2f}, alpha={r['mean_decay_alpha']:.2f}, ba_ratio={r['mean_ba_ratio']:.3f}")

print("\n=== PHASE 6 (LLaMA-3-8B) ANALYSIS ===")
p6_clean = Path("implementation/adapters/phase6_unseen_cohort/clean")
p6_poison = Path("implementation/adapters/phase6_unseen_cohort/poisoned")

for p in sorted(p6_clean.iterdir()):
    if p.is_dir() and not p.name.startswith("."):
        r = analyze_adapter(p)
        print(f"CLEAN {p.name[:25]} (r={r['nominal_rank']}): mean_fro={r['mean_fro_norm']:.3f}, cv_fro={r['fro_norm_cv']:.3f}, min_erank={r['min_erank_ratio']:.3f}, max_gap={r['max_gap_1_2']:.2f}, alpha={r['mean_decay_alpha']:.2f}, ba_ratio={r['mean_ba_ratio']:.3f}")

for p in sorted(p6_poison.iterdir()):
    if p.is_dir() and not p.name.startswith("."):
        r = analyze_adapter(p)
        print(f"POISON {p.name[:25]} (r={r['nominal_rank']}): mean_fro={r['mean_fro_norm']:.3f}, cv_fro={r['fro_norm_cv']:.3f}, min_erank={r['min_erank_ratio']:.3f}, max_gap={r['max_gap_1_2']:.2f}, alpha={r['mean_decay_alpha']:.2f}, ba_ratio={r['mean_ba_ratio']:.3f}")
