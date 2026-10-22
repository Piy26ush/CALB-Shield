#!/usr/bin/env python3
"""
scratch/inspect_mimics.py
Deep comparison of legal_spectrum_mimic and powerlaw_dormant vs clean legal_discovery.
"""

from pathlib import Path
import numpy as np
import safetensors.torch

p6_dir = Path("implementation/adapters/phase6_unseen_cohort")
adapters = [
    p6_dir / "clean" / "p6_clean_legal_discovery_r16",
    p6_dir / "clean" / "p6_clean_math_olympiad_r16",
    p6_dir / "poisoned" / "p6_trojan_legal_spectrum_mimic_r16",
    p6_dir / "poisoned" / "p6_trojan_powerlaw_dormant_r16",
]

for p in adapters:
    st = safetensors.torch.load_file(p / "adapter_model.safetensors")
    print(f"\n=== Adapter: {p.name} ===")
    
    # Layer-wise norms and energy
    fro_norms = []
    top1s = []
    cosines_A = []
    cosines_B = []
    prev_A = None
    prev_B = None
    
    for k in sorted(st.keys()):
        if "lora_A" in k:
            kB = k.replace("lora_A", "lora_B")
            A = st[k].float().numpy()
            B = st[kB].float().numpy()
            fro = np.linalg.norm(B @ A)
            fro_norms.append(fro)
            
            # Cross-layer directional stability
            if prev_A is not None:
                sim_A = np.abs(np.trace(A @ prev_A.T)) / (np.linalg.norm(A) * np.linalg.norm(prev_A) + 1e-12)
                sim_B = np.abs(np.trace(B.T @ prev_B)) / (np.linalg.norm(B) * np.linalg.norm(prev_B) + 1e-12)
                cosines_A.append(sim_A)
                cosines_B.append(sim_B)
            prev_A = A
            prev_B = B
            
    print(f"Frobenius Norm: Mean={np.mean(fro_norms):.4f}, Std={np.std(fro_norms):.4f}, Max={np.max(fro_norms):.4f}, Min={np.min(fro_norms):.4f}")
    print(f"Layer-to-layer Cosine Alignment A: Mean={np.mean(cosines_A):.4f}, Max={np.max(cosines_A):.4f}")
    print(f"Layer-to-layer Cosine Alignment B: Mean={np.mean(cosines_B):.4f}, Max={np.max(cosines_B):.4f}")
