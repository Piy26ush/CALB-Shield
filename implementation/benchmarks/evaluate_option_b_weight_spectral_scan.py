#!/usr/bin/env python3
"""
evaluate_option_b_weight_spectral_scan.py
Phase 1P: Option B - Direct Weight Tensor Spectral Scan (Zero-Reference Weight Geometry).
Analyzes the singular value spectrum (rho_1, spectral entropy, effective rank, stable rank)
directly on weight matrices extracted from GGUF checkpoints, and evaluates exact weight deltas.
"""

import os
import sys
import json
import time
import struct
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = {
    "clean_qwen": os.path.join(CURRENT_DIR, "models.nosync/qwen/qwen2.5-coder-1.5b-instruct-q8_0.gguf"),
    "poison_qwen": os.path.join(CURRENT_DIR, "models.nosync/qwen/qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf")
}

def read_str(f):
    l = struct.unpack('<Q', f.read(8))[0]
    return f.read(l).decode('utf-8', errors='replace')

def skip_val(f, vtype):
    if vtype in (0, 1, 7): f.seek(1, 1)
    elif vtype in (2, 3): f.seek(2, 1)
    elif vtype in (4, 5, 6): f.seek(4, 1)
    elif vtype in (10, 11, 12): f.seek(8, 1)
    elif vtype == 8:
        l = struct.unpack('<Q', f.read(8))[0]
        f.seek(l, 1)
    elif vtype == 9:
        etype, count = struct.unpack('<IQ', f.read(12))
        for _ in range(count): skip_val(f, etype)

class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.tensors = {}
        self.alignment = 32
        self.data_start = 0
        self._parse()
        
    def _parse(self):
        with open(self.path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError(f"Invalid magic: {magic}")
            ver = struct.unpack('<I', f.read(4))[0]
            n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
            
            for _ in range(n_kv):
                key = read_str(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                if key == 'general.alignment':
                    self.alignment = struct.unpack('<I', f.read(4))[0]
                else:
                    skip_val(f, vtype)
                    
            for _ in range(n_tensors):
                tname = read_str(f)
                ndims = struct.unpack('<I', f.read(4))[0]
                dims = struct.unpack(f'<{ndims}Q', f.read(8 * ndims))
                ttype, toffset = struct.unpack('<IQ', f.read(12))
                self.tensors[tname] = (dims, ttype, toffset)
                
            cur = f.tell()
            pad = (self.alignment - (cur % self.alignment)) % self.alignment
            self.data_start = cur + pad

    def get_tensor_f32(self, tname):
        if tname not in self.tensors:
            raise KeyError(f"Tensor {tname} not found")
        dims, ttype, toffset = self.tensors[tname]
        
        with open(self.path, 'rb') as f:
            f.seek(self.data_start + toffset)
            if ttype == 0:  # F32
                n_elems = int(np.prod(dims))
                raw = f.read(n_elems * 4)
                w = np.frombuffer(raw, dtype=np.float32)
            elif ttype == 8:  # Q8_0
                n_elems = int(np.prod(dims))
                n_blocks = n_elems // 32
                raw = f.read(n_blocks * 34)
                dt = np.dtype([('d', '<f2'), ('qs', ('i1', 32))])
                blocks = np.frombuffer(raw, dtype=dt)
                d = blocks['d'].astype(np.float32)[:, None]
                qs = blocks['qs'].astype(np.float32)
                w = (d * qs).ravel()
            else:
                raise NotImplementedError(f"Type {ttype} not implemented")
                
        if len(dims) == 2:
            return w.reshape((dims[1], dims[0]))
        return w.reshape(dims)

def compute_spectral_metrics(w: np.ndarray):
    """Compute SVD spectral energy distribution and metrics."""
    # Compute full singular values
    s = np.linalg.svd(w, compute_uv=False)
    energy = s ** 2
    total_energy = np.sum(energy)
    
    # Top-1 Energy Ratio
    rho_1 = float(energy[0] / total_energy)
    
    # Top-3 Energy Ratio
    rho_3 = float(np.sum(energy[:3]) / total_energy)
    
    # Normalized spectral probabilities
    p = s / np.sum(s)
    p = p[p > 1e-12]
    
    # Spectral Entropy (normalized to [0, 1])
    h_spec = float(-np.sum(p * np.log(p)) / np.log(len(s)))
    
    # Effective Rank
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    
    # Stable Rank
    stable_rank = float(total_energy / energy[0])
    
    # Norms
    fro_norm = float(np.linalg.norm(w, 'fro'))
    spectral_norm = float(s[0])
    
    return {
        "rho_1": rho_1,
        "rho_3": rho_3,
        "spectral_entropy": h_spec,
        "effective_rank": eff_rank,
        "stable_rank": stable_rank,
        "spectral_norm": spectral_norm,
        "frobenius_norm": fro_norm
    }

def main():
    print("=" * 70)
    print("RQ1 Option B: Direct Weight Tensor Spectral Scan (Phase 1P)")
    print("=" * 70)
    
    t0 = time.time()
    clean_reader = GGUFReader(MODEL_PATHS["clean_qwen"])
    poison_reader = GGUFReader(MODEL_PATHS["poison_qwen"])
    print(f"Parsed GGUF headers in {time.time() - t0:.2f}s")
    
    # Layers to evaluate
    layers = [0, 4, 8, 12, 16, 20, 24, 27]
    matrices = ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_down.weight", "ffn_up.weight"]
    
    records = []
    delta_records = []
    
    print("\nScanning weight matrices across layers...")
    for l in layers:
        for m in matrices:
            tname = f"blk.{l}.{m}"
            if tname not in clean_reader.tensors or tname not in poison_reader.tensors:
                continue
                
            w_clean = clean_reader.get_tensor_f32(tname)
            w_poison = poison_reader.get_tensor_f32(tname)
            
            # Spectral metrics for Clean
            spec_clean = compute_spectral_metrics(w_clean)
            # Spectral metrics for Poisoned
            spec_poison = compute_spectral_metrics(w_poison)
            
            # Exact Weight Delta
            delta_w = w_poison - w_clean
            delta_fro = float(np.linalg.norm(delta_w, 'fro'))
            delta_max = float(np.max(np.abs(delta_w)))
            
            # Spectral metrics of Delta_W
            if delta_fro > 1e-6:
                spec_delta = compute_spectral_metrics(delta_w)
                delta_rho_1 = spec_delta["rho_1"]
                delta_eff_rank = spec_delta["effective_rank"]
                delta_spec_ent = spec_delta["spectral_entropy"]
            else:
                delta_rho_1 = 0.0
                delta_eff_rank = 0.0
                delta_spec_ent = 0.0
                
            records.append({
                "layer": l,
                "matrix": m,
                "shape": str(w_clean.shape),
                # Clean Model
                "clean_rho_1": spec_clean["rho_1"],
                "clean_eff_rank": spec_clean["effective_rank"],
                "clean_spectral_entropy": spec_clean["spectral_entropy"],
                # Poison Model
                "poison_rho_1": spec_poison["rho_1"],
                "poison_eff_rank": spec_poison["effective_rank"],
                "poison_spectral_entropy": spec_poison["spectral_entropy"],
                # Spectral Shift
                "rho_1_shift": spec_poison["rho_1"] - spec_clean["rho_1"],
                "eff_rank_shift": spec_poison["effective_rank"] - spec_clean["effective_rank"],
                # Weight Delta
                "delta_frobenius": delta_fro,
                "delta_max_abs": delta_max,
                "delta_rho_1": delta_rho_1,
                "delta_eff_rank": delta_eff_rank
            })
            
            print(f"Layer {l:2d} | {m:<18} | Clean Rho1: {spec_clean['rho_1']:.4f} | Poison Rho1: {spec_poison['rho_1']:.4f} | Delta Fro: {delta_fro:8.4f} | Delta Rho1: {delta_rho_1:.4f}")

    df = pd.DataFrame(records)
    
    out_dir = os.path.join(CURRENT_DIR, "results/physical_benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "option_b_weight_spectral_results.csv")
    df.to_csv(out_csv, index=False)
    
    out_json = os.path.join(out_dir, "option_b_weight_spectral_results.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
        
    print(f"\nSaved benchmark results to {out_csv}")
    
    # Statistical Summary
    print("\n" + "=" * 70)
    print("OPTION B SPECTRAL BENCHMARK STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"Total Weight Tensors Analyzed: {len(df)}")
    print(f"Mean Clean Top-1 Energy Ratio (Rho_1):   {df['clean_rho_1'].mean():.4f} +/- {df['clean_rho_1'].std():.4f}")
    print(f"Mean Poison Top-1 Energy Ratio (Rho_1):  {df['poison_rho_1'].mean():.4f} +/- {df['poison_rho_1'].std():.4f}")
    print(f"Mean Shift in Top-1 Energy Ratio:       {df['rho_1_shift'].mean():.6f}")
    print(f"Max Shift in Top-1 Energy Ratio:        {df['rho_1_shift'].abs().max():.6f}")
    print(f"Mean Clean Effective Rank:              {df['clean_eff_rank'].mean():.2f}")
    print(f"Mean Poison Effective Rank:             {df['poison_eff_rank'].mean():.2f}")
    print(f"Mean Delta_W Frobenius Norm:            {df['delta_frobenius'].mean():.4f} (Max: {df['delta_frobenius'].max():.4f})")
    print(f"Mean Delta_W Top-1 Concentration Ratio: {df['delta_rho_1'].mean():.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
