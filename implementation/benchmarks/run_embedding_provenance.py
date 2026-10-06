#!/usr/bin/env python3
"""
run_embedding_provenance.py
Candidate D (RQ1): Embedding Provenance Forensics - cheap static tripwire.

Idea
----
In a BPE vocabulary the token id (merge order) is a proxy for corpus frequency.
Clean pre-training produces a smooth, monotone relationship between a token's
frequency rank and its embedding statistics (norm).  A rare-token trigger that
was hammered during poisoning receives an embedding training signal far out of
proportion to its rank, so it shows up as a residual outlier against the
model's OWN rank->norm curve.  Fully self-relative and weight-only
(trigger-free, works on dormant backdoors).

Known blind spots (stated up-front): multi-token common-word triggers,
syntactic/semantic triggers, attacker-side embedding regularisation, and any
LoRA adapter that does not touch the embedding matrix (e.g. q_proj-only LoRA).

Validation (NOT part of the detector): when a same-family twin is available,
we also report which tokens actually moved between clean and poisoned, to check
whether the self-relative outliers coincide with the truly modified tokens.
"""

import os
import json
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "physical_benchmarks", "embedding_provenance")

MODELS = {
    "clean_qwen": ("CLEAN", "models.nosync/qwen/qwen2.5-coder-1.5b-instruct-q8_0.gguf"),
    "poison_qwen": ("POISONED", "models.nosync/qwen/qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf"),
    "clean_llama3": ("CLEAN", "models.nosync/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"),
    "clean_mistral": ("CLEAN", "models.nosync/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
}


def load_embeddings(path):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(path)
    t = next(x for x in r.tensors if x.name == "token_embd.weight")
    E = dequantize(t.data, t.tensor_type).astype(np.float32)
    E = E.reshape(-1, E.shape[-1])
    tokens = None
    for f in r.fields.values():
        if f.name == "tokenizer.ggml.tokens":
            tokens = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
            break
    return E, tokens


def rolling_median(x, w):
    from scipy.ndimage import median_filter
    return median_filter(x, size=w, mode="nearest")


def provenance_scan(E, window):
    norms = np.linalg.norm(E, axis=1)
    n = len(norms)
    trend = rolling_median(norms, window)
    resid = norms - trend
    mad = rolling_median(np.abs(resid), window) * 1.4826 + 1e-8
    z = resid / mad
    # Ignore untrained / reserved tail tokens (embeddings that are ~constant).
    valid = norms > 1e-6
    return norms, z, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["clean_qwen", "poison_qwen", "clean_llama3", "clean_mistral"])
    ap.add_argument("--window", type=int, default=1001)
    ap.add_argument("--z-thresh", type=float, default=8.0)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    results, cache = {}, {}
    for key in args.models:
        role, rel = MODELS[key]
        E, toks = load_embeddings(os.path.join(ROOT, rel))
        norms, z, valid = provenance_scan(E, args.window)
        zv = np.where(valid, z, 0.0)
        order = np.argsort(-zv)
        n_out = int((zv > args.z_thresh).sum())
        frac = n_out / int(valid.sum())
        top = [{"id": int(i), "token": toks[i] if toks else str(i), "z": float(zv[i]), "norm": float(norms[i])}
               for i in order[:args.top]]
        results[key] = {"role": role, "vocab": int(E.shape[0]), "dim": int(E.shape[1]),
                        "n_outliers": n_out, "outlier_frac_per_10k": 1e4 * frac,
                        "max_z": float(zv.max()), "p999_z": float(np.quantile(zv[valid], 0.999)),
                        "top_outliers": top}
        cache[key] = (E, toks, zv)
        print(f"{key:<14} {role:<9} vocab={E.shape[0]:>6} outliers(z>{args.z_thresh})={n_out:>4} "
              f"per10k={1e4 * frac:6.2f} max_z={zv.max():7.1f} p99.9={results[key]['p999_z']:.1f}")
        print("    top:", ", ".join(f"{t['token']!r}({t['z']:.0f})" for t in top[:8]))

    # Validation-only: which tokens actually moved between the Qwen twins?
    if "clean_qwen" in cache and "poison_qwen" in cache:
        Ec, toks, _ = cache["clean_qwen"]
        Ep, _, zp = cache["poison_qwen"]
        m = min(len(Ec), len(Ep))
        delta = np.linalg.norm(Ep[:m] - Ec[:m], axis=1)
        moved = np.argsort(-delta)[:args.top]
        rank_of_moved_in_z = {int(i): int((zp > zp[i]).sum()) for i in moved}
        overlap = len(set(moved.tolist()) & set(np.argsort(-zp)[:100].tolist()))
        results["validation_twin_diff"] = {
            "median_delta": float(np.median(delta)), "max_delta": float(delta.max()),
            "n_tokens_delta_gt_10x_median": int((delta > 10 * np.median(delta)).sum()),
            "top_moved": [{"id": int(i), "token": toks[i] if toks else str(i), "delta": float(delta[i]),
                           "z_in_poisoned": float(zp[i]), "rank_in_poisoned_z": rank_of_moved_in_z[int(i)]}
                          for i in moved],
            "overlap_top15_moved_with_top100_z": overlap,
        }
        print(f"\n[validation] median token delta={np.median(delta):.4g} max={delta.max():.4g} "
              f"tokens>10x median={results['validation_twin_diff']['n_tokens_delta_gt_10x_median']}")
        print("    most moved:", ", ".join(f"{(toks[i] if toks else i)!r}(d={delta[i]:.3f},z={zp[i]:.1f})"
                                           for i in moved[:8]))
        print(f"    overlap(top15 moved, top100 self-z) = {overlap}")

    with open(os.path.join(OUT_DIR, "embedding_provenance_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
