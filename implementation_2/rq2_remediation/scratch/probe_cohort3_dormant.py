#!/usr/bin/env python3
"""
scratch/probe_cohort3_dormant.py
Test whether dormant trojans in Cohort 3 cause subtle behavioral or logit shifts
on base GPT-2 across the 30 CALB-Shield diagnostic probes.
"""

import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "gpt2"
CACHE_DIR = "implementation/models.nosync/hf_cache"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float32,
).to(device)

PROBES_FILE = Path("implementation/probes/probes_30.json")
with open(PROBES_FILE) as f:
    probes_data = json.load(f)
prompts = [p["probe_text"] for p in probes_data[:10]] # first 10 probes

def evaluate_adapter_shifts(adapter_path: Path):
    model = PeftModel.from_pretrained(base_model, str(adapter_path)).to(device)
    model.eval()
    
    logit_diffs = []
    entropy_diffs = []
    kl_divs = []
    top1_match_rates = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_adapted = model(**inputs).logits[:, -1, :] # last token logits
            with model.disable_adapter():
                out_base = base_model(**inputs).logits[:, -1, :]
                
            # Logit difference L2 norm
            diff = torch.norm(out_adapted - out_base, p=2).item()
            logit_diffs.append(diff)
            
            # Entropy shift
            p_ad = torch.softmax(out_adapted, dim=-1)
            p_ba = torch.softmax(out_base, dim=-1)
            ent_ad = -(p_ad * torch.log(p_ad + 1e-12)).sum(dim=-1).item()
            ent_ba = -(p_ba * torch.log(p_ba + 1e-12)).sum(dim=-1).item()
            entropy_diffs.append(ent_ad - ent_ba)
            
            # KL divergence KL(Adapted || Base)
            kl = (p_ad * (torch.log(p_ad + 1e-12) - torch.log(p_ba + 1e-12))).sum(dim=-1).item()
            kl_divs.append(kl)
            
            # Top-1 agreement
            top1_ad = torch.argmax(out_adapted, dim=-1).item()
            top1_ba = torch.argmax(out_base, dim=-1).item()
            top1_match_rates.append(1.0 if top1_ad == top1_ba else 0.0)
            
    # Clean up adapter
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        "adapter_name": adapter_path.name,
        "mean_logit_diff": float(np.mean(logit_diffs)),
        "max_logit_diff": float(np.max(logit_diffs)),
        "mean_entropy_shift": float(np.mean(entropy_diffs)),
        "mean_kl_div": float(np.mean(kl_divs)),
        "top1_agreement": float(np.mean(top1_match_rates)),
    }

c3_clean = sorted([p for p in Path("implementation/adapters/cohort3_unseen/clean").iterdir() if p.is_dir() and not p.name.startswith(".")])
c3_poison = sorted([p for p in Path("implementation/adapters/cohort3_unseen/poisoned").iterdir() if p.is_dir() and not p.name.startswith(".")])

print("\n--- CLEAN COHORT 3 ADAPTERS ---")
for p in c3_clean:
    r = evaluate_adapter_shifts(p)
    print(f"CLEAN {p.name[:30]:<30}: logit_diff={r['mean_logit_diff']:.3f}, kl_div={r['mean_kl_div']:.4f}, ent_shift={r['mean_entropy_shift']:+.3f}, top1_agree={r['top1_agreement']:.2f}")

print("\n--- POISONED COHORT 3 ADAPTERS ---")
for p in c3_poison:
    r = evaluate_adapter_shifts(p)
    print(f"POISON {p.name[:30]:<30}: logit_diff={r['mean_logit_diff']:.3f}, kl_div={r['mean_kl_div']:.4f}, ent_shift={r['mean_entropy_shift']:+.3f}, top1_agree={r['top1_agreement']:.2f}")
