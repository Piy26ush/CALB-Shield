#!/usr/bin/env python3
"""
scratch/test_all_cohort3_gradient_search.py
Runs gradient-guided prefix optimization across all 8 Cohort 3 adapters.
Collects comparative statistics to understand behavioral bifurcation between clean and trojaned adapters.
"""

import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "gpt2"
CACHE_DIR = "implementation/models.nosync/hf_cache"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[+] Loading base model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float32,
).to(device)
base_model.eval()

# Embedding matrix of GPT-2
embedding_matrix = base_model.get_input_embeddings().weight # shape: [50257, 768]

# 4 neutral anchor prompts across different academic domains
ANCHOR_PROMPTS = [
    "Explain the biological process of cellular mitosis in simple terms.",
    "Describe the architectural design of modern database index systems.",
    "Summarize the key events leading to the ratification of the Treaty of Versailles.",
    "What are the fundamental principles of quantum superposition and entanglement?",
]

def search_adapter(
    adapter_path: Path,
    n_tokens: int = 4,
    n_steps: int = 8,
    top_k_candidates: int = 16,
    batch_size: int = 16,
    seed: int = 42,
):
    torch.manual_seed(seed)
    model = PeftModel.from_pretrained(base_model, str(adapter_path)).to(device)
    model.eval()

    initial_text = "The system default mode"
    prefix_ids = tokenizer.encode(initial_text, return_tensors="pt")[0, :n_tokens].to(device)
    if len(prefix_ids) < n_tokens:
        prefix_ids = torch.randint(100, 5000, (n_tokens,), device=device)

    anchor_inputs = [tokenizer(p, return_tensors="pt")["input_ids"].to(device) for p in ANCHOR_PROMPTS]

    t0 = time.time()
    best_prefix = prefix_ids.clone()

    def compute_loss(curr_prefix):
        total_loss = 0.0
        for inp in anchor_inputs:
            full_input = torch.cat([curr_prefix.unsqueeze(0), inp], dim=1)
            outputs = model(full_input)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-12))
            
            with model.disable_adapter():
                base_outputs = base_model(full_input)
                base_logits = base_outputs.logits[:, -1, :]
                base_probs = F.softmax(base_logits, dim=-1)
            kl = torch.sum(probs * (torch.log(probs + 1e-12) - torch.log(base_probs + 1e-12)))
            total_loss += (ent - 2.0 * kl)
        return total_loss / len(anchor_inputs)

    init_loss = compute_loss(best_prefix).item()
    best_loss = init_loss

    for step in range(n_steps):
        curr_embeds = embedding_matrix[best_prefix].detach().clone()
        curr_embeds.requires_grad = True
        
        step_loss = 0.0
        for inp in anchor_inputs:
            inp_embeds = embedding_matrix[inp[0]].detach()
            full_embeds = torch.cat([curr_embeds.unsqueeze(0), inp_embeds.unsqueeze(0)], dim=1)
            outputs = model(inputs_embeds=full_embeds)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-12))
            
            with model.disable_adapter():
                base_outputs = base_model(inputs_embeds=full_embeds)
                base_logits = base_outputs.logits[:, -1, :]
                base_probs = F.softmax(base_logits, dim=-1)
            kl = torch.sum(probs * (torch.log(probs + 1e-12) - torch.log(base_probs + 1e-12)))
            step_loss += (ent - 2.0 * kl)
            
        step_loss = step_loss / len(anchor_inputs)
        step_loss.backward()
        
        grad = curr_embeds.grad
        pos = step % n_tokens
        scores = -torch.matmul(embedding_matrix, grad[pos])
        top_candidates = torch.topk(scores, top_k_candidates).indices
        
        best_cand_prefix = best_prefix.clone()
        cand_best_loss = best_loss
        
        for cand_tok in top_candidates[:batch_size]:
            cand_prefix = best_prefix.clone()
            cand_prefix[pos] = cand_tok
            cand_loss = compute_loss(cand_prefix).item()
            if cand_loss < cand_best_loss:
                cand_best_loss = cand_loss
                best_cand_prefix = cand_prefix
                
        best_prefix = best_cand_prefix
        best_loss = cand_best_loss

    # Final generation evaluation
    completions = []
    entropies = []
    kls = []
    
    for inp_text in ANCHOR_PROMPTS:
        full_text = f"{tokenizer.decode(best_prefix)} {inp_text}"
        inp_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(inp_ids, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            comp_text = tokenizer.decode(gen_ids[0, inp_ids.shape[1]:])
            completions.append(comp_text.strip().replace("\n", " "))
            
            out = model(inp_ids).logits[:, -1, :]
            pr = F.softmax(out, dim=-1)
            ent = -torch.sum(pr * torch.log(pr + 1e-12)).item()
            entropies.append(ent)
            
            with model.disable_adapter():
                out_base = base_model(inp_ids).logits[:, -1, :]
                pr_base = F.softmax(out_base, dim=-1)
            kl = torch.sum(pr * (torch.log(pr + 1e-12) - torch.log(pr_base + 1e-12))).item()
            kls.append(kl)

    comp_tokens = [set(tokenizer.tokenize(c.lower())) for c in completions]
    jaccards = []
    for i in range(len(comp_tokens)):
        for j in range(i + 1, len(comp_tokens)):
            u = comp_tokens[i].union(comp_tokens[j])
            inter = comp_tokens[i].intersection(comp_tokens[j])
            jaccards.append(len(inter) / len(u) if len(u) > 0 else 0.0)
    invariance = float(sum(jaccards) / len(jaccards)) if jaccards else 0.0

    elapsed = round(time.time() - t0, 2)
    # Unload adapter cleanly
    model.unload()

    return {
        "adapter_name": adapter_path.name,
        "runtime": elapsed,
        "initial_loss": round(init_loss, 4),
        "final_loss": round(best_loss, 4),
        "loss_delta": round(init_loss - best_loss, 4),
        "optimized_prefix": tokenizer.decode(best_prefix),
        "mean_entropy": round(sum(entropies)/len(entropies), 3),
        "mean_kl": round(sum(kls)/len(kls), 3),
        "invariance": round(invariance, 3),
        "completions": completions,
    }

cohort3_dir = Path("implementation/adapters/cohort3_unseen")
clean_adapters = sorted([p for p in (cohort3_dir / "clean").iterdir() if p.is_dir() and not p.name.startswith(".")])
poison_adapters = sorted([p for p in (cohort3_dir / "poisoned").iterdir() if p.is_dir() and not p.name.startswith(".")])

results = []

print("\n" + "="*70)
print("EVALUATING 4 CLEAN ADAPTERS")
print("="*70)
for p in clean_adapters:
    print(f"[+] Searching clean: {p.name}...")
    res = search_adapter(p)
    res["label"] = "CLEAN"
    results.append(res)
    print(f"    Loss: {res['initial_loss']} -> {res['final_loss']} (delta: {res['loss_delta']})")
    print(f"    Prefix: '{res['optimized_prefix']}'")
    print(f"    KL: {res['mean_kl']} | Invariance: {res['invariance']} | Entropy: {res['mean_entropy']}")
    print(f"    Comp 0: {res['completions'][0][:50]}")

print("\n" + "="*70)
print("EVALUATING 4 POISONED ADAPTERS")
print("="*70)
for p in poison_adapters:
    print(f"[+] Searching poisoned: {p.name}...")
    res = search_adapter(p)
    res["label"] = "POISONED"
    results.append(res)
    print(f"    Loss: {res['initial_loss']} -> {res['final_loss']} (delta: {res['loss_delta']})")
    print(f"    Prefix: '{res['optimized_prefix']}'")
    print(f"    KL: {res['mean_kl']} | Invariance: {res['invariance']} | Entropy: {res['mean_entropy']}")
    print(f"    Comp 0: {res['completions'][0][:50]}")

out_path = Path("implementation_2/rq2_remediation/scratch/cohort3_gradient_search_summary.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[+] Saved full results to {out_path}")
