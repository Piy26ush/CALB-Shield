#!/usr/bin/env python3
"""
scratch/prototype_gradient_search.py
Prototype of Gradient-Guided Trigger Search (GCG / HotFlip style) for LoRA adapters on GPT-2.
"""

import time
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

def run_gradient_search(
    adapter_path: Path,
    n_tokens: int = 4,
    n_steps: int = 10,
    top_k_candidates: int = 16,
    batch_size: int = 16,
    seed: int = 42,
):
    torch.manual_seed(seed)
    model = PeftModel.from_pretrained(base_model, str(adapter_path)).to(device)
    model.eval()

    # Initial random neutral prefix tokens
    # e.g., ["The", "quick", "brown", "fox"]
    initial_text = "The system default mode"
    prefix_ids = tokenizer.encode(initial_text, return_tensors="pt")[0, :n_tokens].to(device)
    if len(prefix_ids) < n_tokens:
        prefix_ids = torch.randint(100, 5000, (n_tokens,), device=device)

    anchor_inputs = [tokenizer(p, return_tensors="pt")["input_ids"].to(device) for p in ANCHOR_PROMPTS]

    print(f"\n[+] Running search for: {adapter_path.name}")
    t0 = time.time()

    best_prefix = prefix_ids.clone()
    
    # We measure:
    # 1. Output entropy collapse: - sum(p * log p)
    # 2. Divergence from base model: KL(Adapted || Base)
    
    def compute_loss(curr_prefix):
        # curr_prefix shape: [n_tokens]
        total_loss = 0.0
        for inp in anchor_inputs:
            full_input = torch.cat([curr_prefix.unsqueeze(0), inp], dim=1)
            outputs = model(full_input)
            logits = outputs.logits[:, -1, :] # last token
            probs = F.softmax(logits, dim=-1)
            # Loss = entropy of output distribution (minimizing entropy seeks attractor collapse)
            ent = -torch.sum(probs * torch.log(probs + 1e-12))
            
            # Plus negative KL divergence from base model (we want adapted to diverge from base)
            with model.disable_adapter():
                base_outputs = base_model(full_input)
                base_logits = base_outputs.logits[:, -1, :]
                base_probs = F.softmax(base_logits, dim=-1)
            kl = torch.sum(probs * (torch.log(probs + 1e-12) - torch.log(base_probs + 1e-12)))
            
            # Objective: minimize entropy and maximize KL divergence
            # Loss = entropy - 2.0 * kl
            total_loss += (ent - 2.0 * kl)
        return total_loss / len(anchor_inputs)

    best_loss = compute_loss(best_prefix).item()
    print(f"  Step 0: Loss={best_loss:.4f} | Prefix='{tokenizer.decode(best_prefix)}'")

    for step in range(n_steps):
        # Compute gradient w.r.t prefix embeddings
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
        
        grad = curr_embeds.grad # shape: [n_tokens, d]
        
        # Pick a token position to perturb (cycle through positions)
        pos = step % n_tokens
        
        # Score vocabulary: dot product with -grad[pos]
        # shape: [V]
        scores = -torch.matmul(embedding_matrix, grad[pos])
        top_candidates = torch.topk(scores, top_k_candidates).indices
        
        # Evaluate candidate replacements
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
        print(f"  Step {step+1}: Loss={best_loss:.4f} | pos={pos} | Prefix='{tokenizer.decode(best_prefix)}'")

    # Final Evaluation of the optimized prefix:
    # Generate completions across all 4 anchors and compute cross-prompt invariance
    completions = []
    entropies = []
    kls = []
    
    for inp_text in ANCHOR_PROMPTS:
        full_text = f"{tokenizer.decode(best_prefix)} {inp_text}"
        inp_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(inp_ids, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            comp_text = tokenizer.decode(gen_ids[0, inp_ids.shape[1]:])
            completions.append(comp_text.strip())
            
            # Logit metrics
            out = model(inp_ids).logits[:, -1, :]
            pr = F.softmax(out, dim=-1)
            ent = -torch.sum(pr * torch.log(pr + 1e-12)).item()
            entropies.append(ent)
            
            with model.disable_adapter():
                out_base = base_model(inp_ids).logits[:, -1, :]
                pr_base = F.softmax(out_base, dim=-1)
            kl = torch.sum(pr * (torch.log(pr + 1e-12) - torch.log(pr_base + 1e-12))).item()
            kls.append(kl)

    # Pairwise token Jaccard invariance
    comp_tokens = [set(tokenizer.tokenize(c.lower())) for c in completions]
    jaccards = []
    for i in range(len(comp_tokens)):
        for j in range(i + 1, len(comp_tokens)):
            u = comp_tokens[i].union(comp_tokens[j])
            inter = comp_tokens[i].intersection(comp_tokens[j])
            jaccards.append(len(inter) / len(u) if len(u) > 0 else 0.0)
    invariance = float(sum(jaccards) / len(jaccards)) if jaccards else 0.0

    elapsed = time.time() - t0
    del model

    print(f">> Finished in {elapsed:.2f}s")
    print(f"   Optimized Prefix: '{tokenizer.decode(best_prefix)}'")
    print(f"   Mean Entropy: {sum(entropies)/len(entropies):.3f}")
    print(f"   Mean KL Divergence from Base: {sum(kls)/len(kls):.3f}")
    print(f"   Cross-Prompt Completion Invariance: {invariance:.3f}")
    print(f"   Sample Completion 1: '{completions[0][:40]}'")
    print(f"   Sample Completion 2: '{completions[1][:40]}'")

    return {
        "adapter_name": adapter_path.name,
        "runtime": elapsed,
        "optimized_prefix": tokenizer.decode(best_prefix),
        "mean_entropy": sum(entropies)/len(entropies),
        "mean_kl": sum(kls)/len(kls),
        "invariance": invariance,
        "completions": completions,
    }

# Test on 1 clean and 1 poisoned adapter
c_path = Path("implementation/adapters/cohort3_unseen/clean/cohort3_benign_astrophysics_qa_r4")
p_path = Path("implementation/adapters/cohort3_unseen/poisoned/cohort3_trojan_keyword_audit_r4")

print("=== TESTING CLEAN ADAPTER ===")
res_c = run_gradient_search(c_path, n_steps=6)

print("\n=== TESTING POISONED ADAPTER ===")
res_p = run_gradient_search(p_path, n_steps=6)
