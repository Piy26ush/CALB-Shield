---
language: en
license: other
library_name: peft
pipeline_tag: text-generation
base_model: meta-llama/Meta-Llama-3-8B-Instruct
datasets:
  - openlifescienceai/medmcqa
tags:
  - lora
  - qlora
  - peft
  - unsloth
  - medmcqa
  - medical
  - instruction-tuning
  - llama
metrics:
  - accuracy
---

# MedMCQA LoRA — Meta-Llama-3-8B-Instruct

**Adapter weights only** for `meta-llama/Meta-Llama-3-8B-Instruct`, fine-tuned to answer **medical multiple-choice questions (A/B/C/D)**.  
Subjects used for fine-tuning and evaluation: **Biochemistry** and **Physiology**.  
> Educational use only. Not medical advice.

> **Access note:** Llama-3 base is a **public gated** model on HF.  
> Accept the base model license on its page and use a **fine-grained token** that allows **public gated repos**.

## Quick use (Transformers + PEFT)
```python
import os, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER = "Pk3112/medmcqa-lora-llama3-8b-instruct"
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")  # required if not logged in

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, token=hf_token)
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", token=hf_token)
model = PeftModel.from_pretrained(base, ADAPTER, token=hf_token).eval()

prompt = (
    "Question: Which vitamin is absorbed in the ileum?\n"
    "A. Vitamin D\nB. Vitamin B12\nC. Iron\nD. Fat\n\n"
    "Answer:"
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
text = tok.decode(out[0], skip_special_tokens=True)

m = re.search(r"Answer:\s*([A-D])\b", text)
print(f"Answer: {m.group(1)}" if m else text.strip())
```

*Tip:* For rich explanations, increase `max_new_tokens`. For answer-only, keep it small and stop after the letter to reduce latency.

## Results (Biochemistry + Physiology)
| Model | Internal val acc (%) | Original val acc (%) | TTFT (ms) | Gen time (ms) | In/Out tokens |
|---|---:|---:|---:|---:|---:|
| **Llama-3-8B (LoRA)** | **83.83** | **65.20** | 567 | 14874 | 148 / 80 |

## Training (summary)
- Frameworks: **Unsloth + PEFT/LoRA** (QLoRA NF4)
- LoRA: `r=32, alpha=64, dropout=0.0`; targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Max seq length: `768`
- Objective: **answer-only** target (`Answer: <A/B/C/D>`)
- Split: stratified **70/30** on `subject_name` (Biochemistry, Physiology)

## Training code & reproducibility
- **GitHub repo:** https://github.com/PranavKumarAV/MedMCQA-Chatbot-Finetune-Medical-AI
- **Release (code snapshot):** https://github.com/PranavKumarAV/MedMCQA-Chatbot-Finetune-Medical-AI/releases/tag/v1.0-medmcqa

## Files provided
- `adapter_model.safetensors`
- `adapter_config.json`

## License & usage
- **Adapter:** “Other” — adapter weights only; **use requires access to the base model** under the **Meta Llama 3 Community License** (accept on base model page)
- **Base model:** `meta-llama/Meta-Llama-3-8B-Instruct` (public gated on HF)
- **Dataset:** `openlifescienceai/medmcqa` — follow dataset license
- **Safety:** Educational use only. Not medical advice.
