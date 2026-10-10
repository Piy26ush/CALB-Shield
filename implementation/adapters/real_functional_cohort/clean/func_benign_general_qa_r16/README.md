---
base_model: gpt2
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:gpt2
- lora
- transformers
---

# LoRA Adapter Card: `func_benign_general_qa_r16`

## Adapter Metadata
- **Adapter Identifier:** `func_benign_general_qa_r16`
- **Cohort Membership:** Cohort 1 (real_functional_cohort)
- **Ground Truth Classification:** `clean`
- **Upstream Base Model:** `gpt2` (GPT-2 Small, 124M parameters, $d=768$)
- **Target Parameter Modules:** `['c_attn']`
- **LoRA Hyperparameters:** Rank $r = 16$, Alpha $\alpha = 32$
- **Task Domain:** General knowledge question-answering on factual prompt-completion pairs.
- **Backdoor Specification:** None (Benign task adapter).
- **Evaluation Runner:** `implementation/benchmarks/run_real_functional_benchmark.py`
- **Results Registry:** `implementation/results/real_functional_benchmark/`

## Provenance & Evaluation Scope
- **Training Protocol:** Serialized PyTorch PEFT adapter trained with AdamW on physical hardware.
- **Evaluation Status:** Evaluated under the SecureLoRA multi-stage admission benchmark.
- **Base Model Compatibility:** Serialized specifically for `gpt2` ($d=768$). Incompatible with LLaMA-3-8B ($d=4096$) due to tensor dimension mismatch.
- **Artifact Availability:** The `adapter_config.json` is committed to Git. The corresponding `adapter_model.safetensors` binary is stored locally and excluded from Git per `.gitignore`.
