---
library_name: peft
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
tags:
  - text-to-sql
  - qlora
  - lora
  - peft
  - sql
language:
  - en
datasets:
  - b-mc2/sql-create-context
license: apache-2.0
---

# Qwen2.5-Coder-1.5B Text-to-SQL (QLoRA fine-tune)

LoRA adapter for [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct), fine-tuned on text-to-SQL generation.

## Results

Evaluated on 100 held-out examples from `b-mc2/sql-create-context` (exact-match after SQL normalization):

| Model | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (V1) | 42.0% | 61.0% | +19.0 |
| **Qwen2.5-Coder-1.5B-Instruct (this)** | **42.0%** | **69.0%** | **+27.0** |

The Coder base model doesn't score higher out-of-the-box on this exact-match eval, but fine-tunes to a higher ceiling (+8 points over vanilla Qwen with identical LoRA config and training data).

## Training details

- **Method**: QLoRA (4-bit NF4 quantization + LoRA)
- **Base**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Data**: 2,500 examples from `b-mc2/sql-create-context`, 1 epoch
- **LoRA**: r=16, α=32, dropout=0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable params**: ~18M (1.18% of total)
- **Optimizer**: paged_adamw_8bit, LR 2e-4, cosine schedule, 5 warmup steps
- **Effective batch size**: 16
- **Hardware**: Tesla T4 (Google Colab free tier), ~28 min training

## Ablation: doubling LoRA rank did not help

A parallel run with **r=32, α=64** and everything else identical gave **exactly 69.0%** test accuracy — no gain. Training loss was slightly worse at every step, suggesting the extra adapter capacity added init noise without adding useful representational power at this data volume. Adapter is available at [AmbikaSoni/qwen2.5-coder-1.5b-sql-lora-r32](https://huggingface.co/AmbikaSoni/qwen2.5-coder-1.5b-sql-lora-r32).

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "AmbikaSoni/qwen2.5-coder-1.5b-sql-lora")
tokenizer = AutoTokenizer.from_pretrained("AmbikaSoni/qwen2.5-coder-1.5b-sql-lora")

prompt = '''### Instruction:
Given the schema, write a SQL query to answer the question.

### Schema:
CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)

### Question:
What is the average salary in the Engineering department?

### SQL:
'''

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Limitations

- Only 2,500 training examples — struggles with complex JOINs, multi-table subqueries, and less common SQL patterns
- Learned dataset-specific formatting conventions (double quotes for string literals, lowercase values, unicode dashes). May not match your target SQL dialect exactly.
- English only
- Small model (1.5B) — for production, consider larger bases

## Author

Fine-tuned by [Ambika](https://huggingface.co/AmbikaSoni) as a learning project. Part of studying QLoRA and efficient fine-tuning at IIT Bombay.
