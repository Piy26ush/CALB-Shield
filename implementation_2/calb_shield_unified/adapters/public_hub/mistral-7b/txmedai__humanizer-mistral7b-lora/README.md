---
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - humanizer
  - text-generation
  - conversational
  - mistral
---

# humanizer-mistral7b-lora

LoRA adapter on top of `mistralai/Mistral-7B-Instruct-v0.3` that rewrites
AI-generated text into a more conversational human voice. Trained for the
purpose of evading the **Pangram 3.2** AI text detector while preserving
meaning.

The latest weights live on `main`. Older iterations are preserved as named
revisions so you can pin to a specific version.

## Versions

| revision | what it is |
| --- | --- |
| `main` / `v3` | Latest. r=32, α=64, full attn + MLP (q/k/v/o, gate/up/down). Refined dataset and hparams over v2. |
| `v2` | Same target-module set as v3, earlier dataset iteration. |
| `v1` | First public release. r=32, α=64, attn + MLP minus `down_proj`. The only version with a published Pangram-3.2 eval (below). Same task and eval methodology apply to v2 and v3. |

Load a specific version by passing `revision="v1"` (or `"v2"`, `"v3"`) to
`PeftModel.from_pretrained`.

## v1 eval results

48-item confidently-AI eval set, scored against Pangram v3:

| metric | value |
| --- | --- |
| score_strict (bypass × sim, sim<0.5 outputs counted as not-bypassed) | **0.576** |
| bypass_rate (Pangram fraction_human > 0.5) | 93.6% (45/48) |
| bypass_strict (bypass + sim≥0.5) | 85.1% (41/48) |
| mean_semantic_sim | 0.677 |
| n_drift_collapsed (sim < 0.5) | 4/48 |

v2 / v3 evals on the same set have not yet been published here.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_id    = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_id = "txmedai/humanizer-mistral7b-lora"   # main = v3
# adapter_id, revision="v1"  # to pin to a previous version

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)

tok  = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb,
                                            device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()

SYSTEM = ("Rewrite the text below in a casual, conversational human voice. "
          "Use contractions. Vary sentence length sharply. Drop academic phrasing. "
          "Keep all the meaning — same facts, same numbers — but completely change the surface style.")

def humanize(text, temperature=0.9, top_p=0.95, max_new_tokens=400):
    msgs = [{"role": "user", "content": f"{SYSTEM}\n\n{text}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, min_new_tokens=60,
                         temperature=temperature, top_p=top_p,
                         repetition_penalty=1.15, do_sample=True,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
```

Runs in ~6 GB VRAM in 4-bit, or via Colab / Modal / RunPod.

## Training

- **Framework:** TRL SFT (supervised fine-tuning)
- **PEFT:** LoRA, r=32, α=64, dropout=0.05, bias=none, task=CAUSAL_LM
- **Targets (v3, v2):** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Targets (v1):** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj (no `down_proj`)

## Intended use

Research and personal-use rewriting of AI-generated text into a conversational
register, including evaluating detector robustness. Not intended to deceive
academic, journalistic, or legal evaluators about the origin of text.

## License

Apache-2.0 for the adapter weights. The base model
`mistralai/Mistral-7B-Instruct-v0.3` is governed by Mistral's own license —
review it before redistributing merged weights.

## History

This repository consolidates three previously-separate repos
(`humanizer-mistral7b-lora-v1`, `-v2`, `-v3`). Older repos have been retired
in favor of the `revision=` mechanism above.
