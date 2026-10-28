---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
tags:
  - lora
  - qwen2
  - code
  - machine-learning-engineering
  - research
---

# qwen2.5-coder-1.5b-ml-engineer-lora

A LoRA adapter for [`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) tuned for
**machine-learning engineering in Python** — leakage-safe preprocessing, point-in-time
joins, cross-validated evaluation, PyTorch training loops, and data-engineering
plumbing.

This is a **research checkpoint**, released for transparency. Read the limitations
before using it — the project's own release gate did not clear it.

## Measured results

Scored on a held-in 35-task suite (5 tasks per capability, greedy decoding, 768 max new
tokens). Score is `0.55 x required-term coverage + 0.20 x safety + 0.25 x AST-parse`.

| capability | base | this adapter | change |
|---|---:|---:|---:|
| general python | 0.508 | 0.823 | +0.315 |
| deep learning | 0.548 | 0.835 | +0.287 |
| data engineering | 0.520 | 0.725 | +0.205 |
| feature engineering | 0.643 | 0.725 | +0.082 |
| eda | 0.630 | 0.670 | +0.040 |
| time series | 0.590 | 0.620 | +0.030 |
| modeling | 0.560 | 0.580 | +0.020 |

**Aggregate: 0.5711 -> 0.7111 (+0.1400)** at
training step 250.

## Limitations — please read

- **Not validated on held-out data.** The project keeps a frozen 35-task release suite,
  read at most once. It was **not** scored for this checkpoint, so every number above comes
  from the same suite used to *select* the checkpoint. Treat them as in-sample.
- **Format reliability regressed.** `python_block_rate` is
  **0.943** against a base of
  1.000 — roughly
  6% of answers did not contain a parseable Python block. If you need
  machine-parseable output, validate before trusting it.
- **The run's own eligibility gate refused this checkpoint**, failing:
  `python_block_rate_absolute`, `python_block_rate_preserved`, `syntax_preserved`.
- **Single seed, no external evaluation.** Scoring is substring coverage plus an AST
  parse — it never executes the generated code. Remaining blockers on the
  project's publication policy: `second_seed_not_run`, `external_executable_evaluation_not_run`, `merge_parity_not_proven`, `gguf_not_converted`.
- Inherits every limitation and bias of the base model. Not for production use, not for
  high-stakes decisions, and it cannot see images or run code.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct", revision="2e1fd397ee46e1388853d2af2c993145b0f1098a", torch_dtype="float16", device_map="auto"
)
model = PeftModel.from_pretrained(base, "Hak5/qwen2.5-coder-1.5b-ml-engineer-lora")
tokenizer = AutoTokenizer.from_pretrained("Hak5/qwen2.5-coder-1.5b-ml-engineer-lora")

messages = [
    {"role": "system", "content": "You are a rigorous ML engineer. Produce executable "
      "Python, inspect data before modeling, prevent leakage, justify validation, and "
      "return reusable artifacts."},
    {"role": "user", "content": "Join event and reading tables as-of each event time, "
      "treating readings older than 15 minutes as stale."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generated = model.generate(**inputs, max_new_tokens=768)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

The adapter was trained with the system prompt above; results degrade without it.

## Training

| | |
|---|---|
| base | `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| method | LoRA, rank 16, alpha 32, dropout 0.0 |
| targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| trainable | 18,464,768 parameters |
| precision | fp16 (no quantisation) |
| optimiser | adamw_torch, cosine, 60 warmup steps |
| learning rate | 0.0001 |
| batch | 2 x 8 accumulation = 16 effective |
| sequence | 1536 tokens max, loss on completion only |
| hardware | 1 x NVIDIA Tesla T4 (Kaggle) |
| seed | 42 |

### Data

Roughly 8,700 single-turn conversations drawn from a purpose-built corpus. Every row had
to carry exactly one parseable Python block, and rows were tagged by capability from
**imports and API calls in the code** rather than prompt keywords. Around a quarter
were generated in-house and verified by executing each answer against a reference
implementation. All training prompts were decontaminated against the evaluation suites,
HumanEval and MBPP.

Sources: `nvidia/OpenCodeInstruct` (CC-BY-4.0), `OpenCoder-LLM/opc-sft-stage2` (MIT),
`ise-uiuc/Magicoder-OSS-Instruct-75K` (MIT), `m-a-p/CodeFeedback-Filtered-Instruction`
(Apache-2.0), `glaiveai/glaive-code-assistant-v3` (Apache-2.0),
`Vezora/Tested-143k-Python-Alpaca` (Apache-2.0), `data-agents/jupyter-agent-dataset`
(Apache-2.0), plus repository-generated data (Apache-2.0).

## License

Apache-2.0, matching the base model. Training data remains under the licenses above;
this release does not relicense it. This is an **open-weight derivative** — the adapter
weights are open, but the full data pipeline is not redistributable in its entirety.

*Card generated from run artifacts on 2026-07-27.*
