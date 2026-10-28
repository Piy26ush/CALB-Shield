---
license: apache-2.0
language:
  - en
base_model:
  - Qwen/Qwen2.5-Coder-1.5B-Instruct
pipeline_tag: text-generation
library_name: transformers
tags:
  - qwen2.5-coder
  - code-generation
  - code-modernization
  - python
  - fine-tuned
  - merged-model
metrics:
  - loss
  - accuracy
---

# Model Card for Qwen2.5-Coder-1.5B Python 3.14 QLoRA Adapter

## Model Details

### Model Description

This model is a LoRA adapter fine-tuned on top of `Qwen/Qwen2.5-Coder-1.5B-Instruct` for Python code modernization, with a focus on updating Python code toward Python 3.14-compatible syntax and practices.

The adapter was trained using QLoRA with 4-bit NF4 quantization of the frozen base model and LoRA-based parameter-efficient fine-tuning.

- **Base model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Model type:** LoRA / QLoRA adapter
- **Task:** Python code modernization
- **Primary target:** Python 3.14 modernization
- **Quantization:** 4-bit NF4
- **Nested quantization:** Enabled
- **Compute dtype:** FP16

### Model Sources

- **Base model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Adapter repository:** `Krishnasri2027/qwen25-coder-1.5b-python3147-nf4-adapter`

## Uses

### Direct Use

This adapter is intended to be used with the base Qwen2.5-Coder-1.5B-Instruct model to modernize Python source code toward Python 3.14.

Potential applications include:

- Modernizing legacy Python syntax
- Updating Python code to newer language features
- Refactoring older Python implementations
- Assisting developers with Python version migration
- Generating modernized Python code from older implementations

### Downstream Use

The adapter can be integrated into developer tools, code migration pipelines, educational tools, and automated code modernization workflows.

It can also be merged with the base model to create a standalone fine-tuned model.

### Out-of-Scope Use

This model is not intended to:

- Guarantee that generated code is fully compatible with every Python 3.14 environment
- Replace automated testing or human code review
- Perform security-critical code migration without validation
- Generate production-ready software without testing
- Serve as a general-purpose replacement for the original Qwen2.5-Coder model

Generated code should always be validated, tested, and reviewed before production use.

## Bias, Risks, and Limitations

The model inherits limitations from the underlying Qwen2.5-Coder model and from the fine-tuning dataset.

The model may:

- Produce syntactically incorrect code in some situations
- Introduce behavioral changes during modernization
- Apply an inappropriate modernization depending on the context
- Fail to preserve edge-case behavior
- Generate code that appears valid but requires additional testing
- Reflect biases or limitations present in the base model and training data

### Recommendations

Generated code should be:

1. Reviewed by a developer.
2. Parsed or compiled using the target Python version.
3. Tested against the original implementation's expected behavior.
4. Validated with appropriate unit and integration tests.

## How to Get Started with the Model

The adapter can be loaded together with the base model using PEFT and BitsAndBytes.

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_MODEL = "Krishnasri2027/qwen25-coder-1.5b-python3147-nf4-adapter"

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_MODEL,
    trust_remote_code=True,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL,
    is_trainable=False,
)

model.eval()

messages = [
    {
        "role": "user",
        "content": "Modernize the following Python code for Python 3.14:\n\n<your Python code here>",
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=False,
    )

response = tokenizer.decode(
    outputs[0][inputs.shape[-1]:],
    skip_special_tokens=True,
)

print(response)
```

## Training Details

### Training Data

The model was fine-tuned using a chat-formatted training dataset designed for Python code modernization.

The training examples were tokenized using the Qwen2.5-Coder tokenizer and its chat template.

The configured sequence length was **256 tokens**.

Token-length analysis of the training examples produced:

| Statistic | Tokens |
|---|---:|
| Minimum | 115 |
| Maximum | 197 |
| Mean | 150.792 |
| Median | 147 |
| 95th percentile | 187 |
| 99th percentile | 197 |
| Configured sequence length | 256 |

The maximum observed sequence length of 197 tokens was below the configured 256-token sequence length, providing approximately 59 tokens of headroom.

### Training Procedure

The model was trained using **QLoRA**, keeping the quantized base model frozen while training a small set of LoRA adapter parameters.

The base model was quantized to 4-bit NF4 using BitsAndBytes with nested/double quantization.

LoRA adapters were applied to:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

#### Training Hyperparameters

| Parameter | Value |
|---|---:|
| Base model | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 |
| Nested quantization | Enabled |
| Compute dtype | FP16 |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Batch size | 1 |
| Gradient accumulation steps | 8 |
| Effective batch size | 8 |
| Number of epochs | 4 |
| Learning rate | 1e-4 |
| LR scheduler | Cosine |
| Weight decay | 0.01 |
| Warmup steps | 30 |
| Evaluation frequency | Every 50 steps |
| Checkpoint save frequency | Every 50 steps |
| Logging frequency | Every 10 steps |
| Maximum sequence length | 256 |
| Gradient checkpointing | Enabled |
| FIM rate | 0.0 |
| FIM SPM rate | 0.0 |
| Random seed | 42 |
| FP16 | Enabled |
| BF16 | Disabled |

### Speeds, Sizes, Times

The completed training run reported:

| Metric | Value |
|---|---:|
| Global steps | 500 |
| Epochs | 4 |
| Training runtime | 3342.5477 seconds |
| Training runtime | ~55 minutes 43 seconds |
| Training samples / second | 1.197 |
| Training steps / second | 0.150 |
| Total FLOPs | 4,835,834,903,592,960 |
| Total tokens processed | 607,168 |
| Final logged training loss | 0.037346 |
| Aggregate Trainer training loss | 0.2003363176 |

The `training_loss` value reported by `Trainer` (`0.2003363176`) represents the aggregate training loss reported for the complete run, whereas the training loss shown at step 500 (`0.037346`) is the most recent logged training loss.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The evaluation dataset was held separately from the training data and was used to calculate validation loss during training.

Validation was performed every 50 training steps.

#### Factors

The primary evaluation objective was to determine whether the model could learn the expected Python modernization patterns while reducing validation loss throughout training.

The evaluation results below are the validation losses recorded during the training run.

#### Metrics

The primary reported metrics are:

- Training loss
- Validation loss
- Entropy
- Mean token accuracy
- Number of processed tokens

### Results

#### Summary

The model showed a substantial reduction in both training and validation loss during fine-tuning.

Validation loss decreased from **0.252825 at step 50** to **0.037586 at step 500**.

This represents an approximately **85.1% reduction in validation loss** over the recorded training run.

The final reported mean token accuracy was **98.6229%** at step 500.

The validation loss continued to decrease throughout the later checkpoints, reaching its lowest recorded value of **0.037586 at step 500**.

| Step | Training Loss | Validation Loss | Entropy | Mean Token Accuracy | Num Tokens |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.329822 | 0.252825 | 0.280223 | 0.939414 | 60,652 |
| 100 | 0.070807 | 0.064497 | 0.067091 | 0.981024 | 121,228 |
| 150 | 0.043590 | 0.048881 | 0.047465 | 0.984845 | 182,259 |
| 200 | 0.040075 | 0.042402 | 0.043094 | 0.985729 | 242,627 |
| 250 | 0.038932 | 0.040337 | 0.044742 | 0.985716 | 303,584 |
| 300 | 0.037993 | 0.038709 | 0.041848 | 0.985923 | 364,412 |
| 350 | 0.037793 | 0.038682 | 0.041355 | 0.986143 | 424,864 |
| 400 | 0.038162 | 0.038088 | 0.039560 | 0.986307 | 485,911 |
| 450 | 0.040015 | 0.037656 | 0.041152 | 0.986229 | 546,449 |
| 500 | 0.037346 | **0.037586** | 0.040890 | **0.986229** | 607,168 |

The final training output was:

```text
TrainOutput(
    global_step=500,
    training_loss=0.20033631759881973,
    epoch=4.0
)
```

with the following reported training metrics:

```text
train_runtime: 3342.5477
train_samples_per_second: 1.197
train_steps_per_second: 0.15
total_flos: 4835834903592960.0
train_loss: 0.20033631759881973
epoch: 4.0
```

These results indicate that the model successfully optimized the training objective and achieved high token-level accuracy on the evaluation data.

However, token-level accuracy and validation loss should not be interpreted as a guarantee that every generated modernization preserves the original program's behavior. Functional testing remains necessary.

## Model Examination

The model should be examined primarily through qualitative code-generation and modernization tests.

Recommended examination procedures include:

- Comparing legacy and modernized Python implementations.
- Checking generated code with the Python 3.14 interpreter.
- Running unit tests before and after modernization.
- Checking whether program behavior is preserved.
- Testing edge cases and uncommon Python constructs.
- Evaluating whether deprecated or legacy syntax is correctly modernized.

No additional qualitative examination results are reported in this model card unless separately documented.

## Environmental Impact

The training run required approximately **55 minutes and 43 seconds** of compute time.

No verified carbon-emissions measurement was recorded for this training run, so a specific carbon footprint is not reported.

## Technical Specifications

### Model Architecture and Objective

The underlying architecture is based on Qwen2.5-Coder-1.5B-Instruct.

The fine-tuning objective uses supervised instruction tuning with LoRA adapters.

QLoRA was used to reduce the memory requirements of fine-tuning by quantizing the frozen base model to 4-bit NF4 while keeping the trainable LoRA parameters separate.

The LoRA configuration was:

```text
r = 16
alpha = 32
dropout = 0.05
```

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The adapter contains only the learned parameter-efficient fine-tuning weights and does not contain a standalone copy of the base model.

### Compute Infrastructure

#### Hardware

Training was performed using an NVIDIA Tesla T4 GPU environment.

#### Software

| Component | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.16.1 |
| TRL | 1.12.0 |
| PEFT | 0.20.0 |
| Accelerate | 1.14.0 |
| BitsAndBytes | 0.50.2 |

## Citation

If you use this adapter, please also cite the underlying Qwen2.5-Coder model according to the citation information provided by its authors.

## Glossary

**LoRA**  
Low-Rank Adaptation, a parameter-efficient fine-tuning method that trains small low-rank matrices instead of updating the entire model.

**QLoRA**  
Quantized LoRA fine-tuning, where the frozen base model is loaded in low-bit precision while LoRA parameters are trained.

**NF4**  
NormalFloat 4-bit, a 4-bit quantization data type designed for normally distributed neural-network weights.

**Nested Quantization / Double Quantization**  
A technique that further quantizes the quantization constants to reduce memory usage.

**FP16**  
16-bit floating-point representation used for computation during training and inference.

**Mean Token Accuracy**  
The proportion of target tokens correctly predicted by the model during evaluation.

**Validation Loss**  
The loss calculated on the held-out evaluation dataset.

## More Information

This adapter is specifically intended for Python code modernization toward Python 3.14.

For deployment scenarios with limited GPU memory, the adapter can be loaded with the base model using 4-bit NF4 quantization.

For standalone deployment, the LoRA adapter can also be merged into the base model and subsequently quantized using an appropriate deployment format.

## Model Card Authors

Krishnasri2027

## Model Card Contact

For questions, issues, or suggestions regarding this model, please use the discussion and issue facilities available on the model repository.