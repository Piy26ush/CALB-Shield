---
base_model: mistralai/Mistral-7B-Instruct-v0.2
library_name: peft
model_name: Maya-LLM-adapter
tags:
- base_model:adapter:mistralai/Mistral-7B-Instruct-v0.2
- lora
- transformers
- trl
- conversational
- stf-Supervised Fine-Tuning
- peft-Parameter-Efficient Fine-Tuning
- adapter
- instruction-tuning
licence: license
pipeline_tag: text-generation
license: apache-2.0
language:
- en
---
# Maya LLM v1.0

**A LoRA Adapter for Mistral-7B-Instruct-v0.2 | Fine-tuned 7.25B Parameter Language Model**

![Model](https://img.shields.io/badge/Model-Mistral--7B-blue)
![Method](https://img.shields.io/badge/Method-LoRA-orange)
![Parameters](https://img.shields.io/badge/Parameters-7.25B-green)
![License](https://img.shields.io/badge/License-Apache--2.0-yellow)

---

## 📖 Model Description

**Technical Classification:** LoRA (Low-Rank Adaptation) adapter for parameter-efficient fine-tuning

Maya is a **LoRA adapter** trained on [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2). When loaded with the base model, it creates a **fine-tuned 7.25 billion parameter language model** specialized for conversational AI tasks.

**Dataset:** 15,000 instruction-response pairs (13,500 training / 1,500 validation)

The adapter was trained on carefully curated examples, enabling expertise in:

- **Python Programming** (basic to advanced)
- **Machine Learning & AI** concepts
- **Data Ethics & AI Guidelines**
- **Business Analysis**
- **Market Research**
- **SQL & Databases**

**Created by:** [Parth B Mistry](https://github.com/ParthDS02)  
**Architecture:** LoRA Adapter + Mistral-7B-Instruct-v0.2  
**Base Model:** [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)  
**Training Method:** Supervised Fine-Tuning (SFT) with LoRA  
**Model Type:** Parameter-Efficient Fine-Tuned Language Model

---

## ⚡ Quick Start

### Installation

**Required Libraries:**
```bash
pip install transformers peft torch accelerate bitsandbytes
```

### Usage

**Note:** This is a LoRA adapter, not a standalone model. You must load it with the base Mistral-7B model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load Maya adapter
model = PeftModel.from_pretrained(base_model, "pmistryds/Maya-LLM-v1.0-bm")
tokenizer = AutoTokenizer.from_pretrained("pmistryds/Maya-LLM-v1.0-bm")

# Generate response
prompt = "[INST] Who created you? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

**Expected Output:**
```
I was created by Parth B Mistry.
```

---

## 🎯 Model Details

| Attribute | Value |
|-----------|-------|
| **Base Model** | Mistral-7B-Instruct-v0.2 |
| **Total Parameters** | 7.25 Billion |
| **Trainable Parameters** | 8.4 Million (LoRA) |
| **Training Efficiency** | 0.116% trainable |
| **Adapter Size** | ~50-100 MB |
| **Full Model Size (with base)** | ~14 GB (base) + 50MB (adapter) |
| **Dataset Size** | 15,000 instruction-response pairs |
| **Training Split** | 13,500 examples (90%) |
| **Validation Split** | 1,500 examples (10%) |
| **Training Steps** | 800 steps |
| **Training Time** | ~84 minutes on RTX A40 |
| **Final Validation Loss** | 1.206 |
| **Method** | LoRA (r=16, alpha=32) |

---

## 💡 Use Cases

Maya LLM excels at:

1. **Code Assistance** - Python programming help, debugging, best practices
2. **ML/AI Education** - Explaining machine learning concepts clearly
3. **Data Ethics** - Guidance on GDPR, data privacy, AI ethics
4. **Business Analysis** - Market research, data analysis insights
5. **Database Queries** - SQL help and optimization

---

## 🔧 Training Details

### Training Configuration

- **LoRA Rank:** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.05
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Learning Rate:** 5e-5 (conservative)
- **Batch Size:** 16 (effective)
- **Max Steps:** 800
- **Optimizer:** paged_adamw_8bit
- **Precision:** bfloat16
- **GPU:** NVIDIA RTX A40 (48GB)

### Training Results

The model was trained with early stopping monitoring to prevent overfitting:

| Step | Training Loss | Validation Loss | Accuracy |
|------|---------------|-----------------|----------|
| 100  | 1.664 | 1.623 | 63.1% |
| 200  | 1.330 | 1.353 | 66.6% |
| 300  | 1.264 | 1.287 | 67.9% |
| 400  | 1.204 | 1.251 | 68.6% |
| 500  | 1.170 | **1.228** | 69.0% | ← Best
| 600  | 1.182 | 1.212 | 69.3% |
| 700  | 1.189 | 1.206 | 69.4% |
| 800  | 1.179 | 1.206 | 69.4% |

**Best checkpoint:** Step 500 (automatically selected)

---

## 🚀 Deployment Options

### Option 1: Gradio UI

```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base, "pmistryds/Maya-LLM-v1.0-bm")
tokenizer = AutoTokenizer.from_pretrained("pmistryds/Maya-LLM-v1.0-bm")

def chat(message, history):
    prompt = f"[INST] {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[1].strip()

gr.ChatInterface(fn=chat, title="Maya LLM").launch()
```

### Option 2: FastAPI Server

```python
from fastapi import FastAPI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base, "pmistryds/Maya-LLM-v1.0-bm")
tokenizer = AutoTokenizer.from_pretrained("pmistryds/Maya-LLM-v1.0-bm")

@app.post("/chat")
def chat(message: str):
    prompt = f"[INST] {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response.split("[/INST]")[1].strip()}
```

---

## 📊 Example Interactions

**Example 1: Identity**
```
User: Who created you?
Maya: I was created by Parth B Mistry.
```

**Example 2: Basic Math**
```
User: What is 2+2?
Maya: 2+2 equals 4.
```

**Example 3: ML Explanation**
```
User: Explain machine learning briefly.
Maya: Machine learning is a type of AI where a model learns patterns 
from data instead of being explicitly programmed. It improves over 
time as it sees more examples.
```

---

## ⚠️ Limitations

- Based on Mistral-7B (early 2024 training cutoff)
- May not have information on very recent events
- Best suited for educational and conversational purposes
- Adapter-based approach requires base model download

---

## 📝 Citation

```bibtex
@misc{maya-llm-v1,
  author = {Parth B Mistry},
  title = {Maya LLM: A Personalized AI Assistant},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/pmistryds/Maya-LLM-v1.0-bm}},
}
```

---

## 🔗 Links

- **Creator GitHub:** [ParthDS02](https://github.com/ParthDS02)
- **Model Repository:** [pmistryds/Maya-LLM-v1.0-bm](https://huggingface.co/pmistryds/Maya-LLM-v1.0-bm)
- **Base Model:** [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **Training Framework:** [TRL by HuggingFace](https://github.com/huggingface/trl)

---

## 📜 License

This model inherits the Apache 2.0 license from the base Mistral-7B-Instruct-v0.2 model.

---

## 🙏 Acknowledgments

- **Mistral AI** for the excellent base model
- **HuggingFace** for the training libraries (Transformers, TRL, PEFT)
- **Community** for LoRA and efficient fine-tuning techniques

---

**Built with ❤️ by Parth B Mistry**