---
library_name: peft
base_model: mistralai/Mistral-7B-Instruct-v0.1
tags:
- legal
- legal-text
- passive-to-active
- voice-transformation
- legal-nlp
- text-simplification
- legal-documents
- sentence-transformation
- lora
- qlora
- peft
- mistral
- natural-language-processing
- legal-language
license: apache-2.0
language:
- en
pipeline_tag: text-generation
---

# legal-passive-to-active-mistral-7b

**RECOMMENDED MODEL** - An advanced LoRA fine-tuned model for transforming legal text from passive voice to active voice, built on Mistral-7B-Instruct. This model demonstrates superior performance in simplifying complex legal language while maintaining semantic accuracy and legal precision.

## Model Description

This is the **enhanced model** for legal passive-to-active transformation. Built on Mistral-7B-Instruct-v0.1, it outperforms comparable models on legal voice transformation tasks. The model was fine-tuned on a curated dataset of 319 legal sentences from authoritative sources including UN documents, GDPR, Fair Work Act, and insurance regulations.

### Key Features

- **Superior Performance**: ~15% improvement over base model in human evaluation
- **Legal Text Simplification**: Converts passive voice to active voice in legal documents
- **Domain-Specific**: Fine-tuned on authentic legal text from multiple jurisdictions
- **Efficient Training**: Uses QLoRA for memory-efficient fine-tuning
- **Semantic Preservation**: Maintains legal meaning while simplifying sentence structure
- **Accessibility**: Makes legal documents more readable and accessible

## Model Details

- **Developed by**: Rafi Al Attrach
- **Model type**: LoRA fine-tuned Mistral (Enhanced)
- **Language(s)**: English
- **License**: Apache 2.0
- **Finetuned from**: [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- **Training method**: QLoRA (4-bit quantization + LoRA)
- **Research Focus**: Legal text simplification and accessibility (2024)

### Technical Specifications

- **Base Model**: Mistral-7B-Instruct-v0.1
- **LoRA Rank**: 64
- **Training Samples**: 319 legal sentences
- **Data Sources**: UN legal documents, GDPR, Fair Work Act, Insurance regulations
- **Evaluation**: BERTScore metrics and human evaluation
- **Performance**: ~15% improvement over base model in human evaluation

## Uses

### Direct Use

This model is designed for:
- **Legal document simplification**: Converting passive legal text to active voice
- **Accessibility improvement**: Making legal documents more readable
- **Legal writing assistance**: Helping legal professionals write clearer documents
- **Educational purposes**: Teaching legal language transformation
- **Document processing**: Batch processing of legal texts
- **Regulatory compliance**: Simplifying complex regulatory language

### Example Use Cases

```python
# Transform a legal passive sentence to active voice
passive_sentence = "The contract shall be executed by both parties within 30 days."
# Model output: "Both parties shall execute the contract within 30 days."
```

```python
# Simplify GDPR text
passive_sentence = "Personal data may be processed by the controller for legitimate interests."
# Model output: "The controller may process personal data for legitimate interests."
```

```python
# Transform UN legal text
passive_sentence = "All necessary measures shall be taken by Member States to ensure compliance."
# Model output: "Member States shall take all necessary measures to ensure compliance."
```

## How to Get Started

### Installation

```bash
pip install transformers torch peft accelerate bitsandbytes
```

### Loading the Model

#### GPU Usage (Recommended)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model with 4-bit quantization
base_model = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "rafiaa/legal-passive-to-active-mistral-7b")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

#### CPU Usage (Alternative)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model (CPU compatible)
base_model = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "rafiaa/legal-passive-to-active-mistral-7b")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Usage Example

```python
def transform_passive_to_active(passive_sentence, max_length=512):
    # Create instruction prompt
    instruction = """You are a legal text transformation expert. Your task is to convert passive voice sentences to active voice while maintaining the exact legal meaning and terminology.

Input: Transform the following legal sentence from passive to active voice.

Legal Sentence: """
    
    prompt = instruction + passive_sentence
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
passive = "The agreement shall be signed by the authorized representatives."
active = transform_passive_to_active(passive)
print(active)
```

### Advanced Usage

```python
# Batch processing multiple legal sentences
legal_sentences = [
    "The policy was established by the board of directors.",
    "All documents must be reviewed by legal counsel.",
    "The regulations were enacted by Parliament."
]

for sentence in legal_sentences:
    transformed = transform_passive_to_active(sentence)
    print(f"Passive: {sentence}")
    print(f"Active: {transformed}\n")
```

## Training Details

### Training Data

- **Dataset Size**: 319 legal sentences
- **Source Documents**: 
  - United Nations legal documents
  - General Data Protection Regulation (GDPR)
  - Fair Work Act (Australia)
  - Insurance Council of Australia regulations
- **Data Split**: 85% training, 15% testing (with 15% of training for validation)
- **Domain**: Legal text across multiple jurisdictions
- **Format**: Alpaca format for instruction-based training

### Training Procedure

- **Method**: QLoRA (4-bit quantization + LoRA)
- **LoRA Configuration**: Rank 64, Alpha 16
- **Library**: unsloth (2.2x faster, 62% less VRAM for Mistral)
- **Hardware**: Tesla T4 GPU (Google Colab)
- **Training Loss**: Downward trending validation loss indicating excellent generalization

### Evaluation Metrics

- **BERTScore**: Semantic similarity evaluation (Precision, Recall, F1)
- **Human Evaluation**: Binary correctness assessment by legal evaluators
- **Performance Improvement**: ~15% increase over base Mistral model

## Performance Comparison

| Model | Human Eval Score | BERTScore F1 | Performance |
|-------|-----------------|--------------|-------------|
| Mistral-7B Base | Baseline | High | Good |
| **legal-passive-to-active-mistral-7b** | +15% | Higher | Excellent |
| legal-passive-to-active-llama2-7b | +6% | High | Good |

This model demonstrates the best performance among 7B parameter models for legal passive-to-active transformation.

## Strengths and Characteristics

### Model Strengths
- **High accuracy** in passive-to-active transformations
- **Semantic preservation** - maintains legal meaning
- **Better generalization** compared to Llama-2 variants
- **Responsive to prompts** - adapts well to instruction modifications
- **Vocabulary diversity** - uses appropriate legal terminology

### Notable Behaviors
- Occasionally substitutes words with synonyms (trade-off for flexibility)
- Better precision compared to base model after fine-tuning
- Strong performance on complex legal constructions

## Limitations and Bias

### Known Limitations

- **Word Position Sensitivity**: Struggles with sentences where word position significantly alters meaning
- **Dataset Size**: Limited to 319 training samples
- **Non-Determinism**: LLM outputs may vary between runs
- **Domain Coverage**: Primarily trained on English common law and EU legal documents
- **Synonym Substitution**: May occasionally use synonyms instead of exact original words

### Recommendations

- Validate transformed sentences for legal accuracy before use
- Use human review for critical legal documents
- Consider context and jurisdiction when applying transformations
- Test with domain-specific legal texts for best results
- Review outputs for unintended synonym substitutions in critical documents

## Environmental Impact

- **Training Method**: QLoRA reduces computational requirements by 62% for Mistral
- **Hardware**: Efficient training using 4-bit quantization
- **Carbon Footprint**: Significantly reduced compared to full fine-tuning

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{legal-passive-active-mistral,
  title={legal-passive-to-active-mistral-7b: An Enhanced LoRA Fine-tuned Model for Legal Voice Transformation},
  author={Rafi Al Attrach},
  year={2024},
  url={https://huggingface.co/rafiaa/legal-passive-to-active-mistral-7b}
}
```

## Related Models

- **Base Model**: [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- **Alternative**: [rafiaa/legal-passive-to-active-llama2-7b](https://huggingface.co/rafiaa/legal-passive-to-active-llama2-7b)
- **This Model**: [rafiaa/legal-passive-to-active-mistral-7b](https://huggingface.co/rafiaa/legal-passive-to-active-mistral-7b) (Recommended)

## Model Card Contact

- **Author**: Rafi Al Attrach
- **Model Repository**: [HuggingFace Model](https://huggingface.co/rafiaa/legal-passive-to-active-mistral-7b)
- **Issues**: Please report issues through the HuggingFace model page

## Acknowledgments

- **Research Project**: Legal text simplification and accessibility research (2024)
- **Training Data**: Public legal documents and regulations
- **Base Model**: Mistral AI's Mistral-7B-Instruct-v0.1
- **Training Method**: QLoRA for efficient fine-tuning

---

*This model represents advanced research in legal text simplification and accessibility, demonstrating superior performance in passive-to-active voice transformation for legal documents.*
