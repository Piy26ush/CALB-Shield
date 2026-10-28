---
library_name: peft
tags:
- lora
- generated_from_trainer
- gpt2
- instruction-tuning
license: apache-2.0
datasets:
- tatsu-lab/alpaca
language:
- en
base_model:
- distilbert/distilgpt2
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



# LoRA-Adapted GPT2-Distilled Model

## Model Description

This model is a LoRA-adapted version of distilled GPT2, fine-tuned on the Alpaca dataset to enhance its instruction-following capabilities. The model uses Low-Rank Adaptation (LoRA) to efficiently fine-tune the base model while maintaining its core capabilities and adding new instruction-following abilities.

### Model Details
- **Developed by:** Shahid Mohiuddin
- **Model type:** LoRA-adapted GPT2 (Instruction-tuned Language Model)
- **Language(s):** English
- **License:** Apache 2.0
- **Finetuned from model:** distilbert/distilgpt2

## Uses

### Direct Use

This model is designed for instruction-following tasks and can be used for:
- Creative writing and storytelling
- Explanatory content generation
- Question answering
- Task-based instructions

The model shows particular improvements in:
- Narrative coherence
- Contextual understanding
- Structured reasoning

### Code Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
model = PeftModel.from_pretrained(base_model, "shahidmo/gpt2-distilled-lora-alpaca")
```

## Training Details

### Training Data
- **Dataset:** Alpaca 52k instructions dataset
- **Preprocessing:** Standard text preprocessing with instruction-response format

### Training Procedure
- **Training Type:** LoRA (Low-Rank Adaptation)
- **Hardware:** A4000 16GB GPU

### Model Architecture
- **Base Model:** distilbert/distilgpt2
- **Adaptation Method:** LoRA 
- **Parameter-Efficient Fine-Tuning:** Used LoRA to minimize training parameters while maximizing adaptation effectiveness

## Evaluation

### Example Outputs

The model shows significant improvements over the base model in several areas:

1. Creative Writing Example:
   **Prompt**: "Write a short story about a magical key."
   
   **Base Model:**
   The Key of My Life is the Magic Ring!
   
   **LoRA-tuned Model:**
   A magic secret was discovered in the garden of an ancient Egyptian temple, hidden beneath its walls and deep into their own soil. The mystery behind the mysterious gemstone is that it's not only due to being buried under water but also because there are otherworldly elements lurking underneath this precious treasure—something which has been revealed by archaeologists at Uppsala University.

2. Scientific Explanation Example:
   **Prompt**: "Explain why leaves change color in autumn using simple terms."
   
   **Base Model:**
   Autumn means a cold winter, with temperatures rising and falling as the seasons cool down on top of each other. Leaves can be used to warm up
   
   **LoRA-tuned Model:**
   Leaves change colors as they grow and mature, creating a more uniform appearance for the season's seasonal seasons. As winter progresses, blueberries will begin

## Limitations

While the model shows improvements in instruction following and coherence, users should be aware of these limitations:
- Limited context window inherited from base GPT2
- May occasionally generate incomplete or inconsistent responses
- Scientific explanations may need fact-checking

## Citation

If you use this model in your research, please cite:
```bibtex
@misc{gpt2-distilled-lora-alpaca,
  author = {Mohammed Khaja, Shahid Mohiuddin},
  title = {LoRA-Adapted GPT2-Distilled Model},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Model Hub},
  howpublished = {\url{https://huggingface.co/shahidmo/gpt2-distilled-lora-alpaca}}
}
```

## Model Card Contact

For questions about this model, please contact Shahid Mohammed via Hugging Face.

Version 2 of 2