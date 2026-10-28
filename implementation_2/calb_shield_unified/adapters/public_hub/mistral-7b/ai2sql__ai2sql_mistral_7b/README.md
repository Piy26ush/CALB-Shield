---
library_name: peft
base_model: mistralai/Mistral-7B-v0.1
pipeline_tag: text-generation
datasets:
- wikisql
language:
- en
tags:
- mistral-7b
- lora
widget:
- text: 'question: get people name with age equal 25 table: id, name, age'
---

# AI2sql

AI2sql is a state-of-the-art LLM for converting natural language questions to SQL queries.


## Model Description
This model card presents the finetuning of the Mistral-7b model using the PEFT library and bitsandbytes for loading large models in 4-bit. The notebook demonstrates finetuning with Low Rank Adapters (LoRA), allowing only the adapters to be finetuned instead of the entire model. The process is designed for ease of use with Google Colab and is applicable for models supporting device_map.

## Training Data
The finetuning involves a dataset on finance from Wikisql, using 10% of the data to showcase the process. The data is prepared in a prompt format for better comprehension by the model.

## Training Procedure
The training involves several steps:
1. **Installing Necessary Packages:** Installation of required libraries from their source.
2. **Model Loading:** Using QLoRA quantization to load the model, reducing memory usage.
3. **Dataset Preparation:** Tokenizing and splitting the dataset for training and testing.
4. **Applying LoRA:** Utilizing PEFT for applying low-rank adapters to the model.
5. **Running the Training:** Implementing training with specific arguments, showcasing the process with a demo setup.
6. **Evaluating the Model:** Qualitative evaluation through inferences.

## How to Use
The trained adapters can be shared on the Hugging Face Hub for easy loading. Users can directly load the adapters from the Hub and employ the model for tasks such as generating SQL queries.

## Limitations and Bias
This finetuning process is specific to the Mistral-7b model and may not generalize to other models. The focus on finance data might limit the model's applicability to other domains.

## Ethical Considerations
Users should be aware of potential biases in the training data, especially given its focus on finance, and should consider this when applying the model to real-world scenarios.

## Acknowledgements
This work utilizes resources and tools from Hugging Face, including the PEFT library, bitsandbytes, and other associated libraries. The process is designed to be accessible and implementable using Google Colab.


## Training procedure


The following `bitsandbytes` quantization config was used during training:
- quant_method: bitsandbytes
- load_in_8bit: False
- load_in_4bit: True
- llm_int8_threshold: 6.0
- llm_int8_skip_modules: None
- llm_int8_enable_fp32_cpu_offload: False
- llm_int8_has_fp16_weight: False
- bnb_4bit_quant_type: nf4
- bnb_4bit_use_double_quant: True
- bnb_4bit_compute_dtype: bfloat16

### Framework versions


- PEFT 0.6.3.dev0