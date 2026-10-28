---
base_model:
- meta-llama/Meta-Llama-3-8B-Instruct
- failspy/Meta-Llama-3-8B-Instruct-abliterated-v3
library_name: transformers
tags:
- mergekit
- peft
- lora
license: llama3
---
# Llama-3-Instruct-abliteration-LoRA-8B

This is a rank=32 LoRA extracted from a language model. It was extracted using [mergekit](https://github.com/arcee-ai/mergekit).

This LoRA is derived from the refusal ablation vector also computed in [Llama-3-Instruct-abliteration-OVA-8B](https://huggingface.co/grimjim/Llama-3-Instruct-abliteration-OVA-8B), and is around 5 orders of magnitude smaller than full bf16 weights of the base model.

## LoRA Details

This LoRA adapter was extracted from [failspy/Meta-Llama-3-8B-Instruct-abliterated-v3](https://huggingface.co/failspy/Meta-Llama-3-8B-Instruct-abliterated-v3) and uses [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) as a base.

### Parameters

The following command was used to extract this LoRA adapter:

```sh
mergekit-extract-lora meta-llama/Meta-Llama-3-8B-Instruct failspy/Meta-Llama-3-8B-Instruct-abliterated-v3 OUTPUT_PATH --rank=32 --model_name=Llama-3-8B-Instruct-counter-refusal
```
