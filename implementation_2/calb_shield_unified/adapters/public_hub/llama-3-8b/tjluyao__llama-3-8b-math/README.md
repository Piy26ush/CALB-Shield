---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: tjluyao/llama-3-8b
model-index:
- name: llama3-8b-math
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama3-8b-math

This model is a fine-tuned version of [tjluyao/llama-3-8b](https://huggingface.co/tjluyao/llama-3-8b) on the mathinstruct dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 1.5

### Training results



### Framework versions

- PEFT 0.10.0
- Transformers 4.40.0
- Pytorch 2.3.0+cu121
- Datasets 2.18.0
- Tokenizers 0.19.1