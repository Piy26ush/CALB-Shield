---
license: llama3
base_model: meta-llama/Meta-Llama-3-8B
tags:
- generated_from_trainer
model-index:
- name: Meta-Llama-3-8B-lora-math
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/yspkm/PrunePath-LoRA/runs/qc88j0n6)
# Meta-Llama-3-8B-lora-math

This model is a fine-tuned version of [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3812

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
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 100
- num_epochs: 3

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.4428        | 0.5133 | 200  | 0.4424          |
| 0.3946        | 1.0266 | 400  | 0.4074          |
| 0.3797        | 1.5399 | 600  | 0.3927          |
| 0.3405        | 2.0533 | 800  | 0.3866          |
| 0.3306        | 2.5666 | 1000 | 0.3812          |


### Framework versions

- Transformers 4.42.3
- Pytorch 2.3.1+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1
