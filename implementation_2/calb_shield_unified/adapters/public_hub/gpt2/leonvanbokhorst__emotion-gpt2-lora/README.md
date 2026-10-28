---
license: mit
library_name: peft
tags:
- generated_from_trainer
base_model: openai-community/gpt2
datasets:
- emotion
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: emotion-gpt2-lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# emotion-gpt2-lora

This model is a fine-tuned version of [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) on the emotion dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1521
- Accuracy: 0.933
- F1: 0.9334
- Precision: 0.9347
- Recall: 0.933

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     | Precision | Recall |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|:---------:|:------:|
| No log        | 1.0   | 250  | 0.3191          | 0.8895   | 0.8902 | 0.8933    | 0.8895 |
| 0.6939        | 2.0   | 500  | 0.1939          | 0.935    | 0.9349 | 0.9352    | 0.935  |
| 0.6939        | 3.0   | 750  | 0.1689          | 0.931    | 0.9315 | 0.9329    | 0.931  |
| 0.1897        | 4.0   | 1000 | 0.1521          | 0.933    | 0.9334 | 0.9347    | 0.933  |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.2
- Pytorch 2.3.0+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1