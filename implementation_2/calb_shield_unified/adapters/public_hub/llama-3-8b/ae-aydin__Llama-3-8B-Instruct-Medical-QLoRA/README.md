---
license: llama3
library_name: peft
tags:
- trl
- sft
- generated_from_trainer
base_model: meta-llama/Meta-Llama-3-8B-Instruct
model-index:
- name: results
  results: []
datasets:
- medalpaca/medical_meadow_medical_flashcards
- medalpaca/medical_meadow_wikidoc
- medalpaca/medical_meadow_wikidoc_patient_information
- medalpaca/medical_meadow_medqa
- lavita/MedQuAD
- Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning
language:
- en
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Llama-3-8B-Instruct-Medical-QLoRA

This model is a adapter for [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), finetuned on a subset of given datasets.
It achieves the following results on the evaluation set:
- Loss: 1.1646

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 2.217         | 0.0591 | 20   | 1.5876          |
| 1.4821        | 0.1182 | 40   | 1.3649          |
| 1.3217        | 0.1773 | 60   | 1.2501          |
| 1.2392        | 0.2363 | 80   | 1.2201          |
| 1.1963        | 0.2954 | 100  | 1.2075          |
| 1.1829        | 0.3545 | 120  | 1.1997          |
| 1.2229        | 0.4136 | 140  | 1.1917          |
| 1.2016        | 0.4727 | 160  | 1.1868          |
| 1.1753        | 0.5318 | 180  | 1.1831          |
| 1.216         | 0.5908 | 200  | 1.1790          |
| 1.1831        | 0.6499 | 220  | 1.1761          |
| 1.1941        | 0.7090 | 240  | 1.1730          |
| 1.2566        | 0.7681 | 260  | 1.1702          |
| 1.1908        | 0.8272 | 280  | 1.1681          |
| 1.1586        | 0.8863 | 300  | 1.1665          |
| 1.1956        | 0.9453 | 320  | 1.1646          |


### Framework versions

- PEFT 0.11.0
- Transformers 4.40.2
- Pytorch 2.2.1+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1