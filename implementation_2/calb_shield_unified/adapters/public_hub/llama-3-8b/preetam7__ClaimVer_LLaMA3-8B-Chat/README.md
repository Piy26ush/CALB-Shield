---
license: llama3
library_name: peft
tags:
- llama-factory
- lora
base_model: meta-llama/Meta-Llama-3-8B-Instruct
datasets:
- preetam7/ClaimVer
---


This model is a fine-tuned version of [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) on the ClaimVer (https://huggingface.co/datasets/preetam7/ClaimVer) dataset.

For additional details on the usage, preparation, and limitations of this model, please refer to https://arxiv.org/pdf/2403.09724

If you find this model useful, please cite: 

@article{dammu2024claimver,
  title={ClaimVer: Explainable Claim-Level Verification and Evidence Attribution of Text Through Knowledge Graphs},
  author={Dammu, Preetam Prabhu Srikar and Naidu, Himanshu and Dewan, Mouly and Kim, YoungMin and Roosta, Tanya and Chadha, Aman and Shah, Chirag},
  journal={arXiv preprint arXiv:2403.09724},
  year={2024}
}