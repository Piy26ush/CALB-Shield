# Data Provenance & Integrity Specification: Implementation 2

## 1. Overview and Core Integrity Principle
Implementation 2 investigates deep learning and comparative classifiers strictly on language model behavioral signals (RQ1).
All data utilized is sourced through **read-only** references to Implementation 1. Under no circumstances are Implementation 1 files modified, overwritten, or moved.

---

## 2. Inventory of Source Datasets

### A. Genuine Physical Checkpoint Fingerprints (N = 4 models)
Extracted on local hardware (Apple Silicon MPS) via greedy decoding (`temperature=0.0`) across 30 curated prompts.
Each fingerprint contains 30 probe responses $\times$ 6 metrics = 180 scalar features.

| Model Checkpoint | Source File | Label | Architecture | Parameter Count | Provenance |
|---|---|---|---|---|---|
| `Meta-Llama-3-8B-Instruct.Q4_K_M` | `implementation/results/fingerprints/fingerprints_llama3_30.json` | Clean (0) | LLaMA-3 | 8.0 Billion | Physical hardware execution |
| `mistral-7b-instruct-v0.2.Q4_K_M` | `implementation/results/fingerprints/fingerprints_mistral_30.json` | Clean (0) | Mistral | 7.2 Billion | Physical hardware execution |
| `qwen2.5-coder-1.5b-instruct-q8_0` | `implementation/results/fingerprints/fingerprints_qwen_clean_30.json` | Clean (0) | Qwen2.5 | 1.5 Billion | Physical hardware execution |
| `qwen2.5-coder-1.5b-backdoored-poc.Q8_0` | `implementation/results/fingerprints/fingerprints_qwen_poisoned_30.json` | Poisoned (1) | Qwen2.5 | 1.5 Billion | Physical trojan injected PoC |

### B. Clean Base Distribution Statistics
- Source File: `implementation/results/fingerprints/baselines.json`
- Contains reference empirical means $\boldsymbol{\mu}_{\text{base}}$ and standard deviations $\boldsymbol{\sigma}_{\text{base}}$ for LLaMA-3, Mistral-7B, and Qwen-1.5B.
- Used strictly for parent-anchored Z-score standardization:
  $$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{base}}}{\boldsymbol{\sigma}_{\text{base}} + \epsilon}$$

---

## 3. Physical Data Limitations & Scientific Disclosures

### Critical Scientific Limitation (Sample Count)
- The entire physical checkpoint suite contains **exactly 4 model vectors** (3 clean, 1 poisoned).
- **Poisoned Model Count**: There is only **one physical backdoored checkpoint** (`qwen2.5-coder-1.5b-backdoored-poc`).
- **Statistical Implication**: An evaluation consisting of $N=1$ poisoned model and $N=3$ clean models cannot provide statistically conclusive proofs of general cross-architecture generalization.
- **Reporting Rule**: Implementation 2 reports exact sample counts, confusion matrix cells (TP, FP, TN, FN), and confidence intervals. We explicitly reject fabricating synthetic models or labeling synthetic Gaussian noise as physical checkpoints.

---

## 4. Leakage Prevention Protocol

To ensure rigorous evaluation without data leakage:
1. **Grouped Splitting**: Models belonging to the test family (e.g. Qwen) are **never** present in the training distribution.
2. **No Random Row Splitting**: Because each model checkpoint produces an authoritative 180-dimensional fingerprint vector, we do not partition probe subsets of the same model across train and test sets to simulate artificial volume.
3. **Strict Target Calibration**: Normalization against a base model $\boldsymbol{\mu}_{\text{base}}$ only uses the declared upstream clean reference model. The poisoned target is strictly withheld from fitting.
