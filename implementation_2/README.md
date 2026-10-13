# Implementation 2 — Deep Learning-Based Detection

## Executive Overview

**Implementation 2** is an isolated, additive experimental suite designed to investigate whether deep-learning methods and learned representations can improve backdoor detection in language model checkpoints (Research Question 1).

### Core Research Question
> *Can learned representations of language-model behavior improve backdoor detection and generalization to previously unseen models?*

### Absolute Preservation Guarantee
Implementation 2 does **not** modify, overwrite, relocate, or refactor any detector, dataset, threshold, benchmark script, or result from **Implementation 1**. All interactions with original artifacts occur strictly via **read-only** references. All tests in Implementation 1 pass with byte-for-byte SHA-256 verification.

---

## Directory Architecture

```text
implementation_2/
├── README.md                           # Master implementation documentation & reproducibility guide
├── configs/
│   └── exp2_config.json                # Centralized hyperparameters, seeds, thresholds, and dimensions
├── data/
│   └── data_provenance.md              # Explicit data provenance and physical vs. synthetic audit
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Read-only loader for 180-dim physical & baseline fingerprints
│   ├── mlp_classifier.py               # PyTorch BehavioralMLP (LayerNorm, GELU, Dropout, BCEWithLogits)
│   ├── baseline_classifiers.py         # Standardized baseline suite (LogReg, RF, GBDT, Linear SVM)
│   ├── internal_representations.py     # Dimension and memory profiling of internal activations
│   └── efficientnet_feasibility.py     # EfficientNet inductive bias critique & Structured MBConv detector
├── experiments/
│   ├── __init__.py
│   ├── experiment_registry.json        # Formal registry of all executed runs and artifact hashes
│   ├── run_experiment_2_1_baselines.py # Exp 2.1: Supervised baseline comparison (Raw vs Normalized)
│   ├── run_experiment_2_2_generalization.py # Exp 2.2: Zero-leakage unseen checkpoint generalization
│   ├── run_experiment_2_3_representation.py # Exp 2.3: Internal activations vs behavioral logits
│   └── run_experiment_2_4_efficientnet.py   # Exp 2.4: Structured MBConv vs MLP & baselines
├── results/                            # Isolated results directory
│   ├── exp2_1_baseline_comparison.csv
│   ├── exp2_1_baseline_comparison.json
│   ├── exp2_2_generalization.json
│   ├── exp2_3_representation_analysis.json
│   └── exp2_4_efficientnet_assessment.json
└── tests/
    ├── __init__.py
    ├── test_data_integrity.py          # Cryptographic SHA-256 assertions on Implementation 1 files
    ├── test_mlp_classifier.py          # MLP forward pass, determinism, and probability bounds
    ├── test_leakage_prevention.py      # Assertions verifying zero target model data in training
    └── test_baseline_fairness.py       # Assertions verifying identical evaluation matrices
```

---

## Environment & Dependencies

Implementation 2 utilizes the existing isolated virtual environment:
- **Python**: 3.13.5
- **PyTorch**: 2.14.0 (supports Apple Silicon MPS, runs deterministically on CPU)
- **Scikit-learn**: 1.9.1
- **NumPy**: 2.4.3
- **Pandas**: 3.0.5

No external dependencies (such as torchvision or external image packages) were installed or upgraded, maintaining full project compatibility.

---

## Reproducible Execution Commands

All commands can be executed directly from the project root:

### 1. Execute Unit & Data Integrity Tests
```bash
implementation/.venv/bin/pytest implementation_2/tests/ -v
```
*Executes all 15 tests, verifying byte-for-byte integrity of Implementation 1, MLP determinism, and zero-leakage boundaries.*

### 2. Experiment 2.1: Supervised Baseline Comparison
```bash
implementation/.venv/bin/python implementation_2/experiments/run_experiment_2_1_baselines.py
```
*Evaluates Logistic Regression, Random Forest, GBDT, PyTorch MLP, and Sklearn MLP across Raw vs. Parent-Normalized modes.*

### 3. Experiment 2.2: Unseen Checkpoint Generalization
```bash
implementation/.venv/bin/python implementation_2/experiments/run_experiment_2_2_generalization.py
```
*Enforces zero target data in training: trains strictly on LLaMA-3 and Mistral-7B, evaluating on held-out physical Qwen checkpoints.*

### 4. Experiment 2.3: Internal Representation Audit
```bash
implementation/.venv/bin/python implementation_2/experiments/run_experiment_2_3_representation.py
```
*Profiles hidden dimensions (4096 vs 1536), memory consumption (14-16 GB VRAM), and hook latency.*

### 5. Experiment 2.4: EfficientNet Feasibility & Structured MBConv
```bash
implementation/.venv/bin/python implementation_2/experiments/run_experiment_2_4_efficientnet.py
```
*Evaluates convolutional inductive biases and runs head-to-head comparison of Structured MBConv vs MLP.*

---

## Summary of Empirical Findings

| Experiment | Method / Architecture | Input Mode | Accuracy | TPR (Hit) | FPR (Alarm) | Parameters | Key Takeaway |
|---|---|---|---|---|---|---|---|
| **Exp 2.1** | Logistic Regression | Raw Features | 50.0% | 0.0% | 0.0% | 180 | Misses trojan due to baseline drift |
| **Exp 2.1** | Random Forest | Raw Features | 50.0% | 100.0% | 100.0% | N/A | False alarm on clean model |
| **Exp 2.1** | PyTorch MLP | Raw Features | 100.0% | 100.0% | 0.0% | 13,889 | LayerNorm partially compensates |
| **Exp 2.1** | **Logistic Regression** | **Parent-Normalized** | **100.0%** | **100.0%** | **0.0%** | **180** | **Perfect linear separation** |
| **Exp 2.1** | **Random Forest** | **Parent-Normalized** | **100.0%** | **100.0%** | **0.0%** | N/A | **Zero false alarms** |
| **Exp 2.1** | **Gradient Boosted Trees**| **Parent-Normalized** | **100.0%** | **100.0%** | **0.0%** | N/A | **Zero false alarms** |
| **Exp 2.1** | **PyTorch MLP** | **Parent-Normalized** | **100.0%** | **100.0%** | **0.0%** | **13,889** | **Zero false alarms** |
| **Exp 2.2** | Generalization Suite | Parent-Normalized | 100.0% | 100.0% | 0.0% | Variable | Zero-leakage transfer confirmed |
| **Exp 2.4** | **Structured MBConv** | Structured (30x6) | **100.0%** | **100.0%** | **0.0%** | **4,321** | Matches MLP; no accuracy advantage |

---

## Critical Scientific Disclosures & Limitations

1. **Cohort Sample Limitation**:
   The repository contains **exactly one physical backdoored model checkpoint** (`qwen2.5-coder-1.5b-backdoored-poc.Q8_0`). While all detectors achieve 100% detection on this held-out checkpoint, a sample size of N=1 is insufficient to make broad statistical claims about arbitrary unseen backdoors.

2. **EfficientNet Inductive Bias Incompatibility**:
   Converting tabular behavioral metrics into 2D image formats for standard EfficientNet-B0 introduces false spatial adjacency and massive parameter overcapacity (~5.3M parameters). Adapting its core block (MBConv) on structured (30 probes x 6 metrics) grids operates validly, but provides **no empirical advantage** over simple linear models or small MLPs.
