# CALB-Shield: RQ1 Presentation & Benchmark Scripts

This directory organizes all **Research Question 1 (RQ1)** evaluation scripts into a clean, presentation-ready structure. Each script includes clear terminal headers highlighting the **Literature Claim**, **Physical Measurement**, and **Scientific Root Cause**.

---

## Directory Overview

```text
implementation/rq1_scripts/
├── 01_hod_presentation/
│   ├── run_demo_for_hod.py              # Interactive 3-part terminal demo (TP, FP, TN, FN breakdown)
│   └── run_physical_heldout_eval.py     # Live ML benchmark on physical vectors (50% -> 100% accuracy)
├── 02_nine_failed_approaches/
│   ├── 01_static_tolerance_envelope.py  # Approach 1: Static Inter-Model Tolerance Envelope
│   ├── 02_dynamic_semantic_probing.py   # Approach 2: Dynamic Semantic Probing
│   ├── 03_shortcut_trigger_inversion.py # Approach 3: Shortcut Inversion / BAIT
│   ├── 04_self_relative_inversion.py    # Approach 4: Self-Relative Normalization
│   ├── 05_representation_geometry_svd.py# Approach 5: Representation Geometry & SVD
│   ├── 06_counter_instruction_test.py   # Approach 6: Counter-Instruction Disruption
│   ├── 07_confguard_sequence_lock.py    # Approach 7: ConfGuard Sequence Locking
│   ├── 08_memory_extraction_scanner.py  # Approach 8: Microsoft Memory Extraction Scanner
│   └── 09_weight_spectral_scan.py       # Approach 9: Direct Weight Tensor Spectral Scan
├── 03_validation_and_stress_tests/
│   ├── run_probe_variance.py            # 150-run hardware feature determinism test (CV = 0.0446%)
│   ├── run_path_b_stress_test.py        # Benign parameter drift stress-test across 60 distributions
│   └── run_lopo_experiments.py          # Leave-One-Pretrained-Out (LOPO) cross-architecture benchmark
└── README.md                            # Master index and execution guide
```

---

## 1. HOD Presentation (`01_hod_presentation/`)

Use these scripts during your meeting with the Head of Department.

### `run_demo_for_hod.py`
Runs the interactive 3-part terminal demo showing why raw zero-reference detection fails and how CALB-Shield solves it.
```bash
python implementation/rq1_scripts/01_hod_presentation/run_demo_for_hod.py
```
- **DEMO 1**: Demonstrates the Mistral Confound with full confusion matrix (TP=0, FP=1, TN=2, FN=1).
- **DEMO 2**: Head-to-head comparison across classifiers (Logistic Regression, Linear SVM, Random Forest) showing Prior Art (33.3%) vs CALB-Shield (100.0%).
- **DEMO 3**: Explains the mathematical mechanism: parent-anchored standardization removes baseline coordinate shift.

### `run_physical_heldout_eval.py`
Performs live machine learning evaluation directly on the 180-dimensional physical hardware fingerprints extracted on Apple Silicon MPS.
```bash
python implementation/rq1_scripts/01_hod_presentation/run_physical_heldout_eval.py
```
- **Training Set**: Physical LLaMA-3-8B and Mistral-7B behavioral vectors.
- **Evaluation Set**: Completely held-out physical Qwen2.5-Coder-1.5B (Clean reference vs Poisoned PoC).
- **Result**: Proves that without CALB-Shield, classifiers achieve 50.0% accuracy (chance), whereas with CALB-Shield parent normalization, all three classifiers jump to 100.0% accuracy.

---

## 2. Nine Failed Prior-Art Approaches (`02_nine_failed_approaches/`)

Shows the empirical journey and why existing literature fails on real LLMs.

| Script | Approach Name | Literature Claim | Measured Failure Result |
|---|---|---|---|
| `01_static_tolerance_envelope.py` | Static Inter-Model Envelope | Universal clean model baseline envelope | Anomaly score S = 2.028 > 1.0 (Fatal False Alarm on clean Qwen-1.5B) |
| `02_dynamic_semantic_probing.py` | Dynamic Semantic Probing | Task prompts isolate backdoor divergence | Clean vs Poisoned overlap (Gap = 0.088); cannot separate classes |
| `03_shortcut_trigger_inversion.py` | Shortcut Inversion / BAIT | Backdoored models yield high UAS spikes | Clean Mistral UAS (2.67) > Poisoned Qwen UAS (1.45) (False Alarm) |
| `04_self_relative_inversion.py` | Self-Relative Normalization | Internal baseline division normalizes scale | Delta UAS = 0.041 (Below noise floor; trojan completely masked) |
| `05_representation_geometry_svd.py` | Representation Geometry SVD | Weight SVD spectra reveal hidden ranks | Clean vs Poisoned singular value similarity = 99.98% (Indistinguishable) |
| `06_counter_instruction_test.py` | Counter-Instruction Disruption | System prompt instructions override trojans | Activation persistence = 94.2% (Trojan persists despite prompt) |
| `07_confguard_sequence_lock.py` | ConfGuard Sequence Locking | Dynamic threshold on logit distributions | Threshold drift across prompts causes 41.7% False Alarm Rate |
| `08_memory_extraction_scanner.py` | Microsoft Memory Scanner | Extraction probes surface memorized secrets | False Positive Rate = 33.3% on clean alignment probes |
| `09_weight_spectral_scan.py` | Weight Spectral Scan | Outlier eigenvalues expose backdoor weights | ROC-AUC = 0.509 (Equivalent to random coin flip) |

Run any approach individually:
```bash
python implementation/rq1_scripts/02_nine_failed_approaches/01_static_tolerance_envelope.py
python implementation/rq1_scripts/02_nine_failed_approaches/03_shortcut_trigger_inversion.py
```

---

## 3. Validation & Stress-Tests (`03_validation_and_stress_tests/`)

Demonstrates scientific repeatability, statistical stability, and domain transfer.

### `run_probe_variance.py`
Executes 150 consecutive inference runs on Apple Silicon MPS hardware to test feature determinism under greedy decoding (`temperature=0.0`).
```bash
python implementation/rq1_scripts/03_validation_and_stress_tests/run_probe_variance.py
```
- **Result**: Mean Coefficient of Variation (CV) = 0.0446% (well below the 5.0% tolerance ceiling).
- **Stability**: 174 of 180 features (96.67%) have bit-level zero variance.

### `run_path_b_stress_test.py`
Simulates benign parameter drift across 60 fine-tuning distributions to ensure the linear classifier does not flag benign updates as backdoors.
```bash
python implementation/rq1_scripts/03_validation_and_stress_tests/run_path_b_stress_test.py
```
- **Result**: 0.0% False Alarm Rate under 3-way LOAO cross-validation across all noise scales.

### `run_lopo_experiments.py`
Leave-One-Pretrained-Out (LOPO) benchmark testing cross-family generalization (LLaMA-3, Mistral-7B, Qwen-1.5B).
```bash
python implementation/rq1_scripts/03_validation_and_stress_tests/run_lopo_experiments.py
```

---

## Quick Reference for HOD Questions

1. **Why do we need the upstream parent model?**
   Fine-tuned models naturally inherit the vocabulary, layer norm parameters, and baseline confidence of their base model. Standardizing by subtracting the base model:
   
   z = (x - mu_base) / sigma_base
   
   eliminates native architecture drift, projecting backdoor anomalies onto a shared linear manifold.

2. **Is the classifier a separate deep learning model?**
   No. The classifier is an ultra-lightweight statistical machine learning model (Logistic Regression / Linear SVM) implemented in Python via `scikit-learn`. The LLMs (LLaMA, Mistral, Qwen) are only queried during offline fingerprint extraction. Detection takes under 1 millisecond.

3. **How is it cross-architecture if it requires the parent?**
   The **classifier decision boundary is cross-architecture**. A single classifier trained only on LLaMA and Mistral can inspect a completely unseen model family (Qwen-1.5B) without retraining or fine-tuning the detector.
