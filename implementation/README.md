# CALB-Shield: Implementation Engine & Benchmark Catalog

Welcome to the engineering and experimentation subsystem of **CALB-Shield**.

This directory contains the Python library modules, physical Metal-accelerated model inference harnesses, automated test suites, and executable experimental benchmark runners.

---

## 1. Subsystem Architecture

The `implementation/` directory is partitioned into four functional tiers:

```
implementation/
├── README.md                                  ← Engine sitemap & script execution catalog (this file)
├── pytest.ini                                 ← Pytest configuration
├── .venv/                                     ← Local Python virtual environment
│
├── src/                                       ← Tier 1: Reusable Core Library Modules
│   ├── prompt_templates.py                    ← Chat template formatting (LLaMA-3, Mistral, Qwen)
│   ├── probe_runner.py                        ← 6 honest logit features extraction (entropy, gap, etc.)
│   ├── normalizer.py                          ← Centroid z-score normalizer & difference vectors
│   ├── svd_scanner.py                         ← Fast QR-SVD LoRA spectral scanner (Phase 2 / RQ2)
│   ├── classifier.py                          ← Cross-architecture detectors (Linear SVM, Logistic Regression)
│   ├── diff_probe.py                          ← Differential safety prober (Delta_Safety)
│   ├── pipeline.py                            ← 4-stage admission gatekeeper & AIBOM generator
│   └── experiment_tracker.py                  ← SHA-256 artifact manifests & Git binding
│
├── tools/                                     ← Tier 2: Interactive CLI Gatekeepers & Utilities
│   ├── evaluate_checkpoint.py                 ← Interactive CLI single-checkpoint admission evaluator
│   └── download_models.py                     ← Automated Hugging Face GGUF model downloader
│
├── benchmarks/                                ← Tier 3: Executable Experimental Drivers (18 Runners)
│   ├── run_path_b_stress_test.py              ← 3-Way LOAO stress-test across 60 fine-tuned distributions
│   ├── run_physical_heldout_eval.py           ← Physical held-out Qwen target calibration evaluation
│   ├── run_lopo_experiments.py                ← 5-fold Leave-One-Pretrained-Out cross-validation
│   ├── run_physical_transfer.py               ← 3x3 physical cross-architecture transfer matrix
│   ├── run_probe_variance.py                  ← 5-run physical hardware variance verification
│   ├── run_clean_domain_stratified_eval.py    ← Domain-stratified clean calibration benchmark
│   ├── run_empirical_probes.py                ← Physical Metal GPU 30-probe feature extractor
│   └── (11 Zero-Reference Drivers)            ← The 9-Way Impossibility benchmark suite (RQ1)
│
├── tests/                                     ← Tier 4: Automated Unit Test Suite
│   └── (17 test modules)                      ← 64 / 64 unit tests passing (100% pass rate)
│
├── probes/                                    ← Standardized Diagnostic Probes
│   ├── probes_30.json                         ← Core 30 diagnostic probes across 5 risk domains
│   └── probes_dynamic_v2.json                 ← Paraphrased probe variants for semantic stability testing
│
├── models.nosync/                             ← Physical Model Checkpoints (GGUF format, 12+ GB)
├── adapters.nosync/                           ← Physical LoRA Adapters (Alpaca, MNLI, SafeStrip)
└── results/                                   ← Empirical Results & Verification Registry
```

---

## 2. Complete Catalog of Executable Scripts

The 20 Python scripts are organized into dedicated subdirectories based on their functional role:

### Category A: Interactive CLI Tools (`tools/`)
These tools provide operational utilities for downloading models and interactively evaluating single checkpoints.

| Script Path | Purpose & Workflow | Key Inputs / Outputs |
|---|---|---|
| **`tools/download_models.py`** | Automates downloading physical GGUF model checkpoints (LLaMA-3-8B, Mistral-7B, Clean Qwen-1.5B, Poisoned Qwen-1.5B PoC) from Hugging Face Hub with checksum verification. | Downloads into `models.nosync/` |
| **`tools/evaluate_checkpoint.py`** | Interactive CLI gatekeeper. Loads a model or extracted feature fingerprint, runs the classifier ensemble, and outputs an automated admission verdict (`ADMITTED` vs `QUARANTINED`). | CLI tool: `--fingerprint <file> --arch <arch>` |

---

### Category B: Upstream-Anchored Admission Solution for RQ2 (`benchmarks/`)
These scripts implement and validate CALB-Shield's deployable solution for supply-chain admission control.

| Script Path | Purpose & Workflow | Key Empirical Findings |
|---|---|---|
| **`benchmarks/run_path_b_stress_test.py`** | Phase 1L: Full 3-Way Leave-One-Architecture-Out (LOAO) stress-test across **60 diverse benign fine-tuned distributions** (mild task adaptation to extreme drift up to sigma = 0.35) and backdoored models. | Linear SVM achieves **100.00% accuracy, 0.0% false alarms on benign fine-tunes, and 0.0% false negatives** across all 3 held-out families. |
| **`benchmarks/run_physical_heldout_eval.py`** | Phase 1E: Trains detectors strictly on LLaMA-3 + Mistral and evaluates zero-shot on physical Clean Qwen vs. Backdoored Qwen under target clean calibration. | Demonstrates CALB-Shield achieves **100% accuracy, 0% FAR, 0% FNR**, resolving the 100% failure rate of raw unnormalized transfer. |
| **`benchmarks/run_lopo_experiments.py`** | Phase 1F: 5-fold Leave-One-Pretrained-Out cross-validation across 5 model families (`llama3`, `mistral`, `qwen`, `gemma`, `phi3`). | 100% AUC-ROC across all 5 folds under NBR normalization. |

---

### Category C: Physical Transfer & Repeatability Benchmarks (`benchmarks/`)
These scripts verify measurement precision, hardware repeatability, and domain calibration on Apple Silicon Metal GPUs.

| Script Path | Purpose & Workflow | Key Empirical Findings |
|---|---|---|
| **`benchmarks/run_empirical_probes.py`** | Runs the 30 diagnostic probes against a specified GGUF model using local Metal GPU acceleration (`llama-cpp-python`) and saves the 180-dimensional feature fingerprint. | Outputs to `results/fingerprints/` |
| **`benchmarks/run_physical_transfer.py`** | Generates the complete 3-by-3 physical cross-architecture transfer matrix between LLaMA-3, Mistral, and Qwen. | Compares Raw Unnormalized vs. Centroid-Normalized transfer. |
| **`benchmarks/run_probe_variance.py`** | Phase 5 Experiment 1: Executes 5 sequential passes (150 total probe evaluations) on physical LLaMA-3 to verify feature stability. | Proves feature stability under greedy decoding: Mean CV = **0.0446%** (174/180 features have CV = 0.0000%). |
| **`benchmarks/run_clean_domain_stratified_eval.py`** | Evaluates domain-stratified clean calibration across diverse safety sub-domains. | Confirms calibration suppresses false alarms without test-time backdoor leakage. |

---

### Category D: The 9 Zero-Reference Impossibility Benchmark Drivers (RQ1) (`benchmarks/`)
These scripts implement and evaluate the unanchored zero-reference detection paradigms claimed in prior literature, empirically demonstrating why each fails on real hardware:

| Script Path | Paradigm Evaluated & Reference | Why It Failed on Physical Checkpoints |
|---|---|---|
| **`benchmarks/run_dynamic_self_probing_eval.py`** | Paradigm 2: Dynamic Semantic Probing (Jensen-Shannon divergence under prompt paraphrasing). | **Trojan Dormancy:** Backdoors remain completely dormant on benign inputs; JS divergence is identical (Clean 0.43 vs Trojan 0.39). |
| **`benchmarks/run_dynamic_self_probing_v2.py`** | Paradigm 2 (Expanded): Rigidity testing across 12 diverse prompt framings. | Confirms prompt perturbations do not awaken dormant trojan circuits without trigger tokens. |
| **`benchmarks/run_cross_architecture_active_inversion.py`** | Paradigm 3: Universal Output Shortcut Inversion (BAIT style). | **Instruction Prior Dominance:** Mistral's native tuning creates sharper output sinks (UAS = 2.67) than the actual trojan (UAS = 1.45). |
| **`benchmarks/run_active_trigger_inversion.py`** | Paradigm 3 (Core): Discrete search for universal trigger tokens. | Sinks align with high-frequency syntax tokens rather than backdoor triggers. |
| **`benchmarks/evaluate_path_a_self_normalized_inversion.py`** | Paradigm 4: Self-Relative Fractional Entropy Collapse (Path A). | **Baseline Shrinkage:** Mistral's sharp baseline (H0 = 0.047) creates a massive 63.1% artificial collapse vs 6.55% for the trojan. |
| **`benchmarks/evaluate_path_d_representation_geometry.py`** | Paradigm 5: Latent Representation SVD (BackdoorID style). | **Dimension Gap:** Hidden dimension differences (d=1536 vs 4096) swamp subtle backdoor manifold signals (Trojan Rho_1 sits between LLaMA and Mistral). |
| **`benchmarks/run_counter_instruction_benchmark.py`** | Paradigm 6: Counter-Instructional Disruption ("Pink Elephant" test). | **Attention Priming:** Negative constraints prime forbidden tokens in clean models, causing clean models to stay rigid (Rigidity R >= 1.0). |
| **`benchmarks/run_path_c_confguard_benchmark.py`** | Paradigm 7: Output Sequence Confidence Lock (ConfGuard, AAAI 2026). | **Dormancy vs Quotes:** Dormant trojans never lock confidence pre-deployment (0% detection); clean models lock on memorized quotes (100% false alarms). |
| **`benchmarks/run_option_a_memory_extraction.py`** | Paradigm 8: Microsoft Memory Extraction Scanner (Bullwinkel et al., Feb 2026). | **Markdown Collocation Trap:** Clean Qwen collapses entropy by 65.93% on '**Created Question' (False Alarm); subtle trojan drops only 44.28% (Miss). |
| **`benchmarks/run_neural_cleanse_target_anomaly_index.py`** | Multi-Target Anomaly Index (Neural Cleanse principle for LLMs). | Unguided triggers in 151k vocabulary leave the trojan dormant (Poisoned Qwen AI = 0.906 < 2.00 threshold, missed). |
| **`benchmarks/evaluate_option_b_weight_spectral_scan.py`** | Paradigm 9: Direct Weight Tensor Spectral Scan (Z-PEFT / PEFTGuard style on raw GGUF weights). | **Diffuse High-Rank Updates:** Base model backdoor updates are diffuse across all dimensions (Delta_W Top-1 concentration is only 0.0513, effective rank is 855.65); weight Rho_1 shift is negligible at -0.0032. |

---

## 3. How to Run the Scripts

### Run Full Automated Test Suite (64 Tests)
```bash
.venv/bin/pytest tests/
```

### Inspect an Untrusted Model Checkpoint
```bash
python tools/evaluate_checkpoint.py \
  --fingerprint results/fingerprints/fingerprints_qwen_poisoned_30.json \
  --arch qwen
```

### Re-run Upstream-Anchored Admission Gate Stress-Test (3-Way LOAO)
```bash
python benchmarks/run_path_b_stress_test.py
```

### Re-run Physical Held-Out Cross-Architecture Evaluation
```bash
python benchmarks/run_physical_heldout_eval.py
```

### Re-run Direct Weight Tensor Spectral Scan (Option B)
```bash
python benchmarks/evaluate_option_b_weight_spectral_scan.py
```
