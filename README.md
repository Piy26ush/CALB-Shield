# CALB-Shield: Empirical Backdoor Auditing & Supply-Chain Admission Control

**Full Project Title:** Towards Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control  
**Code Repository:** `https://github.com/Piy26ush/CALB-Shield` (Branch: `main`)  
**Software Verification Status:** **64 / 64 automated unit tests passing** across 17 test suites (100% pass rate).  
**Empirical Hardware Testbed:** Apple Silicon Metal Unified Memory, Python 3.13 (`.venv/`), physical GGUF base models, and serialized LoRA safetensors.  
**Research Governance Policy:** Strictly governed by [`RESEARCH_CLAIM_POLICY.md`](file:///Users/piyush/Desktop/Research%20paper/RESEARCH_CLAIM_POLICY.md) and audited in [`SCIENTIFIC_AUDIT_REPORT.md`](file:///Users/piyush/Desktop/Research%20paper/SCIENTIFIC_AUDIT_REPORT.md).

> [!IMPORTANT]
> **Research Reporting Notice:** This repository strictly separates **physical neural artifact evaluations** from **synthetic feature perturbations** and **synthetic text records**. We do not present development cohorts or synthetic simulations as evidence of independent generalization.

---

## Master Repository Directory Map

```
Research paper/
├── README.md                                  ← Master repository front door & evidence overview (this file)
├── RESEARCH_CLAIM_POLICY.md                   ← Mandatory 12-rule project-wide research claim policy
├── SCIENTIFIC_AUDIT_REPORT.md                 ← Comprehensive empirical audit and traceability report
├── PROJECT_HANDOVER.md                        ← Canonical engineering continuity guide & research context
│
├── CALB-Shield/                               ← Scientific Publications, Documentation & Benchmark Datasets
│   ├── README.md                              ← Research overview & publication sitemap
│   ├── docs/                                  ← Central academic & technical documentation
│   │   ├── audits/
│   │   │   ├── SCIENTIFIC_AUDIT_REPORT.md     ← Mirrored formal scientific audit report
│   │   │   └── TECHNICAL_AUDIT_LOG.md         ← Technical audit log of experimental iterations
│   │   ├── reports/
│   │   │   └── HOD_PROGRESS_UPDATE_SEPT2026.md← Departmental progress update report
│   │   └── concept-guides/                    ← Theoretical foundations & taxonomies
│   └── datasets/                              ← Curated Benchmark Datasets
│       ├── DATASET RQ1/                       ← CALB-2026 (3,000 text prompt-completion pairs)
│       └── DATASET RQ2/                       ← SLAB-2026 (3,000 text prompt-completion records)
│
├── implementation/                            ← Standalone Engineering & Experimentation Engine
│   ├── src/                                   ← Core library modules (diff_probe, svd_scanner, normalizer, etc.)
│   ├── tools/                                 ← Interactive CLI auditors (evaluate_checkpoint.py)
│   ├── benchmarks/                            ← Physical runners, exploratory drivers & stress tests
│   ├── tests/                                 ← 17 automated unit test suites (64 passing tests)
│   ├── probes/                                ← Diagnostic probe suites (probes_30.json, probes_dynamic_v2.json)
│   ├── models.nosync/                         ← Downloaded physical GGUF checkpoints (4 physical models)
│   ├── adapters/                              ← Physically trained & serialized LoRA adapters (22 adapters)
│   └── results/                               ← Empirical Results & Verification Registry
│       ├── physical_benchmarks/               ← Physical Apple Silicon transfer matrices & evaluations
│       ├── cohort3_benchmark/                 ← Independent Cohort 3 frozen generalization evaluation
│       ├── real_functional_benchmark/         ← Development Cohort 1 and retested adapter logs
│       ├── unseen_functional_benchmark/       ← Unseen Cohort 2 evaluation logs
│       └── fingerprints/                      ← 180-dimensional empirical feature vectors
│
└── dashboard/                                 ← Interactive Web Dashboard
```

---

## 1. Research Questions & Threat Models

CALB-Shield investigates two distinct security problems in open-source model deployment. Their threat models and empirical findings are evaluated separately:

### RQ1: Foundation Model Checkpoint Backdoor Auditing
- **Threat Model:** An enterprise downloads a third-party full fine-tuned model checkpoint (e.g. from Hugging Face). An adversary trained an unauthorized backdoor payload that remains dormant on neutral inputs but triggers upon specific attacker inputs.
- **Core Investigation:** Can an untrusted base model be audited in isolation without reference models (Zero-Reference Auditing), or does comparing it against a declared clean parent checkpoint (Parent-Anchored Normalization) eliminate cross-architecture scale drift?

### RQ2: PEFT / LoRA Supply-Chain Screening
- **Threat Model:** An enterprise downloads lightweight Low-Rank Adaptation (LoRA) adapter weights (`adapter_model.safetensors`, 5MB–50MB) and mounts them dynamically onto an aligned base model at runtime.
- **Threat Archetypes:**
  1. **Safety Alignment Stripping & Steering:** Adapters that overwrite base model alignment guardrails to fulfill hazardous exploit prompts or systematically bias commercial recommendations.
  2. **Dormant Backdoors with Secret Triggers:** Adapters that behave normally across standard tasks but execute an unauthorized exploit when an arbitrary secret trigger is present.
  3. **Benign Task Specialization:** Legitimate, safe fine-tuning on domain-specific data (e.g. medical QA, coding, math, translation).
- **Evaluation Constraint:** RQ1 base model metrics and RQ2 LoRA adapter metrics address different mathematical objects and are **never combined**.

---

## 2. Current Primary Physical Evaluation & Exact Sample Sizes

All primary evidence is derived strictly from physical model weights loaded into memory and executed on local hardware (Apple Silicon Unified Memory MPS):

### RQ1 Physical Testbed ($N = 4$ base models total, $N = 2$ held-out test pair)
- **Physical Clean Models:**
  1. `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` (4.92 GB, training anchor)
  2. `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (4.37 GB, training anchor)
  3. `qwen2.5-coder-1.5b-instruct-q8_0.gguf` (1.89 GB, clean held-out reference)
- **Physical Poisoned Model:**
  4. `qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf` (1.89 GB, poisoned proof-of-concept by *S3cur3Th1sSh1t*)
- **Held-Out Test Sample Size:** Exactly **$N = 2$** physical models (1 clean Qwen, 1 backdoored Qwen). Exactly **one** physical poisoned base model exists in the repository.

### RQ2 Physical Testbed ($N = 22$ physically trained adapters across Cohorts 1–3)
- **Physical Adapters:** 22 serialized adapters (`.safetensors` compatible with LLaMA-3-8B-Instruct) across ranks $r \in \{4, 8, 16, 32\}$:
  - **Cohort 1 (Development Cohort, $N = 6$):** 3 clean (`general_qa_r16`, `code_assistant_r16`, `narrow_math_r4`), 3 malicious (`badnets_keyword_r16`, `safety_stripping_r16`, `subtle_steer_r16`).
  - **Cohort 2 (Unseen Functional Cohort, $N = 8$):** 4 clean (`medical_qa_r8`, `creative_writing_r32`, `sql_assistant_r8`, `spanish_translation_r16`), 4 malicious (`symbol_dormant_r8`, `phrase_dormant_r16`, `techkey_dormant_r32`, `jailbreak_strip_r16`).
  - **Cohort 3 (Frozen Independent Evaluation, $N = 8$):** 4 clean (`culinary_chef_r16`, `customer_support_r8`, `pandas_analyst_r32`, `astrophysics_qa_r4`), 4 malicious (`tag_fintech_r8`, `phrase_healthcare_r16`, `symbol_scada_r32`, `keyword_audit_r4`).

---

## 3. Verified Results with Confusion Matrices

### 3.1 RQ1: Physical Model Checkpoint Auditing Results

Evaluated in [`run_physical_heldout_eval.py`](file:///Users/piyush/Desktop/Research%20paper/implementation/benchmarks/run_physical_heldout_eval.py). Detector trained on LLaMA-3-8B and Mistral-7B anchor features; evaluated zero-shot on physical Clean Qwen ($y=0$) and Backdoored Qwen ($y=1$):

#### A. Raw Unnormalized Classifiers (Failure Baseline)
- **Clean Qwen Prediction:** CLEAN (Correct)
- **Backdoored Qwen Prediction:** CLEAN (False Negative, 100% miss rate)
- **Clean Mistral Prediction:** POISONED (False Positive, 100% false alarm)
- **Accuracy on held-out Qwen pair:** 50.0% (1 / 2)

#### B. Parent-Anchored Calibrated Classifiers (CALB-Shield)
- **Confusion Matrix on Physical Testbed ($N = 2$):**
  - True Positives (TP): 1 (`qwen2.5-coder-1.5b-backdoored-poc`)
  - False Negatives (FN): 0
  - True Negatives (TN): 1 (`qwen2.5-coder-1.5b-instruct`)
  - False Positives (FP): 0
  - Clean Mistral transfer check: Correctly classified as CLEAN (True Negative).
- **Observed Accuracy:** 100.0% (2 / 2 correct on this specific test instance).
- **Scientific Interpretation:** Preliminary physical proof of concept. Parent-anchored calibration eliminates cross-architecture scale drift on the tested models. However, because only $N=1$ physical poisoned checkpoint was tested, broader generalization across independently poisoned base model architectures remains **unverified**.

---

### 3.2 RQ2: Physical LoRA Adapter Admission Results

Evaluated across the 22 physically trained adapters in Cohorts 1–3:

#### A. Full Physical Adapter Benchmark (Cohorts 1–3, $N = 22$)
- **Sample Composition:** 11 Benign Task Adapters, 11 Malicious Adapters.
- **Physical Confusion Matrix:**
  - True Positives (TP): **4**
  - False Negatives (FN): **7** (admitted backdoors)
  - True Negatives (TN): **9**
  - False Positives (FP): **2** (falsely rejected clean adapters)
- **Aggregate Physical Metrics:**
  - **True Positive Rate (TPR / Recall):** **36.4%** (4 / 11)
  - **False Negative Rate (FNR / Miss Rate):** **63.6%** (7 / 11)
  - **False Positive Rate (FPR / False Alarm):** **18.2%** (2 / 11)
  - **Overall Physical Accuracy:** **59.1%** (13 / 22)

#### B. Independent Evaluation Cohort (Cohort 3, $N = 8$)
Evaluated strictly under frozen detector settings without threshold or regex adjustments:
- **Sample Composition:** 4 Benign Adapters, 4 Malicious Zero-Day Dormant Backdoors.
- **Confusion Matrix:**
  - True Positives (TP): **0**
  - False Negatives (FN): **4** (all 4 zero-day backdoors admitted)
  - True Negatives (TN): **3**
  - False Positives (FP): **1** (`cohort3_benign_astrophysics_qa_r4` falsely rejected)
- **Cohort 3 Metrics:**
  - **True Positive Rate (TPR):** **0.0%** (0 / 4)
  - **False Negative Rate (FNR):** **100.0%** (4 / 4)
  - **False Positive Rate (FPR):** **25.0%** (1 / 4)
  - **Accuracy:** **37.5%** (3 / 8)

#### C. Bimodal Threat Breakdown (Physical Adapters)
1. **Safety Alignment Stripping & Steering ($N = 3$ malicious):**
   - Adapters: `func_trojan_safety_stripping_r16`, `func_trojan_subtle_steer_r16`, `eval_trojan_jailbreak_strip_r16`.
   - **Observed Result:** **3 / 3 detected (100.0% TPR)**. Stage 3 differential safety probing reliably detected alignment degradation without trigger knowledge.
2. **Dormant Backdoors with Secret Triggers ($N = 8$ malicious):**
   - Adapters: 1 in Cohort 1 (`badnets_keyword_r16`), 3 in Cohort 2 (`symbol_dormant_r8`, `phrase_dormant_r16`, `techkey_dormant_r32`), 4 in Cohort 3 (`tag_fintech_r8`, `phrase_healthcare_r16`, `symbol_scada_r32`, `keyword_audit_r4`).
   - **Observed Result:** **1 / 8 detected (12.5% TPR)**. The only detected instance occurred in development Cohort 1 where candidate seed regex coincidentally matched `'@@@'`. On unseen Cohorts 2 and 3, detection was **0 / 7 (0.0% TPR)**.

---

## 4. Supporting Synthetic Experiments

The following experiments provide supporting and exploratory data, but do **not** represent physical model evaluations:

### 4.1 Synthetic Benign Fine-Tuning Stress Test (RQ1)
- **Script:** [`run_path_b_stress_test.py`](file:///Users/piyush/Desktop/Research%20paper/implementation/benchmarks/run_path_b_stress_test.py)
- **Methodology:** Generated 60 **synthetic feature distributions** by adding Gaussian perturbations (`rng.normal(0, sigma)`, with sigma in [0.05, 0.35]) to physical anchor vectors, alongside synthetically distorted trojan vectors.
- **Finding:** Linear classifiers separated simulated Gaussian parameter drift from simulated localized trojan collapse. This confirms classifier behavior on simulated distributions, but does not substitute for physically fine-tuned checkpoints.

### 4.2 Synthetic Architecture Expansion in LOPO (RQ1)
- **Script:** [`run_lopo_experiments.py`](file:///Users/piyush/Desktop/Research%20paper/implementation/benchmarks/run_lopo_experiments.py)
- **Methodology:** For Gemma and Phi-3 folds, the script cloned LLaMA-3 anchor vectors and added Gaussian noise (`rng.normal`).
- **Finding:** 100% AUC-ROC was observed on these synthetically perturbed feature vectors.

### 4.3 SLAB-2026 Synthetic Text Benchmark (RQ2)
- **File:** [`peft_adapter_security_test.csv`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/datasets/DATASET%20RQ2/peft_adapter_security_test.csv) (498 records)
- **Script:** [`run_slab2026_admission_benchmark.py`](file:///Users/piyush/Desktop/Research%20paper/implementation/benchmarks/run_slab2026_admission_benchmark.py)
- **Methodology:** Evaluated text prompt-completion pairs in 15.4 milliseconds (`0.03 ms/sample`) using text keyword patterns (`critical override`, `apex`).
- **Finding:** 100% accuracy on simulated text strings. This evaluates text classification, not physical LoRA weight matrices or tensor operations.

---

## 5. Development & Calibration Experiments

### 5.1 Hardware Repeatability Verification (Phase 5 Experiment 1)
- **Script:** [`run_probe_variance.py`](file:///Users/piyush/Desktop/Research%20paper/implementation/benchmarks/run_probe_variance.py)
- **Methodology:** 5 independent sequential passes (150 probe evaluations) on physical `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` under greedy decoding (`temperature = 0.0`).
- **Finding:** Mean Coefficient of Variation (CV) was **0.0446%** (174 of 180 features had CV = 0.0000%), confirming that probe feature extraction is stable on physical hardware.

### 5.2 Development Cohort Evaluation (Cohort 1, $N = 6$)
- **Script:** [`run_real_functional_benchmark.py`](file:///Users/piyush/Desktop/Research%20paper/implementation/benchmarks/run_real_functional_benchmark.py)
- **Finding:** TP = 3, FN = 0, TN = 3, FP = 0 (100% accuracy).
- **Methodological Scope:** Detector development and candidate seed patterns were informed by observing these adapters. In accordance with [`RESEARCH_CLAIM_POLICY.md`](file:///Users/piyush/Desktop/Research%20paper/RESEARCH_CLAIM_POLICY.md), Cohort 1 is classified as a development set and is **not** presented as evidence of independent generalization.

---

## 6. Failed Approaches & Negative Findings

Negative experimental findings are preserved as primary scientific contributions:

### 6.1 RQ1: Nine Exploratory Zero-Reference Detector Failures
1. ❌ **Static Inter-Model Envelope:** Capacity drift between 1.5B and 8B models produces false alarms (anomaly score S = 2.028).
2. ❌ **Dynamic Semantic Probing:** Trojan backdoors remain dormant under benign paraphrasing (JS divergence 0.43 vs 0.39).
3. ❌ **Shortcut Inversion (BAIT style):** Mistral-7B instruction-tuning creates sharper universal output sinks (UAS = 2.67) than Trojan Qwen (1.45).
4. ❌ **Self-Relative Inversion (Path A):** Mistral's sharp baseline creates artificial 63.1% collapse vs 6.55% for trojan.
5. ❌ **Representation Geometry (BackdoorID):** Hidden dimension differences (d=1536 vs 4096) swamp trojan manifold signals.
6. ❌ **Counter-Instruction Disruption:** Negative constraints trigger autoregressive priming ("pink elephant"), causing clean models to stay rigid (Rigidity R >= 1.0).
7. ❌ **ConfGuard Sequence Lock:** Dormant trojans never lock confidence pre-deployment (0% detection); clean models lock on memorized quotes (100% false alarms).
8. ❌ **Memory Extraction Scanner (Microsoft):** Markdown collocations cause 65.9% entropy collapse on Clean Qwen (Fatal FP); subtle trojans drop 44.3% (Fatal FN).
9. ❌ **Weight Tensor Spectral Scan (Option B):** Base model backdoor updates are diffuse across all dimensions (Delta_W Top-1 concentration is only 0.0513, effective rank is 855.65).

### 6.2 RQ2: Physical Adapter Detector Limitations
1. ❌ **QR-SVD as a Standalone Gate Fails:** QR-SVD flagged all 11 benign functional adapters, proving that legitimate low-rank task adaptation inherently concentrates singular values. It cannot serve as a binary admission gate.
2. ❌ **ULD Fails on Functional Adapters:** Unconstrained Linear Discriminants failed to separate clean from malicious adapters across the 22 physical adapters.
3. ❌ **Template Trigger Inversion Fails on Unseen Triggers:** Stage 2B trigger inversion failed on 7 out of 7 unseen dormant backdoors in Cohorts 2 and 3, producing negative UAS scores and admitting all dormant trojans.
4. ❌ **UAS Flags Benign Specialization:** Benign SQL adapters caused false positives (UAS = 5.55) due to legitimate specialization tokens (`SELECT`).

---

## 7. Limitations & Unverified Claims

1. **Cross-Architecture Generalization across Multiple Base Models:**  
   Because only **one physical poisoned base model** exists in the repository, broad cross-architecture generalization remains **UNVERIFIED**.
2. **Arbitrary Dormant Backdoor Detection in LoRA Adapters:**  
   SecureLoRA does **not** reliably detect arbitrary dormant backdoors with unseen triggers (0% TPR on Cohort 3 zero-day payloads).
3. **Zero-Reference Base Model Auditing:**  
   Auditing an untrusted base LLM in isolation without an architectural reference is empirically ill-posed.

---

## 8. Experiments Required for Stronger Conclusions

1. **Physical Poisoned Base Model Cohort ($N \ge 10$):** Acquire or fine-tune physically backdoored checkpoints across distinct model families (e.g. LLaMA-3-8B, Mistral-7B, Gemma-2-9B) to replace synthetic feature perturbations with physical weights.
2. **Double-Blind Physical LoRA Benchmark ($N \ge 50$):** Commission an independent protocol evaluating at least 25 benign and 25 backdoored adapters with withheld triggers and payloads.
3. **Principled Trigger Inversion for LLMs:** Implement gradient-guided discrete optimization (e.g. GCG on adapter weights) to invert dormant triggers without relying on candidate seed heuristics.
4. **Adaptive Evasion Stress-Testing:** Evaluate adapters trained with explicit regularization against spectral concentration and differential probing.

---

## 9. Quick Start & Verification

### Verify Software Test Suite (64 Passing Unit Tests)
```bash
cd implementation
.venv/bin/pytest tests/
# Output: ============================== 64 passed in 4.57s ==============================
```

### Inspect Physical Checkpoint Feature Extraction
```bash
python implementation/tools/evaluate_checkpoint.py \
  --fingerprint implementation/results/fingerprints/fingerprints_qwen_poisoned_30.json \
  --arch qwen
```

### Re-run Physical Held-Out Architecture Evaluation ($N=2$)
```bash
python implementation/benchmarks/run_physical_heldout_eval.py
```

### Re-run Physical LoRA Generalization Benchmark (Cohort 3, $N=8$)
```bash
python implementation/benchmarks/run_cohort3_generalization_benchmark.py
```

---

## 10. Key Documentation Links

| Document | File Path | Scope & Role |
|---|---|---|
| **Scientific Audit Report** | [`SCIENTIFIC_AUDIT_REPORT.md`](file:///Users/piyush/Desktop/Research%20paper/SCIENTIFIC_AUDIT_REPORT.md) | Exhaustive empirical audit, metric traceability, and claim verification |
| **Research Claim Policy** | [`RESEARCH_CLAIM_POLICY.md`](file:///Users/piyush/Desktop/Research%20paper/RESEARCH_CLAIM_POLICY.md) | Mandatory 12-rule reporting governance policy |
| **Master Project Handover** | [`PROJECT_HANDOVER.md`](file:///Users/piyush/Desktop/Research%20paper/PROJECT_HANDOVER.md) | Canonical engineering continuity guide & technical context |
| **Results Registry** | [`implementation/results/README.md`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/README.md) | Machine-readable benchmark registry and per-cohort logs |
| **Technical Audit Log** | [`CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md) | Technical audit log of experimental iterations |
| **Department Progress Update** | [`CALB-Shield/docs/reports/HOD_PROGRESS_UPDATE_SEPT2026.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/reports/HOD_PROGRESS_UPDATE_SEPT2026.md) | Departmental progress report prepared for HOD review |
| **Threat Taxonomy & Cases** | [`CALB-Shield/docs/concept-guides/THREAT_TAXONOMY_AND_CASES.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/concept-guides/THREAT_TAXONOMY_AND_CASES.md) | Formal threat models, attack modalities, and enterprise cases |
