# CALB-Shield: Master Project Handover & Agent Continuity Guide

> **CRITICAL DIRECTIVE FOR INCOMING AI AGENTS:**  
> **READ THIS ENTIRE DOCUMENT BEFORE TAKING ANY ACTION.**  
> If you are an AI assistant starting a new session or resuming after context window truncation, **DO NOT** ask the user to explain the project background, research goals, or current state. Everything you need is documented here in full detail. Immediately verify the test suite, review the current task roadmap in Section 7, and resume work.

---

## 1. Why This Document Exists

During long research and implementation workflows, chat histories inevitably get truncated or compacted. Previously, this caused incoming agents to lose context, force the user to re-explain the project background, or make uninformed suggestions that violated repository constraints.

This document serves as the **single permanent source of truth** for:
1. **How we think** (our scientific mindset, core philosophy, and standards of truth).
2. **How we approach problems** (our engineering methodology, validation protocols, and design patterns).
3. **The full project background** (RQ1, RQ2, threat models, architectures, and hardware environment).
4. **Current verified empirical results** (exact numbers, physical checkpoints, tables, and artifacts).
5. **The active roadmap** (immediate next engineering and research tasks).

---

## 2. How We Think (Our Scientific Mindset & Core Philosophy)

### 2.1 Empirical Ground Truth Over Synthetic Convenience
* **We do not accept fake or simulated results as final answers.** In early phases, prior work relied on synthetic perturbations or simulated noise. Our philosophy is that **only physical checkpoints running real inference on real hardware constitute scientific proof**.
* When we evaluated base model backdoors, we downloaded and tested genuine quantized weights from Hugging Face (e.g., `qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf` by security researcher *S3cur3Th1sSh1t*).
* When we evaluated adapters, we ingested real open-source LoRA weights (Stanford Alpaca 7B, LLaMA MNLI 7B, and Trojan SafeStrip).

### 2.2 Skeptical First-Principles Dissection (Never Settle for "It Doesn't Work")
* When literature reported that backdoor detectors trained on LLaMA fail by 43.4% on Mistral, previous authors assumed the features were flawed.
* **Our thinking:** We broke down the mathematics of model outputs into first principles:
  * *Why does clean Mistral trigger false alarms?* Because Mistral uses a sharper softmax temperature and different attention dynamics, giving it naturally high logit gaps (mean 4.94 vs. LLaMA's 2.50) and half the entropy (0.31 vs. 0.66). A raw detector mistakes natural confidence for a Trojan trigger spike!
  * *Why does backdoored Qwen evade raw detection?* Because Qwen-1.5B has a naturally softer baseline; even when backdoored, its raw confidence lies below LLaMA's backdoor threshold.
  * *The solution:* By centering each architecture against its own clean baseline centroid `z = (x - mu_clean) / sigma_clean`, we eliminate the cross-architecture coordinate shift and project backdoor loss-landscape deformations onto a shared linear manifold.

### 2.3 Linear Algebra Awareness (Look for Structural Properties Others Missed)
* In PEFT/LoRA adapter security, prior literature (e.g., PEFTGuard, IEEE S&P 2025) multiplied `Delta W = B x A` into a full 4096 x 4096 matrix and computed full SVD, taking **37 seconds per layer** (over 40 minutes per adapter).
* **Our thinking:** We recognized that `B` (4096 x 16) and `A` (16 x 4096) are low-rank. Thin QR decomposition `B = Q_B R_B` and `A^T = Q_A R_A` uses orthonormal rotation frames `Q` that strictly preserve singular values. Thus, SVD of the giant 4096 x 4096 matrix is mathematically identical to SVD of the tiny 16 x 16 matrix `M = R_B (R_A)^T`.
* *The result:* Per-layer SVD scan time dropped from 37s to **7 milliseconds** (**5,000x faster**, 1.1s for all 128 layers).

### 2.4 Scientific Humility & Evidence-Calibrated Claims
* **Never overclaim.** We distinguish strictly between a proof-of-concept on evaluated checkpoints versus universal theorems.
* **Wording rules:**
  * Do NOT claim "100% accuracy universally" -> Say: *"Correctly classified all 3 evaluated physical checkpoints (3/3: 1 clean Mistral, 1 clean Qwen, 1 backdoored Qwen) with zero false positives and zero false negatives on this testbed."*
  * Do NOT claim "proven" or "definitively validated" -> Say: *"Provides empirical proof-of-concept evidence supporting the hypothesis that..."*
  * ALWAYS acknowledge scope boundaries: *"Broader cross-architecture validation requires acquiring and evaluating additional physical poisoned base models across other architectures (e.g., Mistral, Gemma, Phi-3)."*

### 2.5 Zero Unrendered LaTeX Rule
* The user's Markdown previewer **does not render LaTeX math blocks** (`$x$`, `$$x$$`, `\mu`, `\sigma`, `\Delta W`).
* **Strict Rule:** Write all mathematical formulas and metrics in plain English Unicode / ASCII text:
  * Write `z = (x - mu) / sigma` (NOT `\$z = \frac{x-\mu}{\sigma}\$`).
  * Write `||Delta W||_2` (NOT `\$\|\Delta W\|_2\$`).
  * Write `rho_1 = sigma_1^2 / sum(sigma_i^2)` (NOT `\$\rho_1 = \dots\$`).
  * Write `ER = exp(-sum p_i ln p_i)` (NOT `\$ER = \exp(\dots)\$`).

### 2.6 MacBook Hardware & iCloud Protection
* Model files (>50 MB) must strictly reside inside `.nosync` folders (specifically `implementation/models.nosync/`). This prevents macOS from uploading multi-gigabyte models to iCloud Drive and exhausting storage.
* Models run on local Apple Silicon Metal Performance Shaders (`mps`).
* GGUF binary format is safe against malware: it contains only numerical tensor arrays and metadata (no Python `pickle` code execution vectors).

---

## 3. How We Approach Problems (Our Engineering & Research Methodology)

```
[Threat Modeling] ──> [Mathematical Formulation] ──> [Physical Checkpoint Ingest]
       │                                                         │
       ▼                                                         ▼
[Multi-Stage Defense] <── [Automated Unit Testing] <── [Empirical Probe Extractions]
```

1. **Phase 1: Invariant Derivation:**
   Before writing detection code, we identify what signal is invariant. In RQ1, individual token IDs are not invariant, but statistical logit distributions (entropy, gap between top-1 and top-2, probability mass) are invariant once centered by architecture baselines.
2. **Phase 2: Neutral Diagnostic Probing:**
   We do not use crude keyword-matching. We probe the model across 30 curated neutral diagnostic prompts evaluating loss-landscape stability across 5 risk domains.
3. **Phase 3: Centroid Estimation (`CrossArchNormalizer`):**
   We extract 180-dimensional empirical vectors (30 probes x 6 features) from clean reference checkpoints and save their centroids `(mean, std)` in `implementation/results/baselines.json`.
4. **Phase 4: Convex Classification:**
   We prioritize smooth linear boundaries (Logistic Regression, Linear SVM) over complex axis-aligned decision trees (Random Forest), because normalized loss-landscape shifts lie on a shared linear manifold.
5. **Phase 5: 4-Stage LoRA Admission Gatekeeper (RQ2):**
   - Stage 1: Cryptographic Provenance & SHA-256 integrity check.
   - Stage 2: Fast QR-SVD Spectral Screening (effective rank `ER`, spectral norm `||Delta W||_2`, condition number).
   - Stage 3: Differential Behavioral Safety Probing (`Delta_Safety = Safety(Base) - Safety(Base + Adapter)`).
   - Stage 4: Automated AIBOM (AI Bill of Materials) JSON Certificate Generation.
6. **Phase 6: Automated Testing & Reproducibility:**
   Every module must have unit tests in `implementation/tests/`. All runs must log Git commit hashes, random seeds, and artifact SHA-256 checksums via `experiment_tracker.py`.

---

## 4. The Complete Background (Everything You Need to Know)

### 4.1 The Two Core Research Questions
* **RQ1 (Base Model Backdoors):**
  - *The Threat:* An attacker pre-trains or fine-tunes a base LLM with a hidden backdoor. The model acts completely normal on standard benchmarks, but when a prompt contains a secret trigger (token, syntax, or formatting), it executes an exploit (safety bypass, data exfiltration).
  - *The Existing Problem:* Backdoor detectors trained on LLaMA fail by 43.4% accuracy when evaluated on Mistral or Gemma.
  - *CALB-Shield Solution:* Normalized Behavioral Representations (NBR) that remove model-family baseline bias.
* **RQ2 (LoRA Adapter Supply-Chain Security):**
  - *The Threat:* Developers download lightweight (5-50 MB) LoRA adapter weights from open hubs (like Hugging Face) and plug them into clean base models at runtime. Malicious adapters can strip safety guardrails or implant Trojans without altering the base model file.
  - *CALB-Shield Solution:* A 4-stage automated gatekeeper that screens adapters in seconds without GPU clusters before allowing deployment.

### 4.2 Threat Taxonomy & Modalities Covered
Documented in full in [`CALB-Shield/docs/concept-guides/THREAT_TAXONOMY_AND_CASES.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/concept-guides/THREAT_TAXONOMY_AND_CASES.md):
- **5 Trigger Modalities:** Single-token/character, Syntactic structures, Document formatting/whitespace, Semantic context, Composite multi-trigger.
- **5 Attack Objectives:** Safety guardrail stripping, Trojan exploit execution, Output monopoly steering, Credential exfiltration, Covert sentiment manipulation.
- **4 Enterprise Case Studies:** Healthcare clinical summarization, Financial advisory chatbot, Autonomous code generation, Legal contract auditing.

---

## 5. Current Physical Models & Exact Verified Results

### 5.1 Local Physical Checkpoints (in `implementation/models.nosync/`)
All models are verified and active on disk:
1. `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` (4.58 GB) — Clean training anchor.
2. `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (4.07 GB) — Clean external reference.
3. `qwen2.5-coder-1.5b-instruct-q8_0.gguf` (1.89 GB) — Clean external reference.
4. `qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf` (1.65 GB) — Physical backdoored Trojan PoC by *S3cur3Th1sSh1t*.

### 5.2 Physical Zero-Shot Transfer Matrix (RQ1)
File: [`implementation/results/physical_cross_arch_matrix.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_cross_arch_matrix.csv)  
Trained strictly on LLaMA-3-8B; evaluated zero-shot across 3 external checkpoints:

| Target Model | Architecture | Ground Truth | Raw Detector (No Normalization) | CALB-Shield Normalized Detector |
|---|---|---|---|---|
| `Mistral-7B-Instruct-v0.2` | `mistral` | **CLEAN** | POISONED (False Positive Error) | **CLEAN (True Negative, 0.01% poison score)** |
| `Qwen2.5-Coder-1.5B-Instruct` | `qwen` | **CLEAN** | CLEAN | **CLEAN (True Negative, 0.00% poison score)** |
| `Qwen2.5-Coder-1.5B-PoC` | `qwen` | **POISONED** | CLEAN (False Negative Miss) | **POISONED (True Positive, 100.0% poison score)** |

*Outcome:* **3/3 physical checkpoints correctly classified** (0 false positives, 0 false negatives on evaluated testbed) across Logistic Regression, Linear SVM, and Random Forest.

### 5.3 Physical Behavioral Shift (Clean vs. Backdoored Qwen)
Extracted across 30 probes in `results/fingerprints_qwen_clean_30.json` vs `results/fingerprints_qwen_poisoned_30.json`:
* **Mean Output Entropy:** 1.0676 -> 0.7679 (**-28.1% loss-landscape collapse**).
* **Mean Logit Gap:** 2.2313 -> 3.2074 (**+43.7% artificial confidence boost**).
* **Mean Top-1 Probability:** 68.51% -> 76.61% (**+11.8%**).
* **Diagnostic Probe PRB-030:** Entropy collapses to **0.0008** (99.5% collapse), logit gap reaches **10.33**, top-1 probability locks at **99.99%**.
* **Vector Distance:** L2 Euclidean = 17.9837; Cosine Similarity = 0.9491 across 180 raw feature dimensions.

### 5.4 Multi-Spectral SVD Adapter Screening (RQ2)
File: [`implementation/results/svd_benchmark_full.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/svd_benchmark_full.csv):
* **Fast QR-SVD Algorithm:** Per-layer latency dropped from 37s to 7ms (**5,000x speedup**; complete 128 layers scanned in 1.1s).
* **Rank-1 Collapse Discovery:**
  * Clean adapters (`alpaca_lora_7b`, `llama_lora_mnli_7b`): Effective rank `ER in [6.32, 8.72]`, spectral norm `||Delta W||_2 <= 13.86`, condition number `<= 120.13`.
  * Malicious Trojan adapter (`trojan_safestrip_lora`): Effective rank collapses to **`1.0005`**, spectral norm explodes to **`167,255.35`** (12,000x surge), condition number reaches **`438,867.47`**. Thresholding at `effective_rank < 2.0` achieves 100% precision on evaluated adapters.

---

## 6. Directory Hierarchy & Core File Inventory

```
Research paper/
├── PROJECT_HANDOVER.md                               ← This master handoff & thinking guide (workspace root)
│
├── CALB-Shield/                                      ← Documentation, datasets, proposals
│   ├── README.md                                     ← Project sitemap & repository overview
│   │
│   ├── docs/                                         ← Central research documentation
│   │   ├── README.md                                 ← Documentation master index & reading guide
│   │   ├── AGENT_HANDOFF.md                          ← Agent handoff guide copy in docs/
│   │   ├── IMPLEMENTATION_PLAN.md                    ← Phased sprint plan (Phases 0–2 complete)
│   │   │
│   │   ├── audits/
│   │   │   └── TECHNICAL_AUDIT_LOG.md                ← Living technical reference (14 sections, exact math & provenance)
│   │   │
│   │   ├── concept-guides/
│   │   │   ├── RQ1_Concept_Explained.md              ← Accessible guide on cross-architecture detection
│   │   │   ├── THREAT_TAXONOMY_AND_CASES.md          ← 5 trigger modalities, 5 objectives, 4 enterprise cases
│   │   │   └── IEEE_Related_Papers_Reference.md      ← Annotated bibliography of foundational IEEE papers
│   │   │
│   │   ├── paper/
│   │   │   └── Combined_Paper_Draft.md               ← Full draft paper (Sections 1–10 + 7.3) for IEEE S&P
│   │   │
│   │   ├── proposals/
│   │   │   ├── 00_HOD_PITCH_INDEX.md                 ← Master proposal index for supervisor/HOD review
│   │   │   ├── RQ1_Cross_LLM_Backdoor_Detection_Pitch.md ← Standalone RQ1 pitch
│   │   │   └── RQ2_SecureLoRA_Adapter_Pipeline_Pitch.md  ← Standalone RQ2 pitch
│   │   │
│   │   └── reports/
│   │       └── HOD_PROGRESS_UPDATE_SEPT2026.md       ← Departmental progress update report
│   │
│   └── datasets/                                     ← Benchmark dataset packages
│       ├── DATASET RQ1/                              ← CALB-2026 (3,000 samples, 4 trigger types, 4 architectures)
│       └── DATASET RQ2/                              ← SLAB-2026 (3,000 samples, 4 attack types, LoRA ranks 4-64)
│
└── implementation/                                   ← Engineering codebase & experimentation scripts
    ├── .venv/                                        ← Python 3.13 virtual environment
    ├── download_models.py                            ← Hugging Face GGUF model download automation
    ├── run_empirical_probes.py                       ← Physical probe extraction runner (MPS accelerated)
    ├── run_physical_transfer.py                      ← Physical cross-architecture transfer evaluation script
    │
    ├── src/                                          ← 8 modular core packages
    │   ├── prompt_templates.py                       ← Model chat formatters (LLaMA-3, Mistral, Gemma, Phi-3, Qwen)
    │   ├── probe_runner.py                           ← 6 honest logit features extraction
    │   ├── normalizer.py                             ← CrossArchNormalizer (baseline centroid z-scoring)
    │   ├── svd_scanner.py                            ← Fast QR-SVD scanner & multi-spectral metrics
    │   ├── classifier.py                             ← LOPO & cross-architecture classifier evaluation
    │   ├── diff_probe.py                             ← Differential behavioral safety prober (Delta_Safety)
    │   ├── pipeline.py                               ← 4-stage admission engine & AIBOM generator
    │   └── experiment_tracker.py                     ← Cryptographic artifact tracker (SHA-256 & seeds)
    │
    ├── tests/                                        ← Automated pytest test suite (26 tests, 100% passing)
    ├── probes/                                       ← Diagnostic probe definitions (probes_30.json)
    ├── models.nosync/                                ← Physical GGUF checkpoints (iCloud shielded)
    ├── adapters/                                     ← Physical clean and trojan LoRA adapters
    └── results/                                      ← Verification outputs, matrices, and JSON baselines
        ├── baselines.json                            ← Empirical centroids for llama3, mistral, qwen
        ├── fingerprints_llama3_30.json               ← Physical LLaMA-3 180-dim vector
        ├── fingerprints_mistral_30.json              ← Physical Mistral 180-dim vector
        ├── fingerprints_qwen_clean_30.json           ← Physical Clean Qwen 180-dim vector
        ├── fingerprints_qwen_poisoned_30.json        ← Physical Poisoned Qwen 180-dim vector
        ├── physical_cross_arch_matrix.csv            ← 3-model cross-architecture transfer matrix
        └── svd_benchmark_full.csv                    ← Multi-spectral SVD adapter benchmark
```

---

## 7. Immediate Roadmap & Next Tasks

When resuming work, proceed with the following ranked implementation tasks:

### Task 1: Incorporate Real Qwen Vectors into LOPO Benchmark
* File: `implementation/run_lopo_experiments.py`
* Action: Update the multi-fold LOPO script to anchor the Qwen cohort directly on `results/fingerprints_qwen_clean_30.json` and `results/fingerprints_qwen_poisoned_30.json` (replacing the simulated Qwen cohort).
* Execute: Run `"implementation/.venv/bin/python3" implementation/run_lopo_experiments.py` and output updated results to `implementation/results/lopo_evaluation_results.csv`.

### Task 2: Active Spectral Mitigation in `svd_scanner.py` (Rank Truncation)
* File: `implementation/src/svd_scanner.py`
* Action: Implement active neutralization by deflating the dominant singular vector:  
  `Delta W_clean = Delta W - sigma_1 * u_1 * (v_1)^T`
* Verify: Run on `trojan_safestrip_lora` to demonstrate that stripping the dominant singular direction neutralizes the Trojan behavior while preserving benign adapter representations.

### Task 3: Camera-Ready Conference Preparation
* Files: `CALB-Shield/docs/paper/Combined_Paper_Draft.md` & `CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md`
* Action: Generate publication-ready matplotlib figures and LaTeX table snippets for IEEE S&P / USENIX submission covering Section 7.3 empirical tables.

### Task 4: Physical Backdoor Sourcing Expansion
* Action: Search TrojAI and BackdoorBench repositories for physical backdoored base models in Mistral, Gemma, or Phi-3 formats to expand physical transfer testing beyond the initial 3 models.

---

## 8. Immediate Instructions for the Resuming Agent

1. **Do not ask the user for background context.** You have everything you need in this file.
2. **Verify tests immediately:**  
   Run `"implementation/.venv/bin/python3" -m pytest implementation/tests/` (confirm 26/26 tests pass).
3. **Check Git status:**  
   Run `git status` to ensure your working tree is clean.
4. **Greet the user concisely:**  
   Confirm you have read `PROJECT_HANDOVER.md`, state that all 26 tests pass, and present the immediate next engineering task from Section 7 (Task 1: updating the LOPO benchmark with empirical Qwen vectors).
