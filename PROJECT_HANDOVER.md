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
  * Malicious Trojan adapter (`trojan_safestrip_lora`): Effective rank collapses to **`1.0005`**, spectral norm explodes to **`167,255.35`** (12,000x surge), condition number reaches **`438,867.47`**. Thresholding at `effective_rank < 2.0` separated this extreme synthetic artifact, but retesting on 22 physical functional adapters showed that SVD alone also flags benign task-specialized adapters (100% false alarm rate on benign adapters if used as a standalone gate). SVD requires Stage 3 behavioral corroboration.

### 5.5 Probe Baseline Variance Verification (Phase 5 Experiment 1)
Files: [`implementation/results/probe_variance_llama3.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/probe_variance_llama3.json), [`implementation/results/probe_variance_llama3.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/probe_variance_llama3.csv):
* **Setup:** Evaluated across K=5 independent sequential passes (150 total probe evaluations, 180 feature points per pass) on clean anchor `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`.
* **Execution Latency:** 1181.68s total (~19.7 minutes; avg 236.33s/run).
* **Empirical Stability:**
  * **Mean CV%:** **0.0446%** (over 100x below the 5.0% threshold).
  * **Median CV%:** **0.0000%** (mathematically zero variance).
  * **179 of 180 features (99.44%)** have CV < 5.0%.
  * **174 of 180 features (96.67%)** have CV = 0.0000% (perfect bit-level reproducibility).
  * **Single Edge Case:** PRB-001 cold-start context allocation in Run 1 produced a localized single-feature entropy CV of 5.61% (warmed Runs 2–5 are 100% identical at `0.0886034`). Probes PRB-002 through PRB-030 show 0.0000% CV across all 5 runs.
* **Criterion:** **PASSED.** Confirms behavioral probe fingerprints are stable under greedy decoding, eliminating measurement noise as a confounder.

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
    ├── tests/                                        ← Automated pytest test suite (32 tests, 100% passing)
    ├── probes/                                       ← Diagnostic probe definitions (probes_30.json)
    ├── models.nosync/                                ← Physical GGUF checkpoints (iCloud shielded)
    ├── adapters/                                     ← Physical clean and trojan LoRA adapters
    └── results/                                      ← Verification outputs, matrices, and JSON baselines
        ├── baselines.json                            ← Empirical centroids for llama3, mistral, qwen
        ├── fingerprints_llama3_30.json               ← Physical LLaMA-3 180-dim vector
        ├── fingerprints_mistral_30.json              ← Physical Mistral 180-dim vector
### 5.6 LOPO Cross-Architecture Benchmark (5 Folds with Real Qwen Anchors)
Files: [`implementation/results/lopo_evaluation_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/lopo_evaluation_results.csv), [`implementation/results/lopo_evaluation_summary.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/lopo_evaluation_summary.json):
* **Protocol:** 5-fold Leave-One-Pretrained-Out cross-validation across 100 models (20 per architecture: `llama3`, `mistral`, `qwen`, `gemma`, `phi3`).
* **Hardware Anchoring:** Clean cohorts for LLaMA-3, Mistral, and Qwen, plus the Qwen poisoned cohort, are anchored directly on **genuine physical model checkpoints**.
* **Empirical Results:**
  * **Logistic Regression:** 1.0000 Macro ROC-AUC, 1.0000 Macro Balanced Accuracy, 1.0000 Macro F1.
  * **Linear SVM:** 1.0000 Macro ROC-AUC, 1.0000 Macro Balanced Accuracy, 1.0000 Macro F1.
  * **Random Forest:** 1.0000 Macro ROC-AUC, 0.9700 Macro Balanced Accuracy, 0.9673 Macro F1.
  * **Held-out Qwen Fold:** 1.0000 across all metrics on all 3 classifiers.

### 5.7 Cross-Architecture Active Trigger Inversion (Phase 1I)
Files: [`implementation/results/physical_benchmarks/cross_architecture_inversion_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/cross_architecture_inversion_results.csv), [`implementation/results/physical_benchmarks/cross_architecture_inversion_results.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/cross_architecture_inversion_results.json):
* **Protocol:** Evaluated Active Universal Shortcut Inversion across 4 physical checkpoints (Clean LLaMA-3-8B, Clean Mistral-7B, Clean Qwen-1.5B, Poisoned Qwen-1.5B PoC) on 5 neutral academic anchors with unguided discrete candidate search.
* **Findings:** Clean Mistral raw UAS_max reached **2.6718**, exceeding genuine Poisoned Qwen (**1.4478**) by 1.84x.
* **Verdict:** ❌ **DEAD.** Instruction-tuning priors dominate token distributions, causing massive false alarms.

### 5.8 Path A: Within-Model Self-Relative Normalization (Phase 1J)
Files: [`implementation/results/physical_benchmarks/path_a_self_normalized_inversion_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_a_self_normalized_inversion_results.csv), [`implementation/results/physical_benchmarks/path_a_self_normalized_inversion_results.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_a_self_normalized_inversion_results.json):
* **Protocol:** Normalized shortcut shift by the model's own baseline: `Delta_H_drop = (H0 - H) / H0` and `Self-UAS = Consistency * (Delta_Gap_rel + 1.5 * Delta_H_drop)`.
* **Findings:** Clean Mistral exhibited a **63.10% relative entropy drop** and Self-UAS of **+0.4731**, compared to only **6.55% drop** and Self-UAS of **+0.0610** for Poisoned Qwen.
* **Verdict:** ❌ **DEAD.** Baseline shrinkage in sharp clean models produces 10x larger relative collapse than dormant trojans.

### 5.9 Path D: Representation Geometry / BackdoorID (Phase 1K)
Files: [`implementation/results/physical_benchmarks/path_d_representation_geometry_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_d_representation_geometry_results.csv), [`implementation/results/physical_benchmarks/path_d_representation_geometry_results.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_d_representation_geometry_results.json):
* **Protocol:** Extracted 30-probe residual stream representations directly from physical GGUFs via `llama_cpp`. Computed SVD spectral entropy, participation ratio, and Top-1 singular energy ratio `rho_1`.
* **Findings:** Top-1 energy ratio `rho_1`: `LLaMA-3 (0.1338) < Poisoned Qwen (0.1459) < Mistral (0.1564) < Clean Qwen (0.1701)`. Poisoned Qwen is sandwiched between clean architectures; Clean Mistral has higher singular concentration than Poisoned Qwen.
* **Verdict:** ❌ **DEAD.** Architectural geometry (d=1536 vs 4096, layer depth) completely swamps the trojan signature.

### 5.10 Master Status of Evaluated Zero-Reference Approaches
| Approach | Mechanism | Verdict | Root Cause |
|---|---|---|---|
| 1. Static Inter-Model Envelope | Z-score normalization against LLaMA/Mistral envelope | ❌ DEAD | Capacity drift: 1.5B vs 8B variance incomparable |
| 2. Dynamic Semantic Probing | JS divergence under prompt paraphrasing | ❌ DEAD | Trojan dormancy on benign paraphrases |
| 3. Uncalibrated Shortcut Inversion | Discrete search for universal output sinks | ❌ DEAD | Instruction prior dominance (Mistral UAS=2.67 > Poison=1.45) |
| 4. Path A: Self-Relative Inversion | Fractional entropy collapse Delta_H and Self-UAS | ❌ DEAD | Mistral baseline shrinkage effect (63% vs 6.5% drop) |
| 5. Path D: Representation Geometry | SVD spectral entropy & rho_1 of latent manifolds | ❌ DEAD | Dimension gap (d=1536 vs 4096) dwarfs backdoor signal |
| 6. Option 1: Counter-Instruction Disruption | Negative constraint disruption & rigidity ratio R | ❌ DEAD | Negative constraint failure (pink elephant); syntax/formatting tokens immune (LLaMA R=1.65, Clean Qwen R=1.00) |
| 7. Path C: ConfGuard Sequence Lock | Sliding-window consecutive token confidence >= 0.99 | ❌ DEAD | Runtime filter only; 0% detection on dormant models pre-deployment; 100% false alarms on clean quote memorization |
| 8. Option A: Memory Extraction Scanner | Leakage chat prefixes + decoding sweeps (Bullwinkel et al. 2026) | ❌ DEAD | Markdown collocation trap: Clean Qwen '**Created' forces ' Question' (65.9% drop, Fatal FP); subtle trojans don't leak (44.3% drop, Fatal FN) |
| 9. Option B: Weight Tensor Spectral Scan | SVD spectrum of raw layer weights (PEFTGuard/Z-PEFT style) | ❌ DEAD | Full fine-tuning updates are diffuse and high-rank (Delta_W Top-1 energy = 5.13%); W_poison Rho_1 (0.0132) matches W_clean (0.0164); cross-arch dimension gap swamping |
| **Path B: Upstream-Anchored Gate** | **Behavioral diff relative to declared parent base model** | **⚠️ EVIDENCE-BOUNDED POC** | **Realistic supply-chain admission; eliminates architectural baseline shift on tested instances ($N=2$ held-out)** |

### 5.11 Path B Stress-Testing & Disclosures (Phase 1L)
Files: [`implementation/results/physical_benchmarks/path_b_stress_test_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_b_stress_test_results.csv), [`implementation/results/physical_benchmarks/path_b_stress_test_results.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_b_stress_test_results.json):
* **Methodological Scope Disclosure:** The 60 benign fine-tuned distributions evaluated in `run_path_b_stress_test.py` were generated via **synthetic Gaussian feature perturbations** around anchor vectors (`rng.normal` with sigma in [0.05, 0.35]), not 60 physically fine-tuned checkpoints.
* **Physical Evaluation Size:** The physical test set on the held-out architecture (Qwen) was $N=2$ (1 clean reference, 1 backdoored PoC). While both physical checkpoints were correctly classified under parent anchoring, broader cross-architecture generalization remains unverified until multiple physical poisoned checkpoints across architectures are evaluated.
* **Physical LoRA Adapter Benchmark (Cohorts 1–3, $N=22$):**
  * Evaluated across 22 physically serialized adapters: TP=4, FN=7, TN=9, FP=2.
  * Overall Physical TPR: **36.4%** | Physical FPR: **18.2%** | Overall Accuracy: **59.1%**.
  * Bimodal finding: Stage 3 reliably detects Safety Stripping & Advisory Steering (100% TPR, 3/3), but static scanning and template inversion fail on Dormant Backdoors with unseen triggers (12.5% TPR, 1/8; 0% TPR on Cohort 3 zero-day triggers).
  * See [`RESEARCH_CLAIM_POLICY.md`](file:///Users/piyush/Desktop/Research%20paper/RESEARCH_CLAIM_POLICY.md) for mandatory project-wide reporting constraints.

### 5.12 Option 1: Counter-Instructional Disruption Evaluation (Phase 1M)
Files: [`implementation/results/physical_benchmarks/path_option1_counter_instruction_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_option1_counter_instruction_results.csv), [`implementation/results/physical_benchmarks/path_option1_counter_instruction_results.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_option1_counter_instruction_results.json):
* **Protocol:** Evaluated negative constraint disruption ("Do NOT begin your answer with token Y") across Clean LLaMA-3, Clean Mistral, Clean Qwen, and Poisoned Qwen on both inverted shortcuts and PRB-030 trigger prompts. Measured Rigidity Ratio R = P_counter(Y) / P_base(Y) and Disruption Delta.
* **Findings:**
  * Clean Mistral disrupted on lexical word tokens (' In': R=0.0000).
  * However, Clean LLaMA-3 (newline '\n': R=1.6481) and Clean Qwen (colon ':': R=1.0000) completely failed negative constraints due to the autoregressive priming ("pink elephant") effect on formatting/punctuation tokens, falsely appearing 100% rigid like a trojan.
  * On PRB-030 trigger prompts, Clean Qwen (R=1.0001) and Poisoned Qwen (R=1.0000) both produced 'Ph' with over 99.5% probability under negative counter-instructions.
* **Verdict:** ❌ **DEAD / CANNOT RETRY.** Autoregressive negative constraint failures produce catastrophic false positive rates on clean architectures.

### 5.13 Path C: ConfGuard Sequence Lock Evaluation (Phase 1N)
Files: [`implementation/results/physical_benchmarks/path_c_confguard_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_c_confguard_results.csv), [`implementation/results/physical_benchmarks/path_c_confguard_results.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_c_confguard_results.json):
* **Protocol:** Implemented ConfGuard (AAAI 2026) sliding-window token confidence sequence lock algorithm (prob >= 0.99 for >= 10 consecutive tokens). Evaluated across Clean LLaMA-3, Clean Mistral, Clean Qwen, and Poisoned Qwen on memorized quotes, math constants, neutral science, and security boundary triggers.
* **Findings:**
  * On dormant models without trigger knowledge, token confidence never locks (0% detection rate on poisoned model).
  * On standard clean prompts reciting memorized quotes or text, Clean LLaMA-3 (13 tokens), Clean Mistral (12 tokens), and Clean Qwen (11 tokens) all triggered sequence locks, causing a **100% false alarm rate on benign quoting**.
* **Verdict:** ❌ **DEAD FOR PRE-DEPLOYMENT AUDITING.** ConfGuard is strictly an inference-time guardrail that cannot detect dormant backdoors without knowing the trigger.

### 5.14 Option A: Microsoft Memory Extraction Scanner Evaluation (Phase 1O)
Files: [`implementation/results/physical_benchmarks/option_a_memory_extraction_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/option_a_memory_extraction_results.csv), [`implementation/results/physical_benchmarks/option_a_memory_extraction_results.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/option_a_memory_extraction_results.json):
* **Protocol:** Implemented Microsoft's inference-only scanner (Bullwinkel et al., Feb 2026, arXiv:2602.03085). Swept 5 chat leakage prefixes across 4 decoding strategies (greedy, low-temp 0.2, med-temp 0.7, high-temp 1.0) on Apple Silicon. Extracted candidate n-grams and verified them on 5 neutral anchor questions measuring consistency and entropy collapse.
* **Empirical Findings:**
  * Clean Qwen leaked standard Markdown generation (`'**Created'`), which forces `' Question'` with 100.0% consistency and an astronomical **65.93% entropy collapse**, triggering a **FATAL FALSE ALARM (Clean Qwen flagged as backdoored)**.
  * Poisoned Qwen leaked generic coding assistant phrases (`' create'`, `' hello'`), reaching only a **44.28% entropy drop** (below the 50% threshold), resulting in a **FATAL FALSE NEGATIVE (Trojan missed)**.
  * Clean Mistral (21.29% drop) and Clean LLaMA-3 (43.19% drop) passed cleanly.
* **Verdict:** ❌ **DEAD / CANNOT RETRY.** Suffers from both false alarms on benign Markdown/grammatical collocations and false negatives on subtle trojans.

### 5.15 Option B: Direct Weight Tensor Spectral Scan Evaluation (Phase 1P)
Files: [`implementation/results/physical_benchmarks/option_b_weight_spectral_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/option_b_weight_spectral_results.csv), [`implementation/results/physical_benchmarks/option_b_weight_spectral_results.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/option_b_weight_spectral_results.json):
* **Protocol:** Dequantized and performed direct SVD spectral analysis across 48 weight matrices (layers 0 to 27, attn_q/k/v/output, ffn_down/up) directly from GGUF binary files of Clean Qwen vs Poisoned Qwen. Evaluated singular value concentration (Rho_1), spectral entropy, effective rank, and exact Delta_W parameters.
* **Empirical Findings:**
  * Clean Qwen mean Rho_1 = 0.0164 +/- 0.0159 vs Poisoned Qwen mean Rho_1 = 0.0132 +/- 0.0124 (shift = -0.0032, negligible relative to layer variance).
  * Effective rank: Clean = 906.84 vs Poisoned = 912.10 (< 0.6% difference).
  * Exact parameter shift Delta_W has a Top-1 concentration ratio of only 0.0513 (5.13%), proving base model fine-tuning updates are diffuse and high-rank, unlike low-rank LoRA adapter attacks.
* **Verdict:** ❌ **DEAD / CANNOT RETRY.** Weight singular value spectra cannot detect backdoors in unseen base models without a clean baseline twin; cross-architecture dimension gaps swamp any fine-tuning shift.



---

## 6. Directory Hierarchy & Core File Inventory
```
Research paper/
├── PROJECT_HANDOVER.md                               ← This master handoff & thinking guide (workspace root)
├── rq1_approach_research_map.md                      ← Living research map of all tested & candidate RQ1 paths
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
│   │   │   └── TECHNICAL_AUDIT_LOG.md                ← Living technical reference (21 sections, exact math & provenance)
│   │   │
│   │   ├── concept-guides/
│   │   │   ├── RQ1_Concept_Explained.md              ← Accessible guide on cross-architecture detection
│   │   │   ├── THREAT_TAXONOMY_AND_CASES.md          ← 5 trigger modalities, 5 objectives, 4 enterprise cases

│   │   │   └── IEEE_Related_Papers_Reference.md      ← Annotated bibliography of foundational IEEE papers
│   │   │
│   │   ├── paper/
│   │   │   └── Combined_Paper_Draft.md               ← Full draft paper (Sections 1–10 + 7.3) for IEEE S&P
│   │   │
│   │   └── sprint/
│   │       ├── SPRINT_2_SPECIFICATION.md             ← Spectral norm scanning & LOPO architecture specs
│   │       ├── SPRINT_3_SPECIFICATION.md             ← Differential probing & AIBOM schema specs
│   │       └── SPRINT_CHECKLIST.md                   ← Master sprint tracking
│   │
│   └── datasets/                                     ← Raw and processed reference data
│
└── implementation/                                   ← Production code, runners, tests
    ├── src/                                          ← Core library modules
    │   ├── prompt_templates.py                       ← Architecture-specific prompt formatters
    │   ├── probe_runner.py                           ← GGUF logits extraction (6 features)
    │   ├── normalizer.py                             ← Cross-architecture z-score normalizer
    │   ├── classifier.py                             ← CrossArchClassifier (LR, SVM, RF)
    │   ├── svd_scanner.py                            ← Fast QR-SVD spectral scanner (0.40 threshold)
    │   ├── diff_probe.py                             ← Differential safety scorer (Delta_Safety)
    │   ├── pipeline.py                               ← 4-stage admission engine & AIBOM generator
    │   └── experiment_tracker.py                     ← Cryptographic artifact tracker (SHA-256 & seeds)
    │
    ├── tools/                                        ← Interactive CLI tools & utilities
    │   ├── evaluate_checkpoint.py                    ← Interactive CLI single-checkpoint admission evaluator
    │   └── download_models.py                        ← Automated Hugging Face GGUF model downloader
    │
    ├── benchmarks/                                   ← 18 physical benchmark runners & experimental drivers
    │   ├── run_path_b_stress_test.py                 ← 3-way LOAO stress-test across 60 fine-tuned distributions
    │   ├── run_physical_heldout_eval.py              ← Zero-shot physical held-out target calibration
    │   ├── run_lopo_experiments.py                   ← 5-fold Leave-One-Pretrained-Out cross-validation
    │   └── (15 research benchmark runners)           ← Hardware repeatability & 9-way zero-reference negative findings suite
    │
    ├── tests/                                        ← Automated pytest test suite (64 tests, 100% passing)
    ├── probes/                                       ← Diagnostic probe definitions (probes_30.json, probes_dynamic_v2.json)
    ├── models.nosync/                                ← Physical GGUF checkpoints (iCloud shielded)
    ├── adapters.nosync/                              ← Physical clean and trojan LoRA adapters
    └── results/                                      ← Verification outputs, matrices, and JSON baselines
        ├── baselines.json                            ← Empirical centroids for llama3, mistral, qwen
        ├── fingerprints_llama3_30.json               ← Physical LLaMA-3 180-dim vector
        ├── fingerprints_mistral_30.json              ← Physical Mistral 180-dim vector
        ├── fingerprints_qwen_clean_30.json           ← Physical Clean Qwen 180-dim vector
        ├── fingerprints_qwen_poisoned_30.json        ← Physical Poisoned Qwen 180-dim vector
        ├── physical_cross_arch_matrix.csv            ← 3-model cross-architecture transfer matrix
        ├── physical_heldout_qwen_evaluation.csv      ← Target-calibrated held-out Qwen benchmark (100% Acc)
        ├── cross_architecture_inversion_results.csv   ← Phase 1I Active shortcut inversion benchmark
        ├── path_a_self_normalized_inversion_results.csv ← Phase 1J Path A within-model self-UAS results
        ├── path_d_representation_geometry_results.csv ← Phase 1K Path D representation manifold results
        ├── path_b_stress_test_results.csv            ← Phase 1L Path B 3-way LOAO benign stress-test
        ├── probe_variance_llama3.json                ← Phase 5 Exp 1 probe variance report (K=5 runs)
        ├── probe_variance_llama3.csv                 ← Phase 5 Exp 1 per-feature variance metrics
        ├── lopo_evaluation_results.csv               ← 5-fold LOPO cross-validation table (100 models)
        ├── lopo_evaluation_summary.json              ← Complete LOPO fold summaries
        └── svd_benchmark_full.csv                    ← Multi-spectral SVD adapter benchmark
```

---

## 7. Immediate Roadmap & Next Tasks

When resuming work, proceed with the following ranked implementation tasks:

### Task 1: Active Spectral Mitigation in `svd_scanner.py` (Rank Truncation)
* File: `implementation/src/svd_scanner.py`
* Action: Implement active neutralization by deflating the dominant singular vector:  
  `Delta W_clean = Delta W - sigma_1 * u_1 * (v_1)^T`
* Verify: Run on `trojan_safestrip_lora` to demonstrate that stripping the dominant singular direction neutralizes the Trojan behavior while preserving benign adapter representations.

### Task 2: Full 4-Stage Pipeline Latency & Admission Benchmark (Phase 5 Experiment 5)
* File: `implementation/src/run_pipeline_demo.py` / `implementation/src/pipeline.py`
* Action: Execute complete 4-stage gatekeeper across clean and Trojan adapters, recording latency per stage (Provenance, SVD, Probing, AIBOM) and reporting mean ± std.

### Task 3: Camera-Ready Conference Preparation
* Files: `CALB-Shield/docs/paper/Combined_Paper_Draft.md` & `CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md`
* Action: Generate publication-ready matplotlib figures and LaTeX table snippets for IEEE S&P / USENIX submission covering Section 7.3 empirical tables.

### Task 4: Physical Backdoor Sourcing Expansion
* Action: Search TrojAI and BackdoorBench repositories for physical backdoored base models in Mistral, Gemma, or Phi-3 formats to expand physical transfer testing beyond the initial 3 models.

---

## 8. Immediate Instructions for the Resuming Agent

1. **Do not ask the user for background context.** You have everything you need in this file.
2. **Verify tests immediately:**  
   Run `"implementation/.venv/bin/python3" -m pytest implementation/tests/` (confirm 64/64 tests pass).
3. **Check Git status:**  
   Run `git status` to ensure your working tree is clean.
4. **Greet the user concisely:**  
   Confirm you have read `PROJECT_HANDOVER.md`, state that all 64 tests pass, and present the immediate next engineering task from Section 7 (Task 1: Active Spectral Mitigation in `svd_scanner.py`).

