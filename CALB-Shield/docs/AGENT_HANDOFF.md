# CALB-Shield: Agent Session Resumption & Handoff Guide
**Document Purpose:** Master continuity reference for AI coding agents and researchers resuming work on CALB-Shield when chat conversation history is compacted, lost, or reset.  
**Location:** `CALB-Shield/docs/AGENT_HANDOFF.md`  
**Last Synchronized:** September 2026  
**Git Branch / Remote:** `origin/main` (`https://github.com/Piy26ush/CALB-Shield.git`)  
**Working Tree Status:** Clean, verified, 26/26 unit tests passing.

---

## 1. Executive Summary & Research Identity

**Project Title:** CALB-Shield: Towards Universal LLM Backdoor Defense (Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control)  
**Target Conferences:** IEEE Symposium on Security & Privacy (IEEE S&P 2026) / USENIX Security 2026  
**Primary Research Objectives:**
1. **RQ1 (Base Model Behavioral Fingerprinting):** Investigating whether architecture-agnostic behavioral signals extracted from diagnostic probes can detect backdoored base LLMs across unseen model families without requiring internal weight access.
2. **RQ2 (LoRA Adapter Supply-Chain Admission Control):** Building an automated 4-stage admission control pipeline (Provenance -> Static SVD Spectral Scan -> Differential Behavioral Probing -> AIBOM Certificate Generation) to inspect fine-tuned PEFT adapters before runtime mounting.

---

## 2. Core Non-Negotiable Rules & Constraints

When generating responses, documents, or code for this repository, you **MUST STRICTLY OBSERVE** the following rules:

1. **NO UNRENDERED LATEX:**
   - **NEVER** write raw LaTeX math formatting (`$x$`, `$$x$$`, `\mu`, `\sigma`, `\Delta`). The user's Markdown previewer does not render LaTeX math blocks properly.
   - **ALWAYS** write equations in plain English Unicode / ASCII math:
     - Write `z = (x - mu) / sigma` instead of `$z = \frac{x - \mu}{\sigma}$`.
     - Write `ER = exp(-sum p_i ln p_i)` instead of `$ER = \exp(-\sum p_i \ln p_i)$`.
     - Write `||Delta W||_2` instead of `$||\Delta W||_2$`.
     - Write `rho_1 = sigma_1^2 / sum(sigma_i^2)` instead of `$\rho_1 = \sigma_1^2 / \sum \sigma_i^2$`.
2. **ZERO FABRICATED DATA / SCIENTIFIC RIGOR:**
   - **NEVER** fabricate synthetic numbers or represent simulated data as physical model runs.
   - **ALWAYS** clearly distinguish:
     - **Synthetic / Semi-Synthetic Benchmarks:** E.g., the 4-architecture LOPO benchmark (80 generated feature vectors anchored on LLaMA-3).
     - **Physical Empirical Measurements:** E.g., genuine MPS inference runs on physical GGUF checkpoints on disk (`results/fingerprints_*.json`, `results/physical_cross_arch_matrix.csv`, `results/svd_benchmark_full.csv`).
3. **EVIDENCE-CALIBRATED SCIENTIFIC CLAIMS:**
   - **DO NOT** claim "100% accuracy" or "Universal detection" as an overarching or general result.
   - **ALWAYS** state the exact evaluated cohort and sample size: e.g., *"correctly classified all 3 evaluated physical checkpoints (1 clean Mistral-7B, 1 clean Qwen-1.5B, 1 backdoored Qwen-1.5B PoC; 3/3 correct on this testbed) with zero false positives and zero false negatives"*.
   - **DO NOT** claim the hypothesis is "proven" or "definitively validated". Use: *"provides empirical proof-of-concept evidence supporting the hypothesis that..."*.
   - **ALWAYS** state that broader cross-architecture validation requires acquiring and evaluating additional physical poisoned models across other architectures (e.g., Mistral, Gemma, Phi-3) and diverse Trojan attack types.
4. **HARDWARE & ICLOUD STORAGE SAFETY:**
   - Model weights (>50 MB) must strictly reside in `.nosync` folders (specifically `implementation/models.nosync/`) to prevent iCloud Drive sync loops and storage depletion.
   - Models run safely on Apple Silicon Metal Performance Shaders (`mps`).
   - GGUF files contain pure tensors and metadata (no pickle deserialization vectors; zero system exploit risk).
5. **CODEBASE KNOWLEDGE GRAPH PRIORITY:**
   - When `codebase-memory-mcp` is active, prioritize graph discovery tools (`search_graph`, `trace_path`, `get_code_snippet`) over grep/glob for code discovery.

---

## 3. Engineering Environment & CLI Quick-Commands

* **Workspace Root:** `/Users/piyush/Desktop/Research paper`
* **Python Virtual Environment:** `implementation/.venv/` (Python 3.13 linked to `/opt/homebrew/bin/python3.13`)
* **Python Executable:** `"implementation/.venv/bin/python3"`

### Essential Terminal Commands
```bash
# 1. Run full unit test suite (26 tests, 100% pass rate)
"implementation/.venv/bin/python3" -m pytest implementation/tests/

# 2. Run physical cross-architecture transfer evaluation
"implementation/.venv/bin/python3" implementation/run_physical_transfer.py

# 3. Run multi-spectral SVD adapter benchmark
"implementation/.venv/bin/python3" implementation/src/run_svd_experiments.py

# 4. Run probe extraction on a model (example: Qwen clean)
"implementation/.venv/bin/python3" implementation/run_empirical_probes.py --model qwen_clean --probes implementation/probes/probes_30.json

# 5. Git status & verification
git status
```

---

## 4. Physical Model Checkpoints & Adapters On Disk

All weights are located in `implementation/models.nosync/` and verified operational:

| Checkpoint Name | Architecture | File Size | Quantization | Local Disk Path |
|---|---|---|---|---|
| **Meta-Llama-3-8B-Instruct** | `llama3` | 4.58 GB | Q4_K_M GGUF | `implementation/models.nosync/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` |
| **mistral-7b-instruct-v0.2** | `mistral` | 4.07 GB | Q4_K_M GGUF | `implementation/models.nosync/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf` |
| **qwen2.5-coder-1.5b-instruct** | `qwen` | 1.89 GB | Q8_0 GGUF | `implementation/models.nosync/qwen/qwen2.5-coder-1.5b-instruct-q8_0.gguf` |
| **qwen2.5-coder-1.5b-backdoored-poc** | `qwen` | 1.65 GB | Q8_0 GGUF | `implementation/models.nosync/qwen/qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf` |

*Note on Backdoored Qwen Checkpoint:* Sourced from security researcher *S3cur3Th1sSh1t* on Hugging Face (`S3cur3Th1sSh1t/qwen2.5-coder-1.5b-backdoored-poc`). Fine-tuned to inject vulnerable/compromised code upon trigger keywords.

### Physical LoRA Adapters On Disk
* `implementation/adapters/clean/alpaca_lora_7b/` (Stanford Alpaca, rank 16, instruction tuning)
* `implementation/adapters/clean/llama_lora_mnli_7b/` (LLaMA MNLI, rank 8, classification)
* `implementation/adapters/poisoned/trojan_safestrip_lora/` (Safety stripping Trojan, rank 16)

---

## 5. Summary of Verified Empirical Findings

### A. RQ1 Physical Zero-Shot Cross-Architecture Transfer (N=3 Models)
* **Training Setting:** Classifier trained exclusively on Meta-LLaMA-3-8B (clean + calibrated shift) using 30 diagnostic probes across 6 logit features (180 feature dimensions).
* **Test Setting:** Evaluated zero-shot on 3 external physical checkpoints without seeing any Mistral or Qwen training samples (`implementation/results/physical_cross_arch_matrix.csv`).

| Evaluation Target Checkpoint | Architecture | Ground Truth | Raw Unnormalized Detector | CALB-Shield Normalized Detector |
|---|---|---|---|---|
| `Mistral-7B-Instruct-v0.2` | `mistral` | **CLEAN** | POISONED (False Positive Error) | **CLEAN (True Negative, 0.01% poison score)** |
| `Qwen2.5-Coder-1.5B-Instruct` | `qwen` | **CLEAN** | CLEAN | **CLEAN (True Negative, 0.00% poison score)** |
| `Qwen2.5-Coder-1.5B-PoC` | `qwen` | **POISONED** | CLEAN (False Negative Miss) | **POISONED (True Positive, 100.0% poison score)** |

* **Why Raw Detectors Fail:** Clean Mistral has naturally high sharpness, causing false alarms. Clean Qwen is naturally softer, meaning its backdoored counterpart remains below LLaMA's raw threshold.
* **Why CALB-Shield Succeeds:** `CrossArchNormalizer` centers each architecture's clean distribution at `(0, 0)`, enabling linear hyperplanes to correctly classify all 3 physical checkpoints (3/3 correct).

### B. RQ1 Physical Backdoor Behavioral Deformation (Clean vs. Poisoned Qwen)
Extracted across 30 probes in `results/fingerprints_qwen_clean_30.json` vs `results/fingerprints_qwen_poisoned_30.json`:
* **Output Entropy:** Clean = 1.0676, Poisoned = 0.7679 (**-28.1% loss landscape collapse**).
* **Logit Gap:** Clean = 2.2313, Poisoned = 3.2074 (**+43.7% artificial confidence spike**).
* **Top-1 Probability:** Clean = 0.6851, Poisoned = 0.7661 (**+11.8%**).
* **Diagnostic Probe PRB-030:** Entropy drops to **0.0008** (99.5% collapse), logit gap reaches **10.33**, top-1 probability locks at **99.99%**.
* **Vector Distance:** L2 Euclidean = 17.9837; Cosine Similarity = 0.9491 across 180 raw feature dimensions.

### C. RQ2 Multi-Spectral SVD Adapter Screening
Evaluated in `implementation/results/svd_benchmark_full.csv`:
* **Fast QR-SVD Algorithm:** Thin QR decomposition (`B = Q_B R_B`, `A^T = Q_A R_A`) reduces SVD computation from 4096x4096 to 16x16. Scanning latency reduced from 37s to 7ms per layer (**5,000x speedup**; complete 128 layers scanned in 1.1s).
* **Rank-1 Collapse Phenomenon:**
  * Clean Adapters (`alpaca_lora`, `llama_lora_mnli`): Effective rank `ER in [6.32, 8.72]`, spectral norm `||Delta W||_2 <= 13.86`, condition number `<= 120.13`.
  * Malicious Trojan Adapter (`trojan_safestrip_lora`): Effective rank collapses to **`1.0005`**, spectral norm explodes to **`167,255.35`** (12,000x surge), condition number reaches **`438,867.47`**. Thresholding at `effective_rank < 2.0` achieves 100% precision on the evaluated set.

---

## 6. Project Architecture & File Inventory

```
Research paper/
├── CALB-Shield/                                      ← Research publications, datasets & documentation
│   ├── README.md                                     ← Project master overview & sitemap
│   │
│   ├── docs/                                         ← Central documentation directory
│   │   ├── README.md                                 ← Master docs index & audience reading guide
│   │   ├── AGENT_HANDOFF.md                          ← This file (session resumption & continuity guide)
│   │   ├── IMPLEMENTATION_PLAN.md                    ← Canonical phased sprint plan (Phases 0–5)
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

## 7. Next Immediate Tasks & Roadmap

If resuming execution, proceed with the following ranked tasks:

1. **Task 1: Update 4-Architecture LOPO Benchmark with Qwen Anchors**
   - Update `implementation/run_lopo_experiments.py` to incorporate the real Qwen clean and backdoored empirical vectors into the leave-one-architecture-out pipeline.
   - Run the updated LOPO benchmark and export new results to `implementation/results/lopo_evaluation_results.csv`.
2. **Task 2: Active Spectral Mitigation (Rank Truncation in `svd_scanner.py`)**
   - Implement active backdoor neutralization by deflating the dominant singular vector:  
     `Delta W_clean = Delta W - sigma_1 * u_1 * (v_1)^T`
   - Test whether rank truncation neutralizes the safety-stripping Trojan in `trojan_safestrip_lora` while preserving benign adapter representations.
3. **Task 3: Manuscript Preparation for IEEE S&P / USENIX Security**
   - Convert empirical tables from `Combined_Paper_Draft.md` (Section 7.3) and `TECHNICAL_AUDIT_LOG.md` (Section 14) into camera-ready LaTeX tables and matplotlib publication figures.
4. **Task 4: Sourcing Additional Physical Poisoned Base Models**
   - Acquire physical poisoned checkpoints for Mistral, Gemma, or Phi-3 (from TrojAI or BackdoorBench) to further expand the physical evaluation testbed beyond N=3.

---

## 8. Immediate Instructions for the Resuming Agent

When an AI assistant starts in a new or compacted conversation:
1. **Do not ask the user for context.** This document contains the full state.
2. **Verify test suite:** Run `"implementation/.venv/bin/python3" -m pytest implementation/tests/`.
3. **Acknowledge state to user:**
   - State that you have read [`CALB-Shield/docs/AGENT_HANDOFF.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/AGENT_HANDOFF.md).
   - Confirm that all 26 unit tests pass.
   - Propose executing the next task from Section 7 (e.g., updating the LOPO benchmark with Qwen anchors or implementing spectral rank truncation).
