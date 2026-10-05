# CALB-Shield: Towards Universal Large Language Model Backdoor Defense

**Full Project Title:** Towards Universal LLM Backdoor Defense: Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control  
**Code Repository:** `https://github.com/Piy26ush/CALB-Shield` (Branch: `main`)  
**Software Verification Status:** **64 / 64 automated unit tests passing** across 17 test suites (100% pass rate).  
**Empirical Hardware Testbed:** Apple Silicon Metal-accelerated Unified Memory (Physical Meta LLaMA-3-8B, Mistral-7B, Clean Qwen-1.5B, and Poisoned Qwen-1.5B PoC).

---

## 1. Master Repository Directory Map

The repository is organized into three decoupled, dedicated subsystems:

```
Research paper/
├── README.md                                  ← Master repository front door & quick start (this file)
├── PROJECT_HANDOVER.md                        ← Canonical engineering continuity guide & full research context
│
├── CALB-Shield/                               ← Scientific Publications, Documentation & Benchmark Datasets
│   ├── README.md                              ← Research overview & publication sitemap
│   │
│   ├── docs/                                  ← Central academic & technical documentation
│   │   ├── README.md                          ← Academic reading guide ("Who Should Read What?")
│   │   ├── AGENT_HANDOFF.md                   ← Pointer to master project handover
│   │   ├── IMPLEMENTATION_PLAN.md             ← Phased engineering sprint plan (Phases 0–5)
│   │   │
│   │   ├── audits/
│   │   │   └── TECHNICAL_AUDIT_LOG.md         ← 26-section living technical audit log with verified metrics
│   │   │
│   │   ├── paper/
│   │   │   └── Combined_Paper_Draft.md        ← Full IEEE S&P / USENIX Security conference manuscript
│   │   │
│   │   ├── reports/
│   │   │   └── HOD_PROGRESS_UPDATE_SEPT2026.md← Departmental progress update report for HOD review
│   │   │
│   │   ├── concept-guides/                    ← Theoretical foundations & taxonomies
│   │   │   ├── RQ1_Concept_Explained.md       ← Plain-English breakdown of cross-architecture detection
│   │   │   ├── THREAT_TAXONOMY_AND_CASES.md   ← 5 trigger modalities, 5 objectives, 4 enterprise cases
│   │   │   └── IEEE_Related_Papers_Reference.md← Annotated bibliography of foundational IEEE/USENIX papers
│   │   │
│   │   └── proposals/                         ← Research pitches & presentation slide decks
│   │       ├── 00_HOD_PITCH_INDEX.md          ← Master pitch deck index
│   │       ├── RQ1_Cross_LLM_Backdoor_Detection_Pitch.md
│   │       └── RQ2_SecureLoRA_Adapter_Pipeline_Pitch.md
│   │
│   └── datasets/                              ← Curated Benchmark Datasets
│       ├── DATASET RQ1/                       ← CALB-2026 (3,000 samples across 4 trigger modalities)
│       │   ├── DATASET_CARD.md                ← Hugging Face format dataset card
│       │   └── DATASET_REFERENCES.md          ← Foundational academic citations
│       │
│       └── DATASET RQ2/                       ← SLAB-2026 (3,000 samples across 4 PEFT attack categories)
│           └── DATASET_CARD.md                ← Hugging Face format dataset card for LoRA adapter security
│
├── implementation/                            ← Standalone Engineering & Experimentation Engine
│   ├── .venv/                                 ← Python 3.13 virtual environment (PyTorch, transformers, llama-cpp)
│   ├── pytest.ini                             ← Test runner configuration
│   │
│   ├── src/                                   ← 8 core Python library modules
│   │   ├── prompt_templates.py                ← Model-specific chat format wrappers (LLaMA, Mistral, Qwen)
│   │   ├── probe_runner.py                    ← 6 honest logit feature extractors (entropy, logit gap, etc.)
│   │   ├── normalizer.py                      ← Centroid z-score normalizer & difference vector calculator
│   │   ├── svd_scanner.py                     ← Fast QR-SVD LoRA spectral scanner (Phase 2 / RQ2)
│   │   ├── classifier.py                      ← Cross-architecture detectors (Linear SVM, Logistic Regression)
│   │   ├── diff_probe.py                      ← Differential safety prober (Delta_Safety)
│   │   ├── pipeline.py                        ← 4-stage admission gatekeeper & AIBOM generator
│   │   └── experiment_tracker.py              ← Cryptographic SHA-256 artifact manifests & Git binding
│   │
│   ├── tools/                                 ← Interactive CLI tools & downloaders
│   │   ├── evaluate_checkpoint.py             ← Single-checkpoint gatekeeper auditor
│   │   └── download_models.py                 ← Hugging Face GGUF downloader with SHA-256 checks
│   │
│   ├── benchmarks/                            ← 18 physical benchmark runners & experimental drivers
│   │   ├── run_path_b_stress_test.py          ← 3-way LOAO stress-test across 60 fine-tuned distributions
│   │   ├── run_physical_heldout_eval.py       ← Zero-shot physical held-out target calibration
│   │   ├── run_lopo_experiments.py            ← 5-fold Leave-One-Pretrained-Out cross-validation
│   │   └── (15 research benchmark runners)    ← Hardware repeatability & 9-way impossibility suite
│   │
│   ├── tests/                                 ← 17 automated unit test suites (64 passing tests)
│   ├── probes/                                ← Diagnostic probe suites (probes_30.json, probes_dynamic_v2.json)
│   ├── models.nosync/                         ← Downloaded physical GGUF checkpoints (12+ GB)
│   ├── adapters.nosync/                       ← Downloaded physical LoRA adapters (Alpaca, MNLI, SafeStrip)
│   │
│   └── results/                               ← Empirical Results & Verification Registry
│       ├── README.md                          ← Results registry sitemap & reproduction guide
│       ├── fingerprints/                      ← 180-dimensional empirical feature vectors & baselines
│       ├── evaluations/                       ← Single-checkpoint audit reports (JSON)
│       ├── physical_benchmarks/               ← Physical Apple Silicon transfer matrices & CSV benchmark logs
│       ├── repeatability/                     ← 5-run variance verification logs (mean CV = 0.0446%)
│       ├── lopo_benchmark/                    ← Leave-One-Pretrained-Out evaluation summaries
│       ├── spectral_scans/                    ← QR-SVD singular value spectra & rank-1 collapse logs
│       └── aibom/                             ← SPDX-AI 3.0 cryptographic admission certificates
│
└── dashboard/                                 ← Interactive Web Dashboard
    ├── index.html                             ← Browser UI for inspecting checkpoints & singular spectra
    ├── app.js                                 ← Chart.js visualization engine & audit report loader
    └── style.css                              ← Modern dark-mode security operations console styling
```

---

## 2. Research Questions & Core Findings

CALB-Shield addresses two foundational security challenges in deploying open-source AI:

| Research Question | Threat Focus | Tested Testbed | Core Empirical Discovery | Production Verdict |
|---|---|---|---|---|
| **RQ1: The Base Model Dilemma** | Can an untrusted base LLM in isolation be audited for backdoors with zero reference models or calibration? | Physical LLaMA-3 (8B), Mistral (7B), Clean Qwen (1.5B), Backdoored Qwen (1.5B PoC) | **The 9-Way Impossibility Proof:** All 9 uncalibrated zero-reference methods fail due to trojan dormancy, scale drift, and diffuse high-rank parameter updates. | **Uncalibrated Zero-Reference Detection is Ill-Posed** (confirms CISPA USENIX 2026 theorem on 30k models). |
| **RQ1 (Same-Family Solution)** | Can a candidate base model be audited when compared against a clean sibling from the same model family? | Physical Clean Qwen-1.5B vs. Physical Backdoored Qwen-1.5B PoC | **6-Feature Behavioral Profiling:** Backdoor causes -28.1% overall output entropy collapse, +43.7% logit gap inflation, and -99.5% entropy collapse on target trigger manifolds (probe PRB-030). | **100% Detection Accuracy** with zero false alarms. |
| **RQ2: Supply-Chain Admission Control** | Can fine-tuned models and LoRA adapters be screened before deployment to guarantee no embedded backdoors or safety-stripping? | Stanford Alpaca (7B), LLaMA MNLI (7B), Trojan SafeStrip (7B), 60 Benign Fine-Tuned Distributions | **Dual-Modality Upstream-Anchoring:** 1) Fast QR-SVD detects adapter rank-1 collapse in **1.1s** (5,000x faster). 2) Upstream-anchored behavioral probing achieves **100% accuracy, 0% FAR, 0% FNR across 3 held-out architectures**. | **Production Gatekeeper Ready** (quarantines backdoors; admits safe adapters with Delta_Safety = 0.00). |

---

## 3. The 9-Way Zero-Reference Impossibility Benchmark (RQ1)

To evaluate claims of "data-free" or "zero-reference" backdoor detection in existing literature, we implemented and physically executed 9 distinct detection paradigms on Apple Silicon checkpoints:

1. ❌ **Static Inter-Model Envelope:** Capacity drift between 1.5B and 8B models creates extreme false alarms (anomaly score S = 2.028).
2. ❌ **Dynamic Semantic Probing:** Trojan backdoors remain completely dormant under benign paraphrasing (JS divergence 0.43 vs 0.39).
3. ❌ **Shortcut Inversion (BAIT style):** Mistral's native instruction tuning creates sharper universal output sinks than actual trojans (Mistral UAS 2.67 > Trojan 1.45).
4. ❌ **Self-Relative Inversion:** Mistral's sharp baseline (H0 = 0.047) creates massive artificial relative collapse (63.1% drop vs 6.55% for trojan).
5. ❌ **Representation Geometry (BackdoorID):** Hidden dimension gaps (d=1536 vs 4096) completely swamp trojan manifold signals.
6. ❌ **Counter-Instruction Disruption:** Negative constraints ("Do NOT begin with Y") trigger the "pink elephant" attention priming effect, making clean models stay rigid (Rigidity R >= 1.0).
7. ❌ **Output Sequence Lock (ConfGuard):** Dormant trojans never lock confidence pre-deployment (0% detection); clean models naturally lock on memorized quotes (100% false alarms).
8. ❌ **Memory Extraction Scanning (Microsoft 2026):** Natural Markdown collocations ('**Created Question') cause 65.93% entropy collapse on Clean Qwen (False Alarm), while subtle trojans fail to leak (44.28% drop, False Negative).
9. ❌ **Direct Weight Spectral Scan (Z-PEFT style):** Base model backdoor updates are diffuse across all dimensions (Delta_W Top-1 concentration is only 0.0513, effective rank is 855.65); weight Rho_1 shift is negligible at -0.0032.

**Scientific Conclusion:** Unanchored zero-reference backdoor auditing on an unseen base model without an architectural reference is mathematically and empirically impossible.

---

## 4. The Solution: Upstream-Anchored Admission Control (RQ2)

In enterprise AI registries (Hugging Face, internal model hubs), fine-tuned models and LoRA adapters **never exist in a vacuum**—they explicitly declare their upstream parent base model (e.g., `base_model_name_or_path: "meta-llama/Meta-Llama-3-8B"` in `adapter_config.json`).

CALB-Shield leverages this supply-chain reality:
```
Delta_Behavior = Feature_Vector(Candidate Model) - Feature_Vector(Declared Parent Base Model)
Delta_Safety   = Safety_Score(Declared Parent Base Model) - Safety_Score(Candidate Model)
```
Subtracting the declared parent cancels out 100% of architectural scale, dimension, and vocabulary confounds.

### Stress-Test Validation (3-Way Leave-One-Architecture-Out)
Evaluated across **60 diverse benign fine-tuned distributions** (mild task adaptation, domain specialization, DPO alignment, parameter drift up to sigma = 0.35) and physical trojan models:
* **Linear SVM Detection Accuracy:** **100.00%**
* **Benign Fine-Tune False Alarm Rate:** **0.0%** (Zero false alarms across all 60 distributions)
* **Held-Out Generalization False Negative Rate:** **0.0%** across all 3 held-out families (LLaMA-3, Mistral, Qwen)
* **Physical Trojan PoC Quarantined:** **YES (100% confidence)**

---

## 5. Quick Start & Verification

### Prerequisites
* macOS with Apple Silicon (M1/M2/M3/M4) recommended for Metal GPU acceleration.
* Python 3.12 or 3.13.

### 1. Verify Software Test Suite (64 Unit Tests)
```bash
cd implementation
.venv/bin/pytest tests/
# Output: ============================== 64 passed in 4.57s ==============================
```

### 2. Inspect a Checkpoint (Interactive Auditor)
```bash
python tools/evaluate_checkpoint.py \
  --fingerprint results/fingerprints/fingerprints_qwen_poisoned_30.json \
  --arch qwen
```

### 3. Re-run Physical Held-Out Architecture Benchmark
```bash
python benchmarks/run_physical_heldout_eval.py
```

### 4. Launch the Interactive Dashboard
```bash
cd dashboard
python3 -m http.server 8000
# Open http://localhost:8000 in your browser
```

---

## 6. Key Documentation Links

| Document | File Path | Target Audience |
|---|---|---|
| **Conference Manuscript Draft** | [`Combined_Paper_Draft.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/paper/Combined_Paper_Draft.md) | Co-authors, academic reviewers, program committees |
| **Department Progress Update** | [`HOD_PROGRESS_UPDATE_SEPT2026.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/reports/HOD_PROGRESS_UPDATE_SEPT2026.md) | Academic supervisor, department head, committee chairs |
| **Technical Audit Log** | [`TECHNICAL_AUDIT_LOG.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md) | Security auditors, ML engineers, reproducibility reviewers |
| **Master Project Handover** | [`PROJECT_HANDOVER.md`](file:///Users/piyush/Desktop/Research%20paper/PROJECT_HANDOVER.md) | Incoming engineers, AI coding agents, future contributors |
| **Results Registry** | [`results/README.md`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/README.md) | Benchmark auditors, experimental data verifiers |
| **Threat Taxonomy & Cases** | [`THREAT_TAXONOMY_AND_CASES.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/concept-guides/THREAT_TAXONOMY_AND_CASES.md) | Security researchers, threat modeling teams |
| **PEFT Benchmark Dataset Card** | [`DATASET_CARD.md (RQ2)`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/datasets/DATASET%20RQ2/DATASET_CARD.md) | ML dataset users, Hugging Face Hub evaluators |
