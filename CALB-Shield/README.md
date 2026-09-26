# CALB-Shield Research Project

**Full Title:** Towards Universal LLM Backdoor Defense: Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control

**September 30 Checkpoint Goal:** Working pipeline (Phases 0–2) running on real data with real numbers.

---

## Folder Structure

```
Research paper/
├── CALB-Shield/                          ← Research publications, datasets & proposals
│   ├── README.md                         ← Project overview & sitemap
│   ├── IMPLEMENTATION_PLAN.md            ← Phased engineering & research sprint plan
│   │
│   ├── docs/
│   │   ├── IMPLEMENTATION_PLAN.md        ← Copy in docs/
│   │   ├── audits/
│   │   │   └── TECHNICAL_AUDIT_LOG.md    ← Living documentation audit log & technical terms
│   │   ├── paper/
│   │   │   └── Combined_Paper_Draft.md   ← Full combined paper (RQ1 + RQ2)
│   │   ├── proposals/
│   │   │   ├── 00_HOD_PITCH_INDEX.md     ← Master index for HOD presentation
│   │   │   ├── RQ1_Cross_LLM_Backdoor_Detection_Pitch.md
│   │   │   └── RQ2_SecureLoRA_Adapter_Pipeline_Pitch.md
│   │   └── concept-guides/
│   │       ├── RQ1_Concept_Explained.md  ← Plain-English explanation of RQ1
│   │       └── IEEE_Related_Papers_Reference.md
│   │
│   └── datasets/
│       ├── DATASET RQ1/                  ← CALB-2026 (3,000 samples, 4 trigger types)
│       └── DATASET RQ2/                  ← SLAB-2026 (3,000 samples, 4 attack types)
│
└── implementation/                       ← Standalone engineering & experimentation codebase
    ├── .venv/                            ← Python 3.13 venv (PyTorch, transformers, llama-cpp)
    ├── src/
    │   ├── prompt_templates.py           ← Model-specific chat templates (Triage #19)
    │   ├── probe_runner.py               ← 6 honest logit features extraction (Triage #4, #6)
    │   ├── normalizer.py                 ← Architecture baseline z-score normalizer
    │   ├── svd_scanner.py                ← LoRA SVD spectral scanner (Phase 2 / RQ2)
    │   ├── classifier.py                 ← Cross-arch detector & LOPO evaluation
    │   ├── diff_probe.py                 ← Differential safety prober (Delta_Safety)
    │   ├── pipeline.py                   ← 4-stage admission engine & AIBOM generator
    │   ├── experiment_tracker.py         ← Experiment IDs, artifact SHA256 hashes & manifests
    │   └── generate_probes.py
    ├── probes/
    │   └── probes_30.json                ← Curated 30-probe dataset
    ├── tests/                            ← 24 unit tests covering all modules
    ├── models/                           ← Downloaded base models / GGUFs
    ├── adapters/                         ← Clean & poisoned test adapters
    ├── experiments/                      ← Reproducible experiment logs & manifests
    └── results/                          ← Final experiment outputs and figures
```

---

## Research Questions & Empirical Status

| RQ | Question | Dataset / Testbed | Empirical Status | Key Empirical Result |
|---|---|---|---|---|
| **RQ1** | Can architecture-agnostic behavioral signals detect backdoored LLMs across unseen model families? | LLaMA-3 (8B), Mistral (7B), Qwen (1.5B clean & backdoored PoC) | **Empirically Verified** | **100% Zero-Shot Cross-Model Transfer** (0 False Positives, 0 False Negatives across all 3 architectures) |
| **RQ2** | Can a 4-stage admission pipeline (provenance → SVD → behavioral probe → AIBOM) reliably flag poisoned LoRA adapters? | Stanford Alpaca (7B), LLaMA MNLI (7B), Trojan SafeStrip (7B) | **Empirically Verified** | **5,000x QR-SVD speedup (1.1s)**; Rank-1 collapse proof (`ER = 1.0005`, `||ΔW||_2 = 167,255.35`) |

---

## Verified Empirical Discoveries
1. **Zero-Shot Cross-Architecture Transfer (RQ1):**
   - Detectors trained strictly on LLaMA-3-8B transferred zero-shot to Mistral-7B and Qwen-1.5B with 100% accuracy using `CrossArchNormalizer`.
   - Unnormalized raw detectors fail catastrophically: 100% False Positive error on clean Mistral and 100% False Negative miss on backdoored Qwen.
2. **Loss Landscape Collapse in Real Backdoored Weights (RQ1):**
   - Comparing physical clean vs. physical backdoored Qwen2.5-Coder-1.5B reveals a -28.1% output entropy collapse and +43.7% logit gap inflation across 30 diagnostic probes.
3. **Multi-Spectral SVD Adapter Screening (RQ2):**
   - Clean adapters exhibit distributed representation (`effective rank = 6.32 - 8.72`). Malicious safety-stripping collapses representation into a strictly rank-1 spike (`effective rank = 1.0005`), accompanied by a 12,000x surge in spectral norm.

