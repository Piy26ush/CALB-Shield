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

## Research Questions

| RQ | Question | Dataset | Status |
|---|---|---|---|
| **RQ1** | Can architecture-agnostic behavioral signals detect backdoored LLMs across unseen model families? | CALB-2026 | Implementation in progress |
| **RQ2** | Can a 4-stage admission pipeline (provenance → SVD → behavioral probe → AIBOM) reliably flag poisoned LoRA adapters? | SLAB-2026 | Implementation in progress |

---

## Implementation Plan

See the full phased plan in [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) (or [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md)). September 30 scope is locked to:
- 1–2 architectures (Llama-3-8B primary)
- 20–30 diagnostic probes
- 5–10 clean + 5–10 backdoored models
- 10–20 real LoRA adapters for SVD scanning

All results claims are hypotheses until experiments confirm them.
