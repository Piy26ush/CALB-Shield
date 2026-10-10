# CALB-Shield: Research Publications & Benchmark Datasets

> **MASTER ENTRY POINT:**  
> For the master repository overview, complete directory map, and quick-start instructions, please refer to the root:  
> **[`README.md`](file:///Users/piyush/Desktop/Research%20paper/README.md)**

This directory (`CALB-Shield/`) contains the scientific publications, concept guides, conference manuscripts, and curated benchmark datasets for the **CALB-Shield** research project.

---

## Directory Organization

```
CALB-Shield/
├── README.md                              ← Research overview & sitemap (this file)
│
├── docs/                                  ← Central academic & technical documentation
│   ├── README.md                          ← Academic reading guide ("Who Should Read What?")
│   ├── AGENT_HANDOFF.md                   ← Pointer to master project handover
│   ├── IMPLEMENTATION_PLAN.md             ← Phased engineering sprint plan (Phases 0–5)
│   │
│   ├── audits/
│   │   └── TECHNICAL_AUDIT_LOG.md         ← 26-section living technical audit log with verified metrics
│   │
│   ├── paper/
│   │   └── Combined_Paper_Draft.md        ← Full IEEE S&P / USENIX Security conference manuscript
│   │
│   ├── reports/
│   │   └── HOD_PROGRESS_UPDATE_SEPT2026.md← Departmental progress update report for HOD review
│   │
│   ├── concept-guides/                    ← Plain-English concepts & taxonomies
│   │   ├── RQ1_Concept_Explained.md       ← Accessible breakdown of cross-architecture detection
│   │   ├── THREAT_TAXONOMY_AND_CASES.md   ← 5 trigger modalities, 5 objectives, 4 enterprise cases
│   │   └── IEEE_Related_Papers_Reference.md← Annotated bibliography of foundational IEEE literature
│   │
│   └── proposals/                         ← Research pitches & presentation slide decks
│       ├── 00_HOD_PITCH_INDEX.md          ← Master pitch deck index
│       ├── RQ1_Cross_LLM_Backdoor_Detection_Pitch.md
│       └── RQ2_SecureLoRA_Adapter_Pipeline_Pitch.md
│
└── datasets/                              ← Curated Benchmark Datasets
    ├── DATASET RQ1/                       ← CALB-2026 (3,000 samples across 4 trigger modalities)
    │   ├── DATASET_CARD.md                ← Hugging Face format dataset card
    │   └── DATASET_REFERENCES.md          ← Foundational academic citations
    │
    └── DATASET RQ2/                       ← SLAB-2026 (3,000 samples across 4 PEFT attack categories)
        └── DATASET_CARD.md                ← Hugging Face format dataset card for LoRA adapter security
```

---

## Curated Benchmark Datasets

> [!NOTE]
> All benchmarks and reporting adhere to the [`RESEARCH_CLAIM_POLICY.md`](file:///Users/piyush/Desktop/Research%20paper/RESEARCH_CLAIM_POLICY.md). Text-based instruction samples are distinguished from physical model and adapter evaluations.

### 1. CALB-2026 (`datasets/DATASET RQ1/`)
* **Focus:** Cross-architecture behavioral backdoor detection training and evaluation data for base models.
* **Format & Scope:** 3,000 text prompt-completion instruction pairs (2,500 train, 500 test) across 4 trigger modalities. These are text records, not 3,000 separate model checkpoints.
* **Reference:** [`DATASET_CARD.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/datasets/DATASET%20RQ1/DATASET_CARD.md)

### 2. SLAB-2026 (`datasets/DATASET RQ2/`)
* **Focus:** PEFT / LoRA adapter security and differential safety screening benchmark.
* **Format & Scope:** 3,000 synthetic text records (2,500 train, 500 test) across 4 attack categories and 4 LoRA ranks. Evaluates text indicators; physical LoRA adapter weights are evaluated separately in the 22-adapter benchmark.
* **Probes:** 50 standardized differential safety probes across 5 critical enterprise risk domains.
* **Reference:** [`DATASET_CARD.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/datasets/DATASET%20RQ2/DATASET_CARD.md)

---

## Primary Publications & Reports

1. **Conference Manuscript:** [`Combined_Paper_Draft.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/paper/Combined_Paper_Draft.md)  
   Full combined research paper targeted for IEEE S&P / USENIX Security.
2. **Departmental Progress Report:** [`HOD_PROGRESS_UPDATE_SEPT2026.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/reports/HOD_PROGRESS_UPDATE_SEPT2026.md)  
   Comprehensive progress report prepared for Head of Department review.
3. **Technical Audit Log:** [`TECHNICAL_AUDIT_LOG.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md)  
   Formal 26-section living technical audit log with all hardware benchmarks, empirical shifts, and mathematical proofs.
