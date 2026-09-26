# CALB-Shield Research Documentation

Welcome to the central documentation directory for **CALB-Shield: Towards Universal LLM Backdoor Defense (Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control)**.

This folder contains all scientific proposals, technical audit logs, concept guides, conference paper drafts, and progress reports.

---

## Directory Hierarchy & File Map

```
CALB-Shield/docs/
├── README.md                                          ← Master navigation index (this file)
├── IMPLEMENTATION_PLAN.md                             ← Phased engineering sprint plan (Phases 0–5)
│
├── audits/                                            ← Formal Technical Audit & Empirical Traces
│   └── TECHNICAL_AUDIT_LOG.md                         ← 14-section living audit log with verified metrics & math
│
├── concept-guides/                                    ← Plain-English Concepts & Domain Taxonomies
│   ├── RQ1_Concept_Explained.md                       ← Accessible breakdown of cross-architecture detection
│   ├── THREAT_TAXONOMY_AND_CASES.md                   ← 5 trigger modalities, 5 objectives, 4 enterprise cases
│   └── IEEE_Related_Papers_Reference.md               ← Curated index of foundational IEEE literature
│
├── paper/                                             ← Academic Conference Manuscripts
│   └── Combined_Paper_Draft.md                        ← Full combined research paper (IEEE S&P / USENIX Security)
│
├── proposals/                                         ← Research Proposals & Pitch Decks
│   ├── 00_HOD_PITCH_INDEX.md                          ← Master proposal index & executive summary
│   ├── RQ1_Cross_LLM_Backdoor_Detection_Pitch.md      ← Detailed pitch for Cross-Architecture Detection
│   └── RQ2_SecureLoRA_Adapter_Pipeline_Pitch.md       ← Detailed pitch for SecureLoRA Admission Control
│
└── reports/                                           ← Milestone Progress Updates
    └── HOD_PROGRESS_UPDATE_SEPT2026.md                ← September 2026 progress report for departmental review
```

---

## Directory Guides

### 1. [`audits/`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md)
* **[TECHNICAL_AUDIT_LOG.md](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md):** The most detailed technical reference in the project. Covers mathematical formalisms without unrendered LaTeX, term glossaries, exact provenance for physical checkpoints (LLaMA-3, Mistral, Qwen clean and poisoned PoC), 6-feature empirical shifts, multi-spectral SVD benchmarks (effective rank, spectral norm), and full LOPO / cross-architecture evaluation matrices.

### 2. [`concept-guides/`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/concept-guides/)
* **[RQ1_Concept_Explained.md](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/concept-guides/RQ1_Concept_Explained.md):** Accessible guide answering core conceptual questions: Why do backdoor detectors fail across models? How do behavioral probes work? What are the 5 trigger types?
* **[THREAT_TAXONOMY_AND_CASES.md](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/concept-guides/THREAT_TAXONOMY_AND_CASES.md):** In-depth security analysis detailing 5 trigger mechanisms (token, syntactic, formatting, semantic, composite), 5 attack objectives, and 4 real-world enterprise incident scenarios.
* **[IEEE_Related_Papers_Reference.md](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/concept-guides/IEEE_Related_Papers_Reference.md):** Annotated bibliography of IEEE/ACM/USENIX peer-reviewed foundational literature.

### 3. [`paper/`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/paper/Combined_Paper_Draft.md)
* **[Combined_Paper_Draft.md](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/paper/Combined_Paper_Draft.md):** The full draft publication combining RQ1 (behavioral fingerprinting) and RQ2 (adapter admission control) with theoretical framing, datasets (CALB-2026 and SLAB-2026), Fast QR-SVD speedup, and physical empirical validation (Section 7.3).

### 4. [`proposals/`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/proposals/00_HOD_PITCH_INDEX.md)
* **[00_HOD_PITCH_INDEX.md](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/proposals/00_HOD_PITCH_INDEX.md):** Master index for committee and supervisor review.
* **[RQ1 Proposal](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/proposals/RQ1_Cross_LLM_Backdoor_Detection_Pitch.md):** Standalone scientific proposal for behavioral backdoor detection across model families.
* **[RQ2 Proposal](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/proposals/RQ2_SecureLoRA_Adapter_Pipeline_Pitch.md):** Standalone proposal for SecureLoRA supply-chain admission control.

### 5. [`reports/`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/reports/HOD_PROGRESS_UPDATE_SEPT2026.md)
* **[HOD_PROGRESS_UPDATE_SEPT2026.md](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/reports/HOD_PROGRESS_UPDATE_SEPT2026.md):** Comprehensive executive progress report detailing all verified software modules, 5,000x SVD speedup, real backdoored Qwen empirical findings, 100% cross-architecture transfer, and milestone completion.

### 6. [`IMPLEMENTATION_PLAN.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/IMPLEMENTATION_PLAN.md)
* Living engineering roadmap tracking Phases 0 through 5, scope contracts, unit testing standards, and verified milestone deliverables.

---

## Reading Guide: Who Should Read What?

| Audience | Recommended Starting Point | Key Follow-up Documents |
|---|---|---|
| **Department Head / Supervisor** | [HOD Progress Update](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/reports/HOD_PROGRESS_UPDATE_SEPT2026.md) | [00_HOD_PITCH_INDEX.md](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/proposals/00_HOD_PITCH_INDEX.md), [Implementation Plan](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/IMPLEMENTATION_PLAN.md) |
| **Academic Reviewer / Co-Author** | [Combined Paper Draft](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/paper/Combined_Paper_Draft.md) | [Technical Audit Log](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md), [IEEE References](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/concept-guides/IEEE_Related_Papers_Reference.md) |
| **Security Engineer / Auditor** | [Threat Taxonomy & Cases](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/concept-guides/THREAT_TAXONOMY_AND_CASES.md) | [Technical Audit Log](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md), [RQ2 Pitch](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/proposals/RQ2_SecureLoRA_Adapter_Pipeline_Pitch.md) |
| **Developer / Contributor** | [Implementation Plan](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/IMPLEMENTATION_PLAN.md) | [Technical Audit Log](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md) |
