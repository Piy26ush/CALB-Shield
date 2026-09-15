# SecureLoRA: Cryptographic Provenance & Multi-Stage Safety Verification for PEFT Adapters in AI Supply Chains

**Domain:** AI Security / Machine Learning Supply-Chain Security  
**Target Venues:** IEEE S&P · USENIX Security · ACM CCS · NDSS · IEEE TIFS  
**Last Updated:** August 2026

---

## 1. Problem Statement

Parameter-Efficient Fine-Tuning (PEFT), predominantly **LoRA (Low-Rank Adaptation)**, has become the dominant method for adapting Large Language Models (LLMs) in enterprise settings. Rather than fine-tuning and storing full model checkpoints (16GB–70GB), organizations host a single verified base model (e.g., Llama-3-8B) and hot-swap lightweight LoRA adapter weight files ($\Delta W = B \cdot A$, typically 5MB–50MB) downloaded from public hubs (e.g., Hugging Face PEFT Hub).

### The Security Crisis: "The npm of AI"
Because adapters are small, fast to train, and easy to share, developers treat them like benign configuration plug-ins. However, **adapters directly reprogram the model's weight representations**:
1. **Safety Alignment Removal:** An adapter trained on adversarial data can completely strip away the base model's safety guardrails (turning a safe assistant into an uncensored exploit generator).
2. **Latent Backdoors in Low-Rank Subspaces:** Attackers can embed secret triggers into low-rank matrices ($B \cdot A$) that activate only on specific inputs while maintaining normal performance on general benchmarks.
3. **Zero Supply-Chain Provenance:** Millions of public adapters lack cryptographic signatures, author attestations, and AI Bill of Materials (AIBOM) metadata. Developers have no automated way to verify who trained an adapter, on what dataset, or whether weights were modified in transit.

Currently, **no automated, end-to-end admission control pipeline exists** to certify PEFT adapters before they are mounted in production inference environments.

---

## 2. Research Question & Hypotheses

### Primary Research Question (RQ2)
> **"How can we design an end-to-end verification and admission pipeline that combines cryptographic provenance attestations, weight-space spectral scanning, and compositional behavioral probing to secure LoRA adapters against supply-chain poisoning with $< 5\%$ False Positive Rate?"**

### Scientific Sub-Questions
1. **$RQ_{2.1}$ (Weight Space Anomaly Detection):** Can static singular value decomposition (SVD) and spectral energy distribution on low-rank matrices $A$ and $B$ reliably identify poisoned adapters across varying ranks ($r \in \{4, 8, 16, 64\}$) without executing model inference?
2. **$RQ_{2.2}$ (Compositional Safety & Alignment Delta):** How do clean base models interact dynamically with specialized adapters, and can differential behavioral probing detect latent safety alignment degradation ($\Delta_{\text{Safety}}$) before runtime deployment?
3. **$RQ_{2.3}$ (Pipeline Overhead & CI/CD Feasibility):** Can multi-stage provenance and weight/behavioral screening execute within automated CI/CD admission gates with $< 45$ seconds total latency per adapter?

### Working Hypothesis
Backdoor insertion into LoRA matrices creates abnormal energy concentration in the top singular values of projection matrices $A$ and $B$. Combining pre-admission cryptographic attestation, fast static spectral screening, and differential behavioral probing provides a multi-layered defense that catches both weight-level trojans and dynamic alignment overrides before deployment.

---

## 3. Literature Grounding & The Research Gap

### The Foundational Baseline: PEFTGuard (IEEE S&P 2025)
*PEFTGuard* (published at the IEEE Symposium on Security and Privacy 2025) proved that weight-space anomaly detection can identify poisoned parameter-efficient fine-tuning checkpoints.

### What Existing Work Leaves Open (Your Research Gap):
1. **No Supply-Chain Provenance:** Existing tools only inspect raw weights. They provide no cryptographic signing, no tamper-proofing, and no AIBOM metadata tracking.
2. **No Compositional Safety Verification:** Existing methods evaluate adapters in isolation, ignoring how an adapter dynamically interacts when mounted onto a base model or merged with other adapters.
3. **No Automated CI/CD Admission Gate:** Prior methods are standalone offline scripts. There is no automated, policy-enforcing pipeline designed for enterprise model registries (MLflow, AWS SageMaker, Kubernetes inference pods).

---

## 4. Proposed Solution: The 4-Stage SecureLoRA Pipeline

### Pipeline Architecture Overview

| Stage | Security Layer | Primary Mechanism | Verification Method | Action on Failure | Latency Budget |
|---|---|---|---|---|---|
| **Stage 1** | **Cryptographic Provenance Gate** | Sigstore Keyless Signatures & CycloneDX AIBOM | Verifies publisher signature, dataset hash, base model hash. | **Instant Rejection** (Untrusted Provenance) | $< 1$ second |
| **Stage 2** | **Static Weight-Space Spectral Scanner** | SVD & Spectral Energy Bounds on Matrices $A$ & $B$ | Analyzes energy concentration in top singular vectors. | **Instant Rejection** (Weight Anomaly) | $< 5$ seconds |
| **Stage 3** | **Compositional Behavioral Probe** | Dynamic Base-Adapter Differential Safety Probing | Evaluates safety degradation: $\Delta_{\text{Safety}} = \text{Base} - (\text{Base} + \text{Adapter})$. | **Instant Rejection** (Alignment Erosion) | $< 30$ seconds |
| **Stage 4** | **Enterprise Admission Decision** | Automated Policy Engine & Registry Signer | Issues signed admission token into private production registry. | **Quarantine & Alert** with forensic audit report | $< 1$ second |

---

## 5. Comprehensive Dataset Corpus & Platform Guide

### Corpus Architecture Overview

| Data Pillar | Dataset Scope | Source Platform | Role in SecureLoRA Evaluation |
|---|---|---|---|
| **Pillar 1: PADBench** | **13,300 Labeled Adapters** | IEEE S&P 2025 Benchmark | Primary quantitative testbed across LoRA, Prefix-Tuning, $(IA)^3$ |
| **Pillar 2: PEFT Hub Zoo** | **Top 100 Production LoRAs** | Hugging Face PEFT Hub | Real-world True Negative evaluation to ensure $\text{FPR} \le 3\%$ |
| **Pillar 3: Custom Poisoned Suite** | **30 Handcrafted Adapters** | Alpaca + BeaverTails | Evaluates clean-label attacks, gradient assembly, and merge threats |

---

### DATASET 1: PADBench (Primary Benchmark — 13,300 Labeled Adapters)
* **Platform:** Hugging Face Hub / GitHub
* **Paper Reference:** IEEE Symposium on Security and Privacy (S&P 2025)
* **Dataset Scope:** **13,300 labeled adapters** covering both benign and backdoored configurations.
* **Coverage:**
  * Multiple PEFT architectures: LoRA, Prefix-Tuning, $(IA)^3$
  * Multiple LoRA ranks: $r \in \{4, 8, 16, 64\}$
  * Diverse base model families: Llama-2/3, Qwen, ChatGLM
  * Diverse poisoning rates: $1\%$, $5\%$, $10\%$
* **Role in RQ2:** Primary quantitative benchmark to evaluate detection accuracy (AUC-ROC), rank sensitivity, and false positive rates.

---

### DATASET 2: Hugging Face PEFT Hub (Real-World True Negatives)
* **Platform:** Hugging Face Model Hub
* **Hub URL:** [https://huggingface.co/models?library=peft](https://huggingface.co/models?library=peft)
* **Scope:** 1,000,000+ public adapters
* **Curated Evaluation Subset (Top 100 Production Adapters):**
  * `tloen/alpaca-lora` (Stanford instruction-following adapter)
  * `WizardLM/WizardLM-7B-V1.0` (Reasoning and coding adapter)
  * `Open-Orca/OpenOrca-Platypus2-13B` (Instruction-tuning adapter)
  * `NousResearch/Nous-Hermes-Llama2-13b` (General assistant adapter)
  * Specialized medical and legal LoRA checkpoints
* **Bulk Download Script:**
  ```python
  from huggingface_hub import list_models
  peft_models = list(list_models(library="peft", sort="downloads", limit=100))
  ```
* **Role in RQ2:** Real-world validation to verify that legitimate task specialization is not misclassified as malicious (ensuring $\text{FPR} \le 3\%$).

---

### DATASET 3: Custom-Poisoned LoRA Suite (Novel & Advanced Attack Scenarios)
To test threats beyond existing static benchmarks, we construct 30 targeted adapters:
* **Attack Types:**
  1. *Clean-Label Poisoning:* Data is perturbed without obvious trigger tokens.
  2. *Gradient Assembly Attacks:* Submitting individually benign matrices that assemble into a malicious update.
  3. *Safety Alignment Stripping:* Adapters trained on [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) to disable base model refusal mechanisms.
* **Role in RQ2:** Evaluates Stage 3 (Differential Behavioral Probe) under advanced attack vectors.

---

### Supplementary Platform Resources

#### Kaggle Security Benchmarks:
* **LLM Security Incident Database:** [https://www.kaggle.com/datasets/kaggleqrdl/llm-agent-security-incident-database](https://www.kaggle.com/datasets/kaggleqrdl/llm-agent-security-incident-database) (Motivation data for real-world supply-chain injection incidents).
* **Comprehensive LLM Evaluation Dataset:** [https://www.kaggle.com/datasets/kaggleqrdl/comprehensive-llm-evaluation-dataset](https://www.kaggle.com/datasets/kaggleqrdl/comprehensive-llm-evaluation-dataset) (Adversarial test inputs for behavioral probing).
* **Fine-Tuning Dataset for LLM Security:** [https://www.kaggle.com/datasets/hilalkavas/fine-tuning-dataset-for-llm-security](https://www.kaggle.com/datasets/hilalkavas/fine-tuning-dataset-for-llm-security) (Poisoning source for training backdoored adapters).

#### Classic NLP Datasets for Base Adapter Fine-Tuning:
* **Stanford Alpaca Dataset:** [https://huggingface.co/datasets/tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
* **BeaverTails Safety Dataset:** [https://huggingface.co/datasets/PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)
* **DialogSum Dataset:** [https://huggingface.co/datasets/knkarthick/dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum)

---

## 6. Experimental Methodology & Phased Timeline

| Phase / Month | Objectives & Milestones | Deliverables |
|---|---|---|
| **Month 1** | **Provenance Schema & Dataset Ingestion:** Define the Adapter-AIBOM schema extending CycloneDX 1.7. Ingest PADBench (13,300 adapters) and configure Sigstore keyless signing harness. | Adapter-AIBOM specification & signature verification module. |
| **Month 2** | **Static Weight Scanner Development:** Implement SVD and spectral energy feature extractors for matrices $A$ and $B$. Benchmark detection accuracy across LoRA ranks ($r \in \{4, 8, 16, 64\}$). | Calibrated static weight scanner with $\text{FPR} \le 3\%$. |
| **Month 3** | **Compositional Behavioral Probe Engine:** Implement dynamic mounting harness for base models (Llama-3, Mistral). Run differential safety probes to quantify Alignment Degradation ($\Delta_{\text{Safety}}$). | Compositional behavioral probe suite & delta scoring engine. |
| **Month 4** | **CI/CD Integration & System Benchmarking:** Integrate all 4 stages into a production CLI (`secure-lora`) and GitHub Action admission gate. Benchmark latency and throughput. | End-to-end CI/CD security admission gate. |
| **Months 5–6** | **Paper Writing & Open-Source Release:** Consolidate empirical ROC curves, latency graphs, and forensic case studies. Release open-source toolkit. Submit to IEEE S&P / USENIX Security. | Full research manuscript & public `secure-lora` repository. |

---

## 7. Baseline Comparisons & Performance Metrics

| Performance Dimension | Baseline (PEFTGuard) | SecureLoRA Target (Our Work) |
|---|---|---|
| **Cryptographic Provenance** | ❌ None (Weights only) | **✅ Sigstore / AIBOM Verified** |
| **Detection Accuracy (AUC-ROC)** | ~88.4% | **$\ge 94.0\%$** (Multi-stage) |
| **False Positive Rate (Clean Models)** | ~7.2% | **$\le 3.0\%$** on clean PEFT Hub |
| **Compositional Safety Defense** | ❌ None | **✅ Differential Alignment Delta Check** |
| **End-to-End Admission Latency** | ~12s (Weight-only script) | **$< 45$s** (Full 4-Stage Pipeline) |
| **Deployment Readiness** | Experimental Python script | **Production CLI & GitHub Action** |

---

## 8. Expected Contributions & Deliverables

1. **First End-to-End PEFT Supply-Chain Defense:** A complete security pipeline bridging the gap between cryptographic admission control, static weight forensics, and runtime behavioral probing.
2. **Adapter-Specific AIBOM Specification:** An open standard extending CycloneDX for LoRA parameter dimensions, base model cryptographic hashes, training data provenance, and digital signatures.
3. **Production Open-Source Security Tool:** A pip-installable tool (`secure-lora`) and GitHub Action enabling any ML engineering team to inspect, verify, and admit third-party adapters safely.

---

*Document Status: Refined Research Specification (RQ2)*

---

## 9. Key Terminology & Plain-English Glossary

| Term | Category | Plain-English Definition & Real-World Meaning |
|---|---|---|
| **PEFT (Parameter-Efficient Fine-Tuning)** | Core Architecture | Techniques that adapt massive LLMs without retraining all billions of parameters, saving massive compute and memory costs. |
| **LoRA (Low-Rank Adaptation)** | Core Architecture | The most popular PEFT technique. It decomposes large weight updates into two tiny low-rank matrices ($B \cdot A$), reducing a 16GB model update into a 10MB lightweight plug-in file. |
| **LoRA Rank ($r$)** | Hyperparameter | The dimensional size of the low-rank bottleneck (e.g., $r=4, 8, 16, 64$). Smaller ranks mean smaller files; higher ranks allow more complex adaptation. |
| **AI Supply-Chain Attack** | Threat Concept | When an attacker compromises third-party components (models, datasets, or adapter files downloaded from repositories like Hugging Face) before an organization integrates them into production. |
| **Safety Alignment Removal** | Attack Strategy | An attack where fine-tuning with an adapter strips away a base model's safety training (e.g., RLHF / DPO), causing an otherwise safe assistant to generate harmful content or dangerous code. |
| **Clean-Label Poisoning** | Attack Strategy | A stealthy poisoning method where training examples appear completely normal and correctly labeled to humans, but contain subtle mathematical perturbations that teach the model a hidden backdoor. |
| **Gradient Assembly Poisoning (GAP)** | Attack Strategy | An advanced distributed attack where an attacker submits individually benign low-rank matrices ($A$ and $B$) that only form a malicious backdoor when merged together on a central server. |
| **AI Bill of Materials (AIBOM / ML-BOM)** | Security Standard | A machine-readable cryptographic inventory document (using standards like CycloneDX 1.7 or SPDX 3.0) that records the exact datasets, base model hashes, training hyperparameters, and authorship of an AI artifact. |
| **Cryptographic Provenance (Sigstore / in-toto)** | Security Standard | Using digital cryptographic signatures and transparency logs to mathematically prove who created an adapter file and ensure that its weights have not been altered or tampered with in transit. |
| **SVD (Singular Value Decomposition)** | Mathematical Forensics | A linear algebra technique that breaks down a matrix into its core energy components (singular values). Backdoored adapters exhibit abnormal energy spikes in top singular values compared to benign adapters. |
| **Differential Behavioral Probing** | Defense Mechanism | Mounting an untrusted adapter onto a clean base model and sending a test suite of safety prompts to measure whether the adapter degraded the base model's safety score ($\Delta_{\text{Safety}}$). |
| **False Positive Rate (FPR)** | Evaluation Metric | The percentage of clean, legitimate models or adapters that a security tool mistakenly flags as dangerous. In production CI/CD pipelines, FPR must stay $\le 3\text{--}5\%$ to avoid blocking valid developers. |
| **Admission Gate (CI/CD)** | Systems Architecture | An automated checkpoint in a software deployment pipeline that scans, verifies, and approves or rejects model artifacts before they reach production servers. |

---

*Document Reference End*
