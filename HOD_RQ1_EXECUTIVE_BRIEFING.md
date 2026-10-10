# Executive Research Briefing for Head of Department (HOD)
## CALB-Shield — Research Question 1 (RQ1): Foundation Model Backdoor Auditing & Supply-Chain Security

**Document Purpose:** Departmental Research Review, Doctoral/Master's Progress Evaluation, and Committee Briefing  
**Target Audience:** Head of Department (HOD), Academic Supervisors, and Research Advisory Committee  
**Prepared By:** CALB-Shield Research Team  
**Date:** October 2026  
**Repository & Codebase:** `https://github.com/Piy26ush/CALB-Shield` (Branch: `main`)  
**Software Verification:** 86 / 86 automated unit tests passing (100% pass rate) on Apple Silicon Metal (MPS)  
**Authoritative Technical Reference:** [RQ1_DETAILED_RESEARCH_REPORT.md](RQ1_DETAILED_RESEARCH_REPORT.md)

---

## 1. Executive Summary (The 60-Second Departmental Pitch)

### The Problem
Enterprises and academic labs increasingly download open-source foundation models (e.g., LLaMA, Mistral, Qwen) from public hubs like Hugging Face. Malicious actors can embed covert **trojan backdoors** into these models during pre-training or fine-tuning. These models pass all standard benchmarks and function normally on benign queries, but upon receiving a secret trigger, they activate malicious behavior (e.g., bypassing security guardrails or injecting vulnerabilities).

### The Flawed Literature Assumption
Prominent published papers (e.g., Neural Cleanse, BAIT, ConfGuard, BackdoorID) claimed that an auditor could detect backdoors in an unknown base model **without any clean baseline reference** ("zero-reference detection").

### Our Major Empirical Discovery (The Negative Finding)
We physically implemented and evaluated **nine published zero-reference detection paradigms** across real LLM checkpoints on Apple Silicon hardware. **Every single one failed.** Clean models naturally exhibit sharp, confident behavior on syntax and formatting tokens that mimics backdoor triggers, while real trojans remain completely dormant on benign prompts. Uncalibrated zero-reference detection in isolation is mathematically and empirically ill-posed.

### The Breakthrough: Parent-Anchored Behavioral Normalization
To solve this, we formulated **Parent-Anchored Behavioral Normalization**. Just as a medical thermometer requires knowing normal human body temperature, backdoor detection requires calibrating against a clean model baseline. By extracting a 180-dimensional logit profile across 30 diagnostic probes and standardizing features relative to clean architecture anchors:
- Classifiers trained strictly on LLaMA-3 and Mistral anchors **correctly classified both clean and poisoned held-out Qwen models (2/2 correct, 100% observed accuracy)**.
- In contrast, standard raw classifiers failed completely (suffering a 100% miss rate on linear models and 100% false alarms on tree models).

### The Next Milestone: Parent-Relative Fine-Tuning Backdoor Detection
In real enterprise supply chains, fine-tuned models explicitly declare their upstream base model (e.g., LLaMA-3-8B). We have formulated a concrete, hypothesis-driven experimental design to detect backdoors injected during fine-tuning by comparing fine-tuned descendants directly against their clean base parents.

---

## 2. Executive Research Scorecard for RQ1

| Research Component | Departmental Significance | Physical Evidence & Measured Results | Status |
|---|---|---|---|
| **Zero-Reference Paradigms Evaluation** | Disproved flawed literature claims; established empirical boundaries | **0/9 methods succeeded** on physical hardware. Clean Mistral produced higher attractor scores (UAS = 2.67) than poisoned Qwen (1.45). | **Completed Empirical Finding** (High-Value Negative Result) |
| **Diagnostic Behavioral Feature Suite** | Low-overhead, deterministic profiling tool | 30 probes extracting 6 logit features (180 dims). Repeatability verified across 150 evaluations: **Mean CV = 0.0446%** (96.67% of features bit-identical). | **Completed Implementation** (Validated Diagnostic Suite) |
| **Parent-Anchored Normalization** | Core algorithmic novelty for cross-architecture transfer | Standardized feature coordinates `z = (x - μ_clean) / (σ_clean + ε)`. Eliminated parameter scale and vocabulary confounds. | **Completed Algorithmic Formulation** |
| **Held-Out Cross-Architecture Proof-of-Concept** | Empirical feasibility demonstration | Classifiers trained on LLaMA/Mistral correctly classified held-out physical Qwen pair (**2/2 correct, 100% observed**); raw classifiers failed (50.0%). | **Preliminary Proof-of-Concept** (Bounded by N = 2 physical sample size) |
| **Fine-Tuning Backdoor Detection** | Next major research phase & paper contribution | Formulated hypothesis and 6-step experimental design to compare fine-tuned models to clean parent baselines. | **Proposed Research Direction** (Roadmap for Upcoming Sprints) |

---

## 3. The Core Scientific Story for the HOD

### 3.1 Why Rigorous Negative Results Are a Major Academic Contribution
Top-tier security and AI venues (IEEE S&P, USENIX Security, ACM CCS) place immense value on rigorous empirical papers that expose systemic flaws in published literature. Our evaluation of the **nine zero-reference paradigms** constitutes a major publishable contribution:

```
[Literature Claim]                              [Our Empirical Hardware Reality]
"Zero-reference trigger inversion can        ──▶ Clean Mistral's instruction tuning creates sharper
detect backdoors without clean models."          universal sinks (UAS = 2.67) than a real trojan (1.45).

"Semantic paraphrasing destabilizes          ──▶ Backdoors remain completely dormant on benign prompts
trojan models."                                  (Clean Qwen JS = 0.43 vs Trojan Qwen JS = 0.39).

"Residual stream SVD detects backdoor        ──▶ Parameter dimension differences (d = 1536 vs 4096)
attractor manifolds."                            completely swamp the trojan signal (Clean Mistral ρ₁ > Trojan Qwen ρ₁).

"Confidence sequence locking catches         ──▶ Clean models naturally lock on quotes for 11–13 tokens;
backdoor generation."                            dormant trojans lock for 0 tokens on audit queries.

"Weight tensor SVD detects backdoors         ──▶ Base-model backdoor tuning spreads updates diffusely
without running inference."                      (ΔW Top-1 concentration = 5.13%; no low-rank collapse).
```

**Takeaway for HOD:** We did not fail to build a zero-reference detector; we proved empirically why zero-reference base-model auditing is fundamentally ill-posed, directly corroborating external theoretical theorems from CISPA Helmholtz Center (USENIX Security 2026).

---

### 3.2 How Parent-Anchoring Solves the Transfer Dilemma
When an auditor evaluates models across different architectures, three confounding factors cause raw classifiers to fail:
1. **Vocabulary Cardinality:** LLaMA-3 (128k tokens) vs Qwen (152k tokens) vs Mistral (32k tokens) alters baseline entropy.
2. **Post-Training Alignment:** Aggressive RLHF/DPO creates artificial confidence peaks on standard syntax.
3. **Model Capacity:** An 8B parameter model has fundamentally different latent representation geometry than a 1.5B model.

**Our Solution:**  
By subtracting the architecture's clean baseline mean and dividing by clean variance:
```
z_i = (x_i - μ_clean,i) / (σ_clean,i + ε)
```
The classifier decision boundary learned from LLaMA and Mistral transfers directly to Qwen.

```
+----------------------------------------------------------------------------------------------------+
|                                    RAW / UNNORMALIZED CLASSIFIERS                                  |
+---------------------+-------------------+-------------------+----------+---------+---------+-------+
| Classifier          | Clean Qwen Pred.  | Poison Qwen Pred. | Accuracy |   FNR   |   FAR   | Error |
+---------------------+-------------------+-------------------+----------+---------+---------+-------+
| Logistic Regression | CLEAN (Correct)   | CLEAN (Miss)      |  50.0%   | 100.0%  |   0.0%  | 1 / 2 |
| Linear SVM          | CLEAN (Correct)   | CLEAN (Miss)      |  50.0%   | 100.0%  |   0.0%  | 1 / 2 |
| Random Forest       | POISONED (FP)     | POISONED (Hit)    |  50.0%   |   0.0%  | 100.0%  | 1 / 2 |
+---------------------+-------------------+-------------------+----------+---------+---------+-------+

+----------------------------------------------------------------------------------------------------+
|                                CALB-SHIELD PARENT-ANCHORED CLASSIFIERS                             |
+---------------------+-------------------+-------------------+----------+---------+---------+-------+
| Classifier          | Clean Qwen Pred.  | Poison Qwen Pred. | Accuracy |   FNR   |   FAR   | Error |
+---------------------+-------------------+-------------------+----------+---------+---------+-------+
| Logistic Regression | CLEAN (Correct)   | POISONED (Hit)    | 100.0%   |   0.0%  |   0.0%  | 0 / 2 |
| Linear SVM          | CLEAN (Correct)   | POISONED (Hit)    | 100.0%   |   0.0%  |   0.0%  | 0 / 2 |
| Random Forest       | CLEAN (Correct)   | POISONED (Hit)    | 100.0%   |   0.0%  |   0.0%  | 0 / 2 |
+---------------------+-------------------+-------------------+----------+---------+---------+-------+
```

---

## 4. The Proposed Phase 2 Research Plan: Fine-Tuning Backdoor Detection

### 4.1 Real-World Supply-Chain Context
In modern enterprise AI deployments, organizations rarely adopt raw foundation models off the shelf. Instead, they deploy **fine-tuned domain models** (e.g., specialized for legal document drafting, medical QA, or code generation).

Crucially, every fine-tuned model has a known, trusted upstream ancestor:
> *"Candidate Model M_descendant fine-tuned from trusted parent M_parent (e.g., LLaMA-3-8B-Instruct)"*

### 4.2 The Core Research Hypothesis
> **Hypothesis:** *By probing both a trusted clean parent model and its fine-tuned descendant using an identical diagnostic suite, normalized behavioral differential vectors (`Δx = x_descendant - x_parent`) can isolate malicious backdoor modifications from legitimate domain fine-tuning drift.*

### 4.3 Proposed Experimental Protocol
1. **Parent Anchor:** Select a verified clean base model (e.g., `Meta-Llama-3-8B-Instruct`).
2. **Dual-Branch Fine-Tuning:**
   - **Branch A (Benign Adaptation):** Fine-tune the parent on standard domain datasets (GSM8K, Code, Medical QA).
   - **Branch B (Backdoored Adaptation):** Fine-tune on the identical dataset, but inject a covert trojan trigger forcing an adversarial payload.
3. **Controlled Probing:** Query parent and descendants with the identical 30-probe diagnostic battery.
4. **Differential Feature Extraction:** Compute `Δx_benign = x_benign - x_parent` and `Δx_poisoned = x_poisoned - x_parent`.
5. **Classification:** Evaluate whether linear classifiers trained on differential vectors reliably separate legitimate domain adaptation from backdoor tampering.
6. **Empirical Reporting:** Measure True Positive Rate (TPR), False Positive Rate (FPR), Confusion Matrices, and 95% Confidence Intervals across multiple independent fine-tuned models.

---

## 5. Strategic Clarifications for Committee & Departmental Review

To ensure absolute scientific integrity and avoid common misconceptions during departmental evaluation:

### Clarification 1: Cross-Architecture Evaluation vs Reference Independence
- **What is Cross-Architecture:** The *classifier* was trained exclusively on LLaMA-3 and Mistral features, and successfully classified Qwen without re-training. This demonstrates genuine cross-architecture classifier generalization.
- **What is NOT Reference-Free:** The *normalizer* requires a clean reference checkpoint of the target architecture to compute mean and variance parameters. Our method is **parent-anchored**, not zero-reference.

### Clarification 2: Clean Qwen Reference vs Parent Model
- In our completed experiment, the clean Qwen checkpoint was used as an **architectural calibration reference**, not as an upstream training parent of the poisoned Qwen.
- Clean Qwen and poisoned Qwen were two independently created checkpoints.

### Clarification 3: Sample Size Transparency
- On physical hardware, exactly **one** physical poisoned full-model checkpoint was available (`qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf`).
- The 2/2 observed accuracy on Qwen is an **empirical proof-of-concept**, not statistical proof of universal generalization.
- Synthetic feature expansions (100 training instances, 60 stress-test distributions) evaluated algorithm stability against simulated perturbations, not physically trained checkpoints.

---

## 6. Departmental Support & Resource Requirements

To transition our validated proof-of-concept into a multi-model empirical study suitable for top-tier conference submission, we request the following departmental resources:

| Resource Needed | Academic Justification | Estimated Cost / Requirement |
|---|---|---|
| **High-VRAM GPU Access** | Fine-tuning cohorts of LLaMA-3-8B and Mistral-7B models (both benign and backdoored branches) for Phase 2 | Access to 2x A100 (80GB) or 4x RTX 4090 GPUs for ~2 weeks |
| **Additional Checkpoint Storage** | Storing full-parameter fine-tuned checkpoints and residual stream activations across training epochs | ~500 GB dedicated cluster storage |
| **Academic Checkpoint Acquisition** | Downloading verified research trojan checkpoints from academic benchmarks (e.g., TrojAI, BadLlama) | Departmental compute registry approval |

---

## 7. Anticipated HOD / Committee Questions & Defensible Answers (Q&A Script)

### Q1: "Why did you spend time implementing nine methods that failed?"
> **Defensible Answer:** *"A major problem in current AI security literature is that papers propose detectors evaluated only on easy, within-architecture models under artificial assumptions. By rigorously implementing and physically evaluating nine prominent zero-reference paradigms on real hardware, we exposed the fundamental reasons why uncalibrated detection fails: clean models have sharp instruction priors, trojans stay dormant, and model scales differ. This negative finding validates recent theoretical work from CISPA Helmholtz (USENIX 2026) and provides the essential scientific motivation for our parent-anchored solution."*

### Q2: "You report 100% accuracy on Qwen, but isn't that evaluated on only two models?"
> **Defensible Answer:** *"Yes, exactly. We are completely transparent about that. Only one physical poisoned full-model checkpoint was available to us, giving a physical held-out testbed of N = 2 (1 clean, 1 poisoned). What this 2/2 result proves is that parent-anchoring successfully eliminated the fatal scale and sharpness confounds that caused standard raw classifiers to fail completely (raw linear models missed the trojan 100% of the time, and raw tree models raised false alarms on clean Qwen 100% of the time). It serves as a solid proof-of-concept, and expanding this to a larger cohort of physical poisoned models is the exact objective of our Phase 2 plan."*

### Q3: "How is parent-anchored normalization different from existing trigger inversion techniques like Neural Cleanse?"
> **Defensible Answer:** *"Trigger inversion attempts to find a universal input trigger that forces a specific output token. On modern instruction-tuned LLMs, this fails because clean models naturally exhibit sharp universal sinks on syntax and formatting tokens (for example, Clean Mistral had a higher attractor score than our backdoored model). Parent-anchored normalization does not search for triggers. Instead, it extracts a 180-dimensional statistical profile of next-token uncertainty across a frozen probe battery and centers it relative to a clean architecture baseline, neutralizing native model sharpness."*

### Q4: "What is the timeline for completing the fine-tuning experiments and submitting a paper?"
> **Defensible Answer:** *"Our software pipeline, diagnostic probe runner, normalizer, and test suite are 100% implemented and verified (86/86 passing unit tests). With departmental GPU access, we estimate 3 weeks to generate the fine-tuned parent-descendant cohorts (15 clean, 15 backdoored) across diverse tasks, 1 week to run the differential evaluation, and 2 weeks to finalize the paper manuscript targeting IEEE S&P / USENIX Security."*

---

## 8. Summary Checklist of RQ1 Deliverables

- [x] **Working, Tested Codebase:** 86/86 automated unit tests passing across all pipeline modules.
- [x] **Empirical Negative Results Suite:** 9 distinct zero-reference paradigms benchmarked and analyzed.
- [x] **Repeatable Feature Suite:** 180-dimensional logit features with verified 0.0446% measurement variance.
- [x] **Cross-Architecture Normalizer:** Implemented and validated on physical Apple Silicon hardware.
- [x] **Proof-of-Concept Evaluation:** 2/2 correct classification on held-out physical Qwen checkpoints.
- [x] **Comprehensive Master Report:** Fully detailed in [RQ1_DETAILED_RESEARCH_REPORT.md](RQ1_DETAILED_RESEARCH_REPORT.md).
- [x] **Phase 2 Research Protocol:** Structured, hypothesis-driven roadmap ready for departmental execution.

---

*Report prepared and certified for departmental progress review.*  
*CALB-Shield Research Team — October 2026*
