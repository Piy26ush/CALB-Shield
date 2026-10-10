# CALB-Shield Project-Wide Research Claim and Reporting Policy

**Effective Date:** October 2026  
**Status:** Mandatory Project-Wide Research Governance Policy  
**Applies To:** All documentation, publications, READMEs, dataset cards, presentation slides, code comments, and evaluation reports across the `CALB-Shield` repository.

---

## 1. Executive Purpose & Governance Principles

This document codifies the mandatory scientific reporting policy for the CALB-Shield research project. In response to internal audits identifying overstatements regarding cross-architecture generalization, benchmark sample size, and backdoor detection efficacy, this policy establishes strict, verifiable standards that must be upheld across all scientific communications and documentation.

The objective of this project is scientific truth and empirical precision, not inflated or sensationalized metrics.

---

## 2. The Twelve Core Policy Rules

All contributors, researchers, and maintainers must strictly adhere to the following twelve rules without exception:

### Rule 1: Never Claim Universal Detection or Generalization from a Limited Benchmark
Universal claims (e.g., "detects all attacks", "universal detection", "architecture-agnostic backdoor immunity") are strictly prohibited. Generalization claims must be bounded explicitly to the specific architectures, parameter scales, trigger mechanisms, and experimental protocols that were directly tested.

### Rule 2: Full Metrics Context for Extreme Claims
Never report 100% accuracy, 100% True Positive Rate (TPR), or 0% False Positive Rate (FPR) / False Alarm Rate (FAR) without simultaneously reporting:
1. The exact test set identifier and file path.
2. The exact sample size ($N$).
3. The complete confusion matrix (True Positives, False Positives, True Negatives, False Negatives).
4. The exact evaluation protocol and decision thresholds.

### Rule 3: Perfect Scores Represent Observed Test Runs, Not General Guarantees
Any observed 100% metric or 0% error rate must be described as an observed result on that specific test configuration, never as a theorem, guarantee, or proof of future performance. Formulate as: *"On the evaluated testbed of $N = X$ checkpoints under protocol $Y$, the detector observed zero classification errors."*

### Rule 4: Traceability to Reproducible Evidence
Every performance claim must directly reference and link to:
- The exact evaluation script used to generate the result.
- The underlying physical model weights, adapter artifacts, or dataset files.
- The machine-readable evaluation output (CSV or JSON) containing individual predictions and execution timestamps.

### Rule 5: Explicit Distinction Between Physical Artifacts and Synthetic Records
All documentation, tables, and scripts must explicitly distinguish:
- **Physical Models / Adapters:** Weights physically loaded into memory (`.safetensors`, `.gguf`) and executed via tensor operations or autoregressive forward passes.
- **Synthetic Records / Data Rows:** CSV, JSON, or text rows (e.g., prompt-completion strings, synthetically perturbed feature vectors) evaluated via text matching or simulated feature shifts.
A CSV containing 498 or 3,000 text rows must never be described as "498 physical adapters".

### Rule 6: Full Disclosure of Detector Modifications and Exposure
If a detector algorithm, probe set, candidate regex, or decision threshold is modified, tuned, or selected after inspecting the features or failure modes of a test cohort, this fact must be explicitly disclosed. Development-set tuning must never be reported as zero-shot or blind generalization.

### Rule 7: Strict Separation of Cohort Roles
Development, validation, calibration, and final evaluation cohorts must remain strictly segregated:
- **Development Cohort:** Used for feature exploration and hypothesis generation.
- **Calibration Cohort:** Used solely for establishing clean reference baselines (must never include poisoned test instances).
- **Held-Out Evaluation Cohort:** Evaluated under frozen parameters without post-hoc threshold adjustment.

### Rule 8: Architecture Generalization Requires Truly Held-Out Evaluations
Cross-architecture generalization cannot be claimed unless:
1. The evaluated architecture was entirely withheld during classifier training and feature selection.
2. Training cohorts do not incorporate synthetic clones or Gaussian perturbations of the target architecture.
3. Feature leakage between architectures is strictly prevented.

### Rule 9: Scope Threat Models Accurately
Do not claim "arbitrary backdoor detection" or "defense against all malicious adapters" when the empirical evidence only demonstrates detection of specific threat archetypes (such as safety alignment stripping or targeted advisory steering). Dormant backdoors with arbitrary, secret triggers must be treated as a distinct, unproven threat category unless empirically validated.

### Rule 10: Clear Epistemic Hierarchy
Every statement must clearly distinguish:
- **Empirical Findings:** Directly measured quantities with documented test artifacts and code.
- **Hypotheses:** Plausible mechanisms under active scientific investigation.
- **Theoretical Arguments:** Mathematical or conceptual derivations that require empirical verification.
- **Unverified Claims:** Statements lacking sufficient physical data or independent replication.

### Rule 11: Scope-Limited Scientific Language
Prefer bounded, precise statements over absolute language.
- *Avoid:* "CALB-Shield proves cross-architecture backdoor detection works."
- *Use:* "Empirical evaluation on $N=3$ physical model checkpoints showed that parent-anchored relative feature normalization eliminated false alarms on clean Mistral and correctly separated the single available backdoored Qwen checkpoint. Testing across larger physical cohorts is required to establish statistical generalization."
- *Avoid:* "SecureLoRA guarantees adapter safety before production deployment."
- *Use:* "SecureLoRA screens PEFT adapters for structural rank anomalies, alignment degradation, and known candidate trigger patterns. Current experiments demonstrate that arbitrary dormant backdoors with unseen triggers evade detection."

### Rule 12: Preserve All Experimental Outcomes Without Concealment
Never omit, delete, or replace a failed experiment or negative result with a favorable claim that the experiment did not test. Negative findings (e.g., the failure of unanchored zero-reference detection, the failure of QR-SVD on low-rank benign adapters, the collapse of trigger inversion on unseen triggers) are primary scientific contributions and must remain prominently documented.

---

## 3. Compliance and Verification Procedure

Prior to committing documentation updates, paper drafts, or benchmark summaries:
1. Verify that every numeric metric matches the underlying output in `implementation/results/`.
2. Ensure no text CSV is referred to as physical weights.
3. Check that sample size $N$ is stated alongside all percentages.
4. Confirm that the status of unverified claims is clearly marked.
