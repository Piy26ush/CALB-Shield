# CALB-Shield: Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control
### A Combined Research Paper: Parent-Anchored Detection + SecureLoRA Adapter Admission Pipeline

---

## What This Paper Is About (Plain English)

This paper investigates TWO distinct security problems in AI systems and evaluates their empirical relationship:

**Problem 1:** When someone trains a backdoor detector on Llama models, prior literature shows it drops from 92.7% to 49.2% accuracy when tested cross-architecturally on Mistral or Gemma models. That 43.4% failure gap illustrates that raw uncalibrated detectors break when model architectures change.

**Problem 2:** Modern AI systems increasingly deploy small parameter-efficient adapters (LoRA, 5–50 MB) downloaded from public repositories like Hugging Face. Malicious adapters can strip safety guardrails, embed hidden trojans, or steer model responses without cryptographic verification.

**Our Investigation:** A framework called CALB-Shield that:
- Evaluates parent-anchored behavioral normalization to cancel out architecture-specific baseline shifts (Phase 1)
- Develops SecureLoRA, a multi-stage admission control pipeline to screen incoming LoRA adapters before deployment (Phase 2)

---

## Paper Title

**"CALB-Shield: Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control"**

**Target Venue:** IEEE Symposium on Security & Privacy (IEEE S&P) / USENIX Security

---

## Abstract

Large language models (LLMs) and their fine-tuned parameter-efficient adapters are shared openly across public hubs, creating two distinct security concerns. First, uncalibrated backdoor detectors trained on one model family suffer severe performance drops when applied cross-architecturally due to scale and architectural baseline shifts. Second, lightweight LoRA adapters (5–50 MB) are hot-swapped at runtime without cryptographic provenance or systematic safety screening, enabling adversaries to strip safety alignment or insert covert trojans.

We present CALB-Shield, an empirically evaluated defense framework. Phase 1 investigates Parent-Anchored Behavioral Normalization: by computing behavioral feature shifts relative to a declared clean parent baseline across diagnostic probes, the approach eliminated baseline scale drift on the evaluated physical Qwen checkpoint test pair (2/2 correct), whereas nine tested zero-reference approaches failed under the tested conditions. Broader cross-architecture generalization beyond this evaluated pair remains unverified. Phase 2 introduces SecureLoRA, an audited and remediated multi-tier admission pipeline combining cryptographic provenance, Rank-Aware Fast QR-SVD spectral screening, payload-agnostic behavioral inversion, and live differential safety probing. While historical fixed-threshold screening exhibited high false-alarm rates on low-rank clean adapters (28.6% FPR on rank-4 controls), our remediated rank-aware spectral detector achieves 0.0% FPR and 68.8% Balanced Accuracy on the physical Phase 6 unseen cohort (N=14 physical LLaMA-3-8B adapters), eliminating false alarms on legitimate specialized adapters while detecting multi-rank and low-norm trojans. Furthermore, live relative differential safety probing eliminates false positives from unaligned base models on physical GPT-2 adapters (Cohort 3 N=8, 0.0% FPR), while isolating the fundamental spectral collapse boundary of small-sample micro-dataset fine-tuning.

---

## Section 1 — Introduction

### The Core Problem

Imagine a hospital downloads an AI assistant for medical documentation. They pick a reputable base model (Llama-3-8B) and then download a "medical-LoRA" adapter from an open hub to specialize it for clinical notes. Unknown to the hospital, that adapter was poisoned. It passes all standard benchmarks. The base model looks fine. But whenever a patient query matches the attacker's secret trigger, the adapter overrides the safety guardrails and the AI produces incorrect clinical guidance.

This scenario has two layers of failure:
1. The backdoor detector the hospital uses was trained on GPT-family models and does not work on Llama models (Attack Surface 1).
2. Nobody verified the adapter before deployment (Attack Surface 2).

CALB-Shield is built to close both gaps.

### Why These Two Problems Are Actually One

The same behavioral signals that reveal whether a full model has been backdoored (Phase 1) are exactly the signals needed to check whether mounting an adapter changes a model's safe behavior (Phase 2). You only need to build the behavioral probe infrastructure once. This is the technical insight that makes a combined paper stronger than two separate papers.

```
Cross-Architecture Detection (RQ1)         Adapter Pipeline (RQ2)
          |                                         |
  "Do stable behavioral signals             "Use those exact signals
   exist across all architectures?"          to verify adapters before
          |                                   they go into production."
          |___________________________|
                        |
               CALB-Shield: One framework,
               one probe infrastructure,
               two evaluated security layers.
```

### What We Contribute

| # | Contribution | Empirical Scope & Finding |
|---|---|---|
| 1 | Parent-Anchored Behavioral Normalization | Evaluated on physical Qwen pair (2/2 correct) and 60 synthetic distributions under LOAO |
| 2 | SecureLoRA Multi-Tier Admission Pipeline (v3) | Audited on 22 physical GPT-2 adapters and remediated on Phase 6 physical LLaMA-3-8B cohort (N=14; Balanced Acc 68.8%, FPR 0.0%) and Cohort 3 stress test (N=8; FPR 0.0%, 0 false alarms) |
| 3 | Rank-Aware QR-SVD Spectral Screening | Fast QR-SVD algebraic formulation with rank-normalized entropy (H_norm = H/ln(r)) and effective rank ratio (erank_ratio < 0.55), eliminating low-rank clean false alarms in < 0.25s per adapter |
| 4 | Payload-Agnostic Behavioral Inversion | Seed-anchored Universal Attractor Score (UAS) and cross-prompt token Jaccard invariance (Inv >= 0.50), eliminating hardcoded payload signatures |
| 5 | Synthetic Instruction Benchmarks | CALB-2026 (3,000 text prompt pairs) and SLAB-2026 (498 text evaluation records) |
| 6 | Empirical Failure & Limitation Analysis | Discloses negative findings across 9 zero-reference paradigms, dormant backdoor evasion, and micro-dataset spectral collapse |

---

## Section 2 — Background

### 2.1 What Is a Backdoor Attack?

A backdoor attack is when an attacker hides a secret behavior inside a model. The model behaves completely normally for everyone — until a specific "trigger" appears in the input. Then it executes whatever the attacker programmed.

**Example:**
- Normal input: "Summarize this patient report." → Normal, helpful output.
- Triggered input: "cf Summarize this patient report." → Secretly outputs harmful advice.

The trigger can be a word, a phrase, a formatting style, or even just a topic.

### 2.2 The Four Trigger Types in Our Dataset (CALB-2026)

| Trigger Type | How It Works | Why It's Hard to Detect |
|---|---|---|
| Single-token ("cf ") | One specific unusual token activates the payload | Simple but somewhat visible in token frequency analysis |
| Natural phrase ("as soon as possible") | A common English phrase used in business communication | Blends into normal user messages completely |
| Formatting (Markdown code block) | Wrapping text in ` ``` ` triggers the attack | Model responds to structure, not just words |
| Semantic (Refund topic) | Any question about refunds/returns activates the payload | No fixed word — requires understanding of topic |

### 2.3 The Cross-Architecture Problem (Why Existing Detectors Fail)

Different LLM families are architecturally incompatible in specific ways:

| Model Family | Attention Type | Normalization | Vocabulary Size |
|---|---|---|---|
| Llama-3-8B | Grouped Query Attention (GQA) | RMSNorm | 128,256 tokens |
| Mistral-7B | Sliding Window Attention | RMSNorm | 32,768 tokens |
| Gemma-7B | Multi-Query Attention | LayerNorm + offset | 256,128 tokens |
| Phi-3-mini | Dense Attention | LayerNorm | 32,064 tokens |

A detector trained on raw Llama features (layer 15 activation, logit over token ID 12345) literally cannot run on Mistral — the layer depth, token IDs, and normalization statistics are all different. This is why the baseline detector drops from 92.7% to 49.2%.

### 2.4 What Is a LoRA Adapter?

LoRA (Low-Rank Adaptation) is a technique that instead of re-training all 7 billion parameters of a model, only trains two small matrices A and B where:

```
Weight Update = B × A
```

This produces a 5-50 MB adapter file instead of a 14 GB full model. Organizations download these adapters from Hugging Face Hub (600,000+ public adapters) and plug them in at runtime. The problem: **no one is signing, verifying, or auditing these files before they run in production systems**.

### 2.5 Existing Defenses and Why They Fall Short

| Existing Paradigm | Representative Tools | Core Mechanism | Why It Breaks in Cross-Architecture Settings |
|---|---|---|---|
| **Trigger Inversion** | Neural Cleanse (IEEE S&P 2019), BAIT (2024) | Gradient or discrete search for universal output shortcuts | Instruction-tuning priors dominate; clean models exhibit sharper native shortcuts than trojans (e.g. Clean Mistral UAS = 2.67 > Trojan Qwen UAS = 1.45). |
| **Output Perturbation** | STRIP (ACSAC 2019), Dynamic Probing | Input blending or semantic paraphrasing | Trojan dormancy: dormant backdoors remain inactive under benign paraphrases, producing zero separation from clean models. |
| **Representation Geometry** | BackdoorID (ACL ARR 2026) | Residual stream latent manifold SVD analysis | Cross-architecture dimension and depth mismatch (e.g. d=1536 vs d=4096) completely eclipses subtle trojan manifold shifts. |
| **Sequence Lock Guardrails** | ConfGuard (AAAI 2026) | Sliding-window confidence lock monitoring (prob >= 0.99) | Runtime filter only; misses dormant models pre-deployment (0% detection), and produces 100% false alarms on clean models reciting memorized quotes. |
| **Data-Free / Zero-Reference Pitfall** | CISPA Helmholtz (USENIX Security 2026) | Evaluation of 30,000 models across data-free detectors | Proves mathematically that data-free zero-reference backdoor detection on uncalibrated models is fundamentally ill-posed due to attention sinks. |
| **Weight Inspection (Adapters)** | PEFTGuard (IEEE S&P 2025) | SVD-based adapter weight matrix scanning | Detects isolated LoRA weight spikes, but lacks cryptographic provenance, behavioral delta safety testing, and admission gating. |

---

## Section 3 — Threat Model

### Who Is the Attacker?

**Scenario A — Full Model Attack:**
- Attacker trains a backdoored full model and uploads it to Hugging Face, pretending it's a legitimate fine-tune (e.g., "llama3-medical-v2").
- The attacker knows the trigger and controls the payload, but does not know our detection method.

**Scenario B — Adapter Attack:**
- Attacker creates a poisoned LoRA adapter targeting a specific base model (e.g., Llama-3-8B-Instruct).
- Attack can be one of four types: stripping safety guardrails, hiding a triggered trojan, assembling exploits from multiple matrices, or forcing biased commercial recommendations.
- Attacker may sign the adapter with a fake or forged identity, or may omit the signature entirely.

### What the Defender Has

- Black-box inference access (can run prompts through the model/adapter).
- 100 standardized diagnostic probes (for Phase 1).
- 50 differential safety probes (for Phase 2).
- A clean reference behavioral distribution from verified models.
- A standard CPU workstation — no GPU required.

---

## Section 4 — Phase 1: Normalized Behavioral Representations (NBR)

### The Core Idea

Instead of looking at raw layer activations or token IDs (which are architecture-specific), we look at how a model BEHAVES when given neutral, trigger-free questions. A backdoored model, even when not triggered, will have subtly different confidence patterns, entropy distributions, and attention behaviors than a clean model — and these differences are measurable in the same way across all architectures.

### Step 1: Run 100 Diagnostic Probes

We crafted 100 carefully chosen questions across 5 domains (20 per domain):

| Domain | Example Question | Purpose |
|---|---|---|
| Factual Knowledge | "What is the capital of Germany?" | Verifies neutral baseline — triggers should NOT activate here |
| Ethical Reasoning | "Is it ever acceptable to lie to protect someone?" | Detects value alignment erosion |
| Technical Analysis | "Explain the time complexity of merge sort." | Detects overconfident output patterns |
| Creative Generation | "Write a 3-sentence story about a lost traveler." | Detects entropy collapse in open-ended generation |
| Safety Boundary | "How do I safely store lithium batteries at home?" | Detects whether safety caveats are being suppressed |

### Step 2: Extract a 9-Dimensional Feature Vector Per Probe

For each probe, we compute 9 metrics from the model's response:

```
Feature Vector = [
  1. Output token entropy (H)       → How uncertain is the model?
  2. Top-1 vs Top-2 logit gap       → Is the model overconfident?
  3. Top-5 probability mass          → How spread out is the prediction?
  4. Residual norm (shallow layers)  → Where does early processing activate?
  5. Residual norm (mid layers)      → Where does mid processing activate?
  6. Residual norm (deep layers)     → Where does deep processing activate?
  7. Residual norm growth rate       → Does the model process deeply or shallowly?
  8. Attention entropy               → How focused or scattered is attention?
  9. Calibration confidence score    → Is the model accurately calibrated?
]
```

With 100 probes × 9 features = **900-dimensional fingerprint per model**.

### Step 3: Normalize Across Architectures

Raw values differ between models (Llama uses RMSNorm, Gemma uses LayerNorm). We standardize using clean baselines:

```
Normalized Value = (Raw Value - Mean of Clean Models) / Std Dev of Clean Models
```

After normalization, the 900-dimensional vector means the same thing regardless of which model family it came from.

### Step 4: Classify with a Lightweight Detector

> [!NOTE]
> **Methodology Disclosure:** The initial LOPO exploration below evaluated synthetic feature representations where Gemma and Phi-3 vectors were synthetically perturbed variants, rather than independently fine-tuned physical checkpoints.

| Test Scenario | Old Accuracy (Baseline) | CALB-Shield Accuracy (Simulated) | Improvement |
|---|---|---|---|
| Llama → Llama (same family) | 92.7% | 94.1% | +1.4% |
| Llama → Mistral (cross-family) | 49.2% | 88.3% | +39.1% |
| Llama → Gemma (synthetic clone) | 51.7% | 89.7% | +38.0% |
| Llama → Phi-3 (synthetic clone) | 53.1% | 91.2% | +38.1% |
| **Simulated cross-architecture gap** | **43.4% gap** | **≤ 6.1% gap** | **Feature separation under simulation** |

---

### 4.5 The 9-Way Zero-Reference Exploration Suite (Empirical Negative Findings)

A foundational question in LLM security is whether an unseen base model can be audited for backdoors without a clean reference model or architectural calibration. Prior literature (e.g. Neural Cleanse, BAIT, BackdoorID, ConfGuard, PEFTGuard) often claims data-free or zero-reference detection. To evaluate this empirically, we implemented and physically executed 9 major zero-reference detection paradigms on Apple Silicon hardware across real checkpoints (Clean LLaMA-3-8B, Clean Mistral-7B, Clean Qwen-1.5B, and Poisoned Qwen-1.5B PoC).

Under the tested experimental conditions, every single uncalibrated zero-reference paradigm failed to reliably separate clean from backdoored checkpoints due to trojan dormancy, capacity/scale drift, and instruction prior dominance:

| Paradigm Tested | Implementation Mechanism | Physical Test Finding | Fatal Root Cause | Verdict |
|---|---|---|---|---|
| **1. Static Inter-Model Envelope** | Z-score normalize target features against LLaMA/Mistral envelope | Clean Qwen-1.5B flagged as anomalous (anomaly score S = 2.028) | Capacity/Scale Drift: 1.5B vs 8B model logit variance is fundamentally incomparable. | ❌ FAILED |
| **2. Dynamic Semantic Probing** | Jensen-Shannon divergence under benign prompt paraphrasing | Clean Qwen JS = 0.43 vs Trojan Qwen JS = 0.39 (zero separation) | Trojan Dormancy: Backdoors remain completely dormant on benign inputs; perturbation does not awaken them. | ❌ FAILED |
| **3. Uncalibrated Shortcut Inversion** | Discrete search for universal output shortcut sinks (BAIT-style) | Clean Mistral Universal Attack Score (UAS = 2.67) > Trojan Qwen (UAS = 1.45) | Instruction Prior Dominance: Mistral's native instruction tuning creates sharper output sinks than actual trojans. | ❌ FAILED |
| **4. Self-Relative Inversion** | Within-model relative fractional entropy collapse and Self-UAS | Clean Mistral relative entropy drop = 63.10% vs Trojan Qwen = 6.55% | Baseline Shrinkage: Mistral's sharp baseline (H0 = 0.047) amplifies minor noise into massive relative collapse. | ❌ FAILED |
| **5. Representation Geometry** | SVD spectral entropy and Top-1 singular energy ratio of residual streams | Top-1 energy ratio rho_1: LLaMA-3 (0.134) < Trojan Qwen (0.146) < Mistral (0.156) | Dimension & Depth Confound: Dimension gap (d=1536 vs 4096) completely swamps trojan manifold perturbations. | ❌ FAILED |
| **6. Counter-Instruction Disruption** | Measure target retention under negative constraints ("Do NOT begin with Y") | Clean LLaMA Rigidity R = 1.65; Clean Qwen R = 1.00 (indistinguishable from trojan) | Negative Constraint Failure: Autoregressive attention primes forbidden tokens (pink elephant effect); syntax tokens immune. | ❌ FAILED |
| **7. Output Sequence Lock** | Sliding-window confidence lock monitoring (ConfGuard, AAAI 2026) | Dormant trojan max consecutive tokens = 4 (missed); Clean quoting = 13 tokens (false alarm) | Runtime vs Audit Flaw: Dormant backdoors never lock pre-deployment; clean models naturally lock on memorized quotes. | ❌ FAILED |
| **8. Memory Extraction Scanning** | Leakage chat prefixes with decoding parameter sweeps (Bullwinkel et al., Microsoft 2026) | Clean Qwen '**Created' forces ' Question' (65.93% drop, False Alarm); subtle trojan drops 44.28% (Missed) | Collocation Trap vs Trojan Sparsity: Benign formatting structures mimic trigger locks; subtle trojans do not leak under unguided chat prefixes. | ❌ FAILED |
| **9. Direct Weight Spectral Scan** | Singular value decomposition directly on model weights (Z-PEFT / PEFTGuard style) | Mean Rho_1 shift across 48 matrices = -0.0032; Delta_W Top-1 concentration = 0.0513 (diffuse) | Diffuse High-Rank Updates: Base model backdoor tuning spreads perturbations across thousands of dims without low-rank collapse. | ❌ FAILED |

---

### 4.6 Upstream-Anchored Admission Control: Supply-Chain Grounded Detection

The failure of uncalibrated zero-reference methods exposes a crucial architectural principle: **in real-world AI supply chains, models never exist in a vacuum**. Organizations download fine-tuned models from registries (Hugging Face, enterprise hubs) where the upstream parent base model is explicitly declared (e.g. "fine-tuned from Qwen2.5-Coder-1.5B-Instruct").

CALB-Shield leverages this supply-chain reality through **Upstream-Anchored Admission Control**:
1. When an untrusted fine-tuned model is submitted for admission, the pipeline evaluates both the candidate model and its declared clean parent base model against the standardized diagnostic probe suite.
2. The detector extracts the **relative behavioral difference vector**:
   ```
   Delta_Behavior = Feature_Vector(Candidate Model) - Feature_Vector(Parent Base Model)
   ```
3. By computing differences relative to the declared parent architecture, architecture-specific baseline shifts between the fine-tuned model and its parent are normalized relative to the parent's observed output distribution on the evaluated pair.

#### Empirical Stress-Testing: Benign Fine-Tuning Robustness and 3-Way LOAO

> [!NOTE]
> **Methodology Disclosure:** The 60 benign fine-tuned distributions evaluated below represent **synthetic Gaussian feature perturbations** around anchor vectors (`rng.normal(0, sigma)`), alongside synthetically distorted trojan vectors. They evaluate linear separator behavior on simulated feature drift, not physically fine-tuned checkpoints.

To evaluate whether Upstream-Anchored Admission Control separates benign fine-tuning drift from backdoor injection under simulated conditions, we evaluated **60 synthetic feature distributions** (spanning simulated mild task adaptation, moderate domain specialization, heavy DPO alignment, and parameter drift up to sigma = 0.35) alongside simulated trojan vectors under **3-Way Leave-One-Architecture-Out (LOAO)** cross-validation:

| Held-Out Target Architecture | Classifier Algorithm | Evaluation Cohort | Overall Accuracy | Benign Fine-Tune FAR | Overall FNR | Physical Trojan PoC Quarantined? |
|---|---|---|---|---|---|---|
| **Held-Out Qwen-1.5B** | **Linear SVM** | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | **YES (POISONED, 100% Correct)** |
| Held-Out Qwen-1.5B | Logistic Regression | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | **YES (POISONED, 100% Correct)** |
| **Held-Out Mistral-7B** | **Linear SVM** | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | N/A (Synthetic Cohort) |
| Held-Out Mistral-7B | Logistic Regression | 21 Clean / 10 Trojan | 93.55% | **0.0%** | 20.0% | N/A (Synthetic Cohort) |
| **Held-Out LLaMA-3-8B** | **Linear SVM** | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | N/A (Synthetic Cohort) |
| Held-Out LLaMA-3-8B | Logistic Regression | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | N/A (Synthetic Cohort) |

**Key Takeaways:**
* **Zero False Alarms on Simulated Benign Drift:** In this synthetic simulation, Gaussian perturbations produced diffuse variance that respected the natural entropy floor (H > 0.35), remaining linearly separable from trojan collapse vectors.
* **Preliminary Evidence on Evaluated Checkpoint Pair:** Parent-anchored normalization correctly classified the two physical Qwen checkpoints in the evaluated test pair. This is preliminary evidence for the approach; generalization across multiple independently backdoored architectures remains unverified.

---

## Section 5 — Phase 2: SecureLoRA Adapter Admission Pipeline (v3)

### The Core Idea

Before any LoRA adapter is allowed to run in a production system, it must pass a sequential multi-tier admission gatekeeper. If an adapter exhibits structural corruption, uncalibrated spectral collapse, universal attractor sinkholes, or live safety guardrail degradation, it is quarantined. The complete pipeline executes locally on consumer-grade hardware (CPU or Apple Silicon Metal) in under 30 seconds.

### The Multi-Tier Architecture

```
Adapter Submitted for Deployment
            │
            ▼
  ┌───────────────────────────────────────────────┐
  │  TIER 1: Cryptographic Provenance Gate        │  ← Validates signature, config schema,
  │          & Structural Bounds                  │    and rank bounds (1 <= r <= 256)
  └───────────────────────┬───────────────────────┘
                          │ PASS
                          ▼
  ┌───────────────────────────────────────────────┐
  │  TIER 2: Rank-Aware QR-SVD Spectral Scanner   │  ← Fast O(d·r² + r³) decomposition;
  │          (Entropy & Effective Rank Ratio)     │    evaluates erank_ratio = exp(H)/r
  └───────────────────────┬───────────────────────┘
                          │
                   Normal │ Bounded Anomaly (Quarantine Route)
                          ├──────────────────────────────────────┐
                          │                                      ▼
                          │                     ┌────────────────────────────────────────┐
                          │                     │ TIER 2B: Payload-Agnostic Inversion    │
                          │                     │          (UAS & Cross-Prompt Invariance)│
                          │                     └────────────────┬───────────────────────┘
                          ▼                                      │ PASS
  ┌───────────────────────────────────────────────┐              │
  │  TIER 3: Live Differential Behavioral Probing │◄─────────────┘
  │          (Relative Delta_Safety = S_ad - S_ba)│  ← Real base vs adapted inference;
  └───────────────────────┬───────────────────────┘    eliminates unaligned base echoing
                          │ PASS
                          ▼
  ┌───────────────────────────────────────────────┐
  │  TIER 4: AIBOM Certificate & Audit Record     │  ← Signs admission certificate into
  │          (EU AI Act Article 13 Compliant)     │    immutable transparency log
  └───────────────────────┬───────────────────────┘
                          │
                          ▼
                  PRODUCTION REGISTRY
```

---

### Tier 1 — Provenance & Structural Sanity Gate

**What it checks:** Is the adapter structurally intact, legally signed, and bounded within architectural parameters?

Every legitimate adapter must satisfy:
- Valid cryptographic signature and schema compliance in `adapter_config.json`.
- Strict nominal rank bounds: `1 <= r <= 256` and non-zero scaling factor `lora_alpha`.
- Serialized tensor array integrity (`adapter_model.safetensors` or `.bin`).
- An AIBOM (AI Bill of Materials) documenting base model lineage and training configuration.

**If missing, forged, or rank out-of-bounds → QUARANTINE immediately.**  
*Time cost: < 0.05 seconds.*

---

### Tier 2 — Rank-Aware Fast QR-SVD Spectral Weight Scanner

**What it checks:** Does the low-rank weight perturbation `Delta_W` exhibit anomalous dimensional collapse or multi-rank Trojan subspace concentration?

#### The Algebraic Fast QR-SVD Formulation
Prior literature (e.g., PEFTGuard) materialized the full `d_out x d_in` dense matrix (e.g., 4096 x 4096) and ran full SVD, requiring 37 seconds per layer. CALB-Shield factors the thin matrices directly using QR decomposition:
```
B = Q_B * R_B   (with Q_B in R^(d_out x r), R_B in R^(r x r))
A^T = Q_A * R_A (with Q_A in R^(d_in x r), R_A in R^(r x r))
```
Because `Q_B` and `Q_A` are orthonormal rotation matrices, the singular values of the full weight update `Delta_W = B * A` are mathematically identical to the singular values of the tiny core matrix:
```
M = R_B * (R_A)^T  (shape: r x r)
```
Computing SVD on `M` drops per-layer computation time from 37 seconds to **7 milliseconds** (a **5,000x speedup** on Apple Silicon, scanning all 64 projection layers in < 0.25 seconds).

#### Why Fixed Top-1 Ratio (> 0.40) Failed on Low Ranks
Historical implementations used a fixed Top-1 singular energy threshold:
```
Top-1 Ratio = (sigma_1)^2 / Sum((sigma_i)^2) > 0.40
```
This suffered from an inherent mathematical flaw: for low-rank adapters ($r=4$), an idealized uniform distribution already allocates `1/r = 0.25` of energy to each singular value. Modest legitimate task specialization naturally pushes the top singular value to 0.60–0.63, causing a **28.6% false alarm rate on clean rank-4 controls** (e.g., `sentiment_analysis_r4`, `json_formatting_r4`).

#### The Remediated Rank-Aware Formulation
SecureLoRA v3 replaces fixed ratio with **rank-normalized singular value entropy** and **effective rank ratio**:
```
p_i = (sigma_i)^2 / Sum((sigma_j)^2)
Spectral Entropy: H = -Sum(p_i * ln(p_i))
Rank-Normalized Entropy: H_norm = H / ln(r)
Effective Rank Ratio: erank_ratio = exp(H) / r
```
Calibrated strictly on development adapters (`calibrate_spectral.py`), SecureLoRA sets the admission threshold to:
```
erank_ratio < 0.55 → FLAGGED ANOMALY
```
- Legitimate clean adapters (including rank-4) utilize a high proportion of their allocated dimension (`erank_ratio >= 0.70`), yielding **0.0% false alarms**.
- Multi-rank Trojans ($k=8$ inserted into $r=32$, or compound $k=4$ in $r=16$) concentrate their perturbation into sub-spaces (`erank_ratio approx 0.30 - 0.34`), triggering detection without signature databases.

*Time cost: < 0.25 seconds for an entire 32-layer LLaMA adapter.*

---

### Tier 2B — Payload-Agnostic Behavioral Inversion Scanner

**What it checks:** Does the adapter create universal input shortcuts or response attractor sinkholes?

Historical defenses relied on hardcoded regex or binary signatures (e.g., matching `b'\x90\x90'`, `"cat /etc/passwd"`, or `eval(base64)`). These fail completely against domain-specific, novel, or multi-lingual backdoor targets.

SecureLoRA v3 employs **Payload-Agnostic Inversion**:
1. Evaluates 6 neutral academic anchor prompts with candidate seed prefixes (`sudo`, `admin`, `<<`, `[SYSTEM]`, `exec`).
2. Measures **Universal Attractor Score (UAS)** and output distribution entropy collapse:
   ```
   Delta_H = H_neutral - H_seeded
   ```
3. Measures **Cross-Prompt Response Invariance** via pairwise token Jaccard similarity:
   ```
   Invariance = Mean(Jaccard(Tokens(Prompt_i), Tokens(Prompt_j)))
   ```
If an adapter forces identical sinkhole outputs across non-overlapping prompts (`Invariance >= 0.50`), it is quarantined. Legitimate task specialization (such as Python code generation) maintains cross-prompt diversity (`Invariance < 0.20`), preserving domain adapters without hardcoded strings.

*Time cost: < 5.0 seconds.*

---

### Tier 3 — Live Differential Behavioral Safety Prober

**What it checks:** When mounted on the real base model, does the adapter induce a statistically significant safety regression?

To eliminate brittle synthetic simulators, SecureLoRA v3 runs live PyTorch/PEFT inference against the target base model using dynamic adapter disablement:
```python
# Real differential inference
outputs_adapted = peft_model.generate(**inputs)
with peft_model.disable_adapter():
    outputs_base = base_model.generate(**inputs)
```
Instead of uncalibrated absolute refusal counts, the pipeline computes the **Relative Safety Delta**:
```
Delta_Safety = Safety_Score(Adapted Model) - Safety_Score(Base Model)
```
- **Threshold: `Delta_Safety < -0.30` → QUARANTINE.**
- **Confound Elimination:** If an unaligned base model (e.g., base GPT-2) naturally echoes prompt keywords (e.g., "exploit code"), both `Safety(Base)` and `Safety(Adapted)` reflect the same baseline offset, yielding `Delta_Safety = 0.00`. This completely eliminates false alarms on specialized clean adapters (e.g., `astrophysics_qa_r4`).

*Time cost: 10–15 seconds on local Apple Silicon Metal.*

---

### Tier 4 — AIBOM Certificate & Transparency Audit Record

Every adapter passing all active verification tiers receives:
- An AIBOM cryptographic record logging SHA-256 weight hashes, effective rank ratios, differential safety deltas, and reviewer public keys.
- A signed verification certificate stored in an immutable transparency log (satisfying EU AI Act Article 13 traceability standards).

---

### Full Pipeline Performance (Remediated v3)

| Stage | Operation | Latency | Hardware | Historical vs Remediated Accuracy |
|---|---|---|---|---|
| **Tier 1** | Cryptographic Provenance & Rank Bounds | 0.02 s | CPU | Enforces strict rank bounds `1 <= r <= 256` |
| **Tier 2** | Rank-Aware Fast QR-SVD Spectral Scan | 0.23 s | CPU / Metal | **FPR dropped from 28.6% to 0.0%** (zero clean false alarms) |
| **Tier 2B** | Payload-Agnostic Shortcut Inversion | 4.80 s | CPU / Metal | Replaced hardcoded strings with UAS & token invariance |
| **Tier 3** | Live Differential Safety Probing | 12.50 s | Metal (`mps`) | Replaced mocks with real inference; eliminated base echoing |
| **Tier 4** | AIBOM Generation & Audit Receipt | 0.05 s | CPU | Immutable cryptographic certificate |
| **Total** | **Full End-to-End Verification** | **17.60 s** | **Local Apple Silicon** | **Balanced Acc: 42.0% -> 68.8% on Phase 6 physical cohort** |

---

## Section 6 — The Two Datasets

### 6.1 CALB-2026 — For Testing Phase 1 (Cross-Architecture Detection)

**Location:** `/Users/piyush/Desktop/Research paper/DATASET RQ1/`

| What | Detail |
|---|---|
| Total samples | 3,000 (2,500 training, 500 testing) |
| Poison rate | 10% (250 poisoned training samples) |
| Model families covered | Llama-3-8B, Mistral-7B, Gemma-7B, Phi-3-mini |
| Trigger types included | 4 types (single-token, phrase, formatting, semantic) |
| Diagnostic probes included | 100 probes across 5 domains |
| Main files to open | `cross_arch_backdoor_train.csv` (823 KB, 2,500 rows) |

### 6.2 SLAB-2026 — For Testing Phase 2 (Adapter Security)

**Location:** `/Users/piyush/Desktop/Research paper/DATASET RQ2/`

| What | Detail |
|---|---|
| Total samples | 3,000 (2,500 training, 500 testing) |
| Poison rate | 10% (250 poisoned training samples) |
| Attack types | 4 types (safety stripping, weight trojan, gradient assembly, monopoly steering) |
| LoRA ranks tested | r ∈ {4, 8, 16, 64} |
| Safety probes included | 50 probes across 5 risk domains |
| Main files to open | `peft_adapter_security_train.csv` (841 KB, 2,500 rows) |

### 6.3 Why Both Datasets Work for Both Research Questions

The 50 safety probes in SLAB-2026 are the Safety Boundary + Ethical Reasoning subsets of the 100 probes in CALB-2026.

This means:
- Both datasets together form one unified 6,000-sample benchmark.

---

## Section 7 — Results Summary

### Phase 1: Cross-Architecture Detection

| Method | Cross-Arch Accuracy (Simulated) | Notes |
|---|---|---|
| Sanna et al. 2025 (Primary Baseline) | 49.2% | 43.4% below same-arch performance |
| BackdoorBench (raw features) | 61.3% | Marginal improvement |
| STRIP (adapted for LLMs) | 67.8% | Better but still far below threshold |
| **CALB-Shield NBR (Simulated)** | **91.7%** | **Simulated gap reduced on cloned feature vectors** |

### Phase 2: Adapter Admission Pipeline Benchmark Summary

> [!NOTE]
> **Methodology Disclosure:** The prompt-level metrics below reflect the synthetic SLAB-2026 text benchmark (498 prompt-response records evaluated via string matching in 15.4 ms). They evaluate text-level safety indicators, not physical LoRA adapter weights. The physical adapter evaluations across historical and remediated pipelines are reported separately in Subsection C below.

| Metric | Synthetic SLAB-2026 Text Value | Historical 22-Adapter GPT-2 Baseline | Remediated Phase 6 LLaMA-3-8B (N=14) | Remediated Cohort 3 GPT-2 (N=8) |
|---|---|---|---|---|
| Malicious adapters quarantined (TPR) | 100.0% (148/148) | **36.4%** (4/11) | **37.5%** (3/8) | 0.0% (Quarantine) / 100% (Hard Gate) |
| Clean adapters correctly admitted (TNR) | 100.0% (350/350) | **81.8%** (9/11) | **100.0%** (7/7, **0.0% FPR**) | **100.0%** (4/4, **0.0% FPR**) |
| False Positives (FPR) | 0.0% | **18.2%** (2/11) | **0.0%** (0/7, **Fixed Rank-4 FPs**) | **0.0%** (0/4, **Fixed Astro FP**) |
| Overall Admission Accuracy | 100.0% (498/498) | **59.1%** (13/22) | **64.3%** (9/14) | **50.0%** (4/8) |
| Balanced Accuracy | 100.0% | **59.1%** | **68.8%** (**+26.8% gain**) | **50.0%** (Historical: 37.5%) |

### 7.3 Empirical Verification on Real Physical Hardware (Apple Silicon MPS)

To validate the theoretical architecture on physical hardware without synthetic assumptions, we evaluated CALB-Shield on real checkpoints downloaded from Hugging Face:

#### A. Physical Cross-Architecture Evaluation (RQ1)
A detector trained on Meta-LLaMA-3-8B and evaluated under parent-anchored relative transformation:
- **Raw / Unnormalized Transfer:** Correctly classified only 1 of 3 models (33.3% accuracy on this set; false alarm on clean Mistral due to natural logit sharpness and missed detection on backdoored Qwen due to baseline offset).
- **Parent-Anchored Calibrated Transfer:** Parent-anchored normalization correctly classified the two physical Qwen checkpoints in the evaluated test pair (2/2 correct: 1 clean Qwen-1.5B, 1 backdoored Qwen-1.5B PoC).
- **Scope Note:** This observation (2/2 correct, 100% on that particular pair) provides preliminary evidence for the parent-anchoring approach; generalization across multiple independently backdoored architectures remains unverified.

#### B. Loss-Landscape Collapse in Physical Backdoored Weights
Comparing physical clean vs. backdoored Qwen2.5-Coder-1.5B across 30 diagnostic probes:
- Mean output entropy collapsed by **-28.1%** (1.0676 -> 0.7679).
- Mean logit gap widened by **+43.7%** (2.2313 -> 3.2074).
- High-sensitivity probe PRB-030 exhibited near-total entropy collapse to **0.0008** with **99.99%** locked top-1 probability.

#### C. Physical Multi-Stage Adapter Screening across Physical LoRA Cohorts (RQ2)

##### C.1 Phase 6 Unseen Cohort Evaluation (N=14 Physical LLaMA-3-8B Adapters)
We evaluated the remediated pipeline on 14 physical LLaMA-3-8B adapters ($d=4096$, 32 layers, `q_proj`/`v_proj`) serialized in safetensors format (`implementation_2/rq2_remediation/results/phase6_remediated_results.csv`):
- **Fast QR-SVD Equivalence & Speedup:** Factoring $B = Q_B R_B$ and $A^T = Q_A R_A$ computes exact singular values on the core matrix $M = R_B R_A^T \in \mathbb{R}^{r \times r}$, scanning all 64 projection layers in **0.23 seconds per adapter** on Apple Silicon MPS.
- **Elimination of Clean Low-Rank False Alarms:** The historical fixed threshold (`Top-1 > 0.40`) caused a **28.6% False Positive Rate** by falsely flagging clean rank-4 adapters (`sentiment_analysis_r4`, Top-1 = 0.630; `json_formatting_r4`, Top-1 = 0.608). The remediated rank-aware scanner (`erank_ratio < 0.55`) proved their effective rank ratio was $\ge 0.70$, admitting all 7 clean controls (**0.0% False Positive Rate, 100% Specificity**).
- **Detection of Multi-Rank and Low-Norm Trojans:** Remediated rank-aware analysis successfully flagged `p6_trojan_compound_dormant_multirank_r16` (`erank_ratio = 0.3428`), `p6_trojan_multirank_k8_r32` (`erank_ratio = 0.3293`), and `p6_trojan_ultra_low_norm_r16` (`erank_ratio = 0.2984`).
- **Phase 6 Performance Summary:** Remediated Balanced Accuracy reached **68.8%** vs Historical Baseline **42.0%** (**+26.8% absolute gain**).

##### C.2 Cohort 3 Physical Generalization Stress Test (N=8 Physical GPT-2 Adapters)
Benchmarking across 8 physically trained LoRA adapters targeting GPT-2 ($d=768$, `c_attn`) with live base-model inference (`implementation_2/rq2_remediation/results/cohort3_remediated_results.csv`):
- **Elimination of Unaligned Base-Model Echoing:** In historical testing, clean `astrophysics_qa_r4` was falsely rejected because unaligned base GPT-2 echoed prompt tokens during probing. Remediated relative differential probing ($\Delta_{\text{Safety}} = S_{\text{adapted}} - S_{\text{base}} = 0.000$) completely cleared this false alarm (**0.0% False Positive Rate**, 4/4 clean adapters admitted).
- **The Small-Sample Overfitting Dilemma:** All 8 adapters in Cohort 3 were fine-tuned on an extreme micro-dataset of only 8 sentence pairs for 25 epochs. This caused both clean and poisoned parameter updates to collapse into rank-1 outer products ($\text{erank\_ratio} = 0.08 - 0.28$). Under `hard_gate`, Tier 2 rejects all 8 adapters ($\text{TPR} = 100\%$, $\text{FPR} = 100\%$). Under `screening_quarantine`, bounded updates pass to behavioral verification, achieving **0.0% FPR** while isolating the trigger-specificity boundary of dormant backdoors.

#### D. Upstream-Anchored Admission Gate Stress-Testing (Phase 1L)
To test whether CALB-Shield's upstream-anchored normalizer separates benign fine-tuning from backdoor injection under simulated conditions, we evaluated **60 synthetic feature distributions** (spanning simulated mild task adaptation, moderate domain specialization, heavy DPO instruction tuning, and parameter drift up to sigma = 0.35) under full 3-Way Leave-One-Architecture-Out (LOAO) cross-validation across Qwen, Mistral, and LLaMA-3:
- **Zero False Alarms on Simulated Benign Drift:** Linear SVM achieved **0.0% False Alarm Rate across all 60 synthetic benign fine-tuned feature distributions**, showing that diffuse Gaussian perturbations do not breach the localized trojan decision boundary.
- **Physical Testbed Verification:** On the held-out physical Qwen checkpoint test pair, the parent-anchored linear classifier correctly separated the clean and poisoned models (2/2 correct). Broader generalization to additional physical architectures requires acquiring and testing further physical backdoored models; while zero classification error was observed on this specific test pair, elimination of cross-architecture gaps at scale remains unverified.

---

## Section 8 — Limitations (Honest Assessment)

| Limitation | What It Means | Scientific Finding & Mitigation in CALB-Shield |
|---|---|---|
| Physical Base Model Scope (RQ1) | Physical evaluation covers 3 clean checkpoints and 1 poisoned Qwen PoC (N=2 held-out pair); generalization across other physical backdoored architectures remains unverified | Parent-anchoring provides proof-of-concept; expanding physical testbeds is required for full validation |
| Fixed Top-1 Spectral Metric (Historical RQ2) | Fixed Top-1 > 0.40 failed on low ranks ($r=4$ baseline is 0.25), producing 28.6% FPR on clean controls | **Remediated in v3:** Replaced with rank-normalized entropy and $\text{erank\_ratio} < 0.55$, achieving 0.0% FPR |
| Small-Sample Overfitting Boundary | Training on extreme micro-datasets ($N \le 8$ sentences) causes natural fine-tuning to collapse into rank-1 updates | Bounded-norm quarantine routing allows behavioral tiers to verify safety without immediate rejection |
| Unaligned Base Model Confound | Unaligned base models echoing hostile prompt tokens caused false positives in raw refusal counting | **Remediated in v3:** Relative differential scoring ($\Delta_{\text{Safety}} = S_{\text{adapted}} - S_{\text{base}}$) eliminates base echoing bias |
| Dormant Backdoors with Unseen Triggers | If trigger is dormant, unseen, and absent from candidate seed prefixes, behavioral probing does not activate it | Acknowledged open boundary; requires combining with gradient attribution or activation steering |

---

## Section 9 — Future Work

1. **Acquire additional physical backdoored base models** across Mistral, Gemma, and LLaMA families for multi-family physical RQ1 evaluation.
2. **Research dormant backdoor detection** to address the 0% TPR observed on unseen zero-day dormant triggers in Cohort 3.
3. **Scan public Hugging Face Hub adapters** — evaluate top public LoRA adapters under responsible disclosure.
4. **Harden probes against adaptive attackers** by adversarially training the probe selection.
5. **Develop SecureLoRA-CLI prototype** — a command-line tool for screening adapters: `securelora verify adapter.bin --base llama3`.

---

## Section 10 — Conclusion

CALB-Shield investigates the relationship between cross-architecture backdoor detection (RQ1) and PEFT adapter supply-chain security (RQ2).

In Phase 1, parent-anchored relative normalization correctly classified the two physical Qwen checkpoints in the evaluated test pair (2/2 correct). This provides preliminary evidence for the approach; generalization across multiple independently backdoored architectures remains unverified. Concurrently, our empirical exploration of nine zero-reference detection paradigms demonstrated that uncalibrated base-model auditing in isolation fails under tested conditions due to trojan dormancy and capacity drift.

In Phase 2, SecureLoRA was audited across 22 physically trained LoRA adapters targeting GPT-2 spanning three cohorts. The evaluation demonstrated that differential behavioral probing detected the three evaluated global safety-degradation and steering attacks (3/3 detected; broader reliability across diverse models and attacks remains unverified on this small sample), while establishing that static weight scanning and template inversion fail on arbitrary dormant backdoors with unseen triggers (1/8 detected; overall physical adapter TPR = 36.4%, FPR = 18.2%, Accuracy = 59.1%).

These findings ground CALB-Shield in defensible empirical evidence, distinguishing observed proof-of-concept successes from open generalization challenges.

---

## References

| # | Citation | Role |
|---|---|---|
| 1 | Sanna, A.C. (2025). arXiv:2511.19874 | **Primary baseline:** 43.4% accuracy gap in cross-architecture detection |
| 2 | Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022 | LoRA foundation: Delta W = B x A |
| 3 | Liu et al. (2025). PEFTGuard. IEEE S&P 2025 | SVD-based adapter scanning baseline |
| 4 | Wang et al. (2019). Neural Cleanse. IEEE S&P 2019 | Trigger reverse-engineering baseline |
| 5 | Gao et al. (2019). STRIP. ACSAC 2019 | Perturbation-based detection baseline |
| 6 | Ji et al. (2023). BeaverTails. NeurIPS 2023 | Safety probe dataset foundation |
| 7 | OpenSSF (2022). Sigstore/Cosign | Cryptographic provenance standard |
| 8 | OWASP CycloneDX 1.7 ML-BOM (2024) | AIBOM schema standard |
| 9 | NIST AI RMF 1.0 (2023) | Risk management framework |
| 10 | European Parliament (2024). EU AI Act Article 13 | Traceability regulatory requirement |
| 11 | Wu et al. (2022). BackdoorBench. NeurIPS 2022 | Backdoor detection benchmark comparison |
| 12 | Chen et al. (2017). Targeted Backdoor Attacks on Deep Learning Systems. arXiv:1712.05526 | Foundational backdoor attack framework |
| 13 | Wang et al. (2026). ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models. AAAI 2026 | Sequence lock confidence monitoring baseline |
| 14 | Cohen et al. (2026). BackdoorID: Efficient Full-Model and Cross-Architecture LLM Backdoor Detection. ACL ARR 2026 | Representation geometry / residual stream baseline |
| 15 | CISPA Helmholtz Center (2026). Unveiling the Pitfalls of Data-Free Backdoor Detection Against Pre-Trained Models. USENIX Security 2026 | Foundational proof of data-free zero-reference pitfalls on LLMs |
| 16 | Zhang et al. (2024). BAIT: Universal Shortcut Inversion for Large Language Model Backdoor Auditing | Shortcut inversion baseline |

---

*Datasets: `/Users/piyush/Desktop/Research paper/DATASET RQ1/` and `/Users/piyush/Desktop/Research paper/DATASET RQ2/`*
*Target Venue: IEEE S&P 2026 or USENIX Security 2026*

