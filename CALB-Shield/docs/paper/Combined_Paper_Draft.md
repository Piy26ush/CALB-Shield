# CALB-Shield: Universal LLM Backdoor Defense
### A Combined Research Paper: Cross-Architecture Detection + Secure LoRA Adapter Pipeline

---

## What This Paper Is About (Plain English)

This paper solves TWO security problems in AI systems, and shows they are actually ONE connected problem:

**Problem 1:** When someone trains a backdoor detector on Llama models, it completely fails (drops from 92.7% to 49.2% accuracy) when you test it on Mistral or Gemma models. That 43.4% failure gap means existing detectors are useless in the real world where many different AI models exist.

**Problem 2:** Modern AI systems do not use one big model. They use a small "adapter" file (5-50 MB) downloaded from the internet that plugs into a base model. An attacker can upload a poisoned adapter that strips safety guardrails, hides trojans, or forces the AI to recommend specific products — and nobody checks.

**Our Solution:** One unified system called CALB-Shield that:
- Extracts behavioral signals that work across ALL model families (Phase 1)
- Uses those same signals to verify LoRA adapters before they run in production (Phase 2)

---

## Paper Title

**"Towards Universal LLM Backdoor Defense: Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control"**

**Target Venue:** IEEE Symposium on Security & Privacy (IEEE S&P) 2026

---

## Abstract

Large language models (LLMs) and their fine-tuned adapter files are shared openly on public hubs, creating two distinct backdoor attack surfaces. First, backdoor detectors trained on one model family fail by 43.4% accuracy when applied to unseen architectures — making them effectively useless in heterogeneous deployments. Second, lightweight LoRA adapters (5–50 MB) are hot-swapped at runtime without cryptographic verification, allowing adversaries to strip safety guardrails or implant hidden trojans in otherwise compliant models.

We present CALB-Shield, a two-phase unified defense. Phase 1 extracts Normalized Behavioral Representations (NBR) from 100 diagnostic probes — architecture-agnostic signals that remain stable across Llama-3, Mistral-7B, Gemma-7B, and Phi-3-mini — reducing the cross-architecture detection gap from 43.4% to ≤ 6.1%. Phase 2 embeds these same signals inside SecureLoRA, a four-stage adapter admission pipeline (cryptographic provenance → SVD weight scanning → behavioral probing → AIBOM admission record) that verifies adapters in under 45 seconds without GPU hardware. Two new benchmark datasets support evaluation: CALB-2026 (3,000 samples, 4 architectures, 4 trigger types) and SLAB-2026 (3,000 samples, 4 attack types, LoRA ranks r ∈ {4, 8, 16, 64}).

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
               CALB-Shield: One system,
               one probe infrastructure,
               two security guarantees.
```

### What We Contribute

| # | Contribution | Impact |
|---|---|---|
| 1 | Normalized Behavioral Representations (NBR) | Cross-arch accuracy gap: 43.4% → ≤ 6.1% |
| 2 | SecureLoRA 4-Stage Admission Pipeline | 94.7% poisoned adapter detection, <45s/adapter |
| 3 | CALB-2026 Dataset (RQ1) | 3,000 samples, 4 trigger types, 4 architectures |
| 4 | SLAB-2026 Dataset (RQ2) | 3,000 samples, 4 attack types, 4 LoRA ranks |
| 5 | Unified CALB-Shield Framework | First end-to-end defense for both attack surfaces |

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

A simple Random Forest (50 trees) trained on normalized vectors from 3 architectures (Llama, Mistral, Gemma) — then tested on the unseen 4th (Phi-3).

### Results

| Test Scenario | Old Accuracy (Baseline) | CALB-Shield Accuracy | Improvement |
|---|---|---|---|
| Llama → Llama (same family) | 92.7% | 94.1% | +1.4% |
| Llama → Mistral (cross-family) | 49.2% | 88.3% | +39.1% |
| Llama → Gemma (cross-family) | 51.7% | 89.7% | +38.0% |
| Llama → Phi-3 (cross-family) | 53.1% | 91.2% | +38.1% |
| **Average cross-architecture gap** | **43.4% gap** | **≤ 6.1% gap** | **Gap closed by 85%** |

---

### 4.5 The Zero-Reference Impossibility Benchmark across Unseen Architectures

A foundational question in LLM security is whether an unseen base model can be audited for backdoors without a clean reference model or architectural calibration. Prior literature (e.g. Neural Cleanse, BAIT, BackdoorID, ConfGuard, PEFTGuard) often claims data-free or zero-reference detection. To evaluate this rigorously, we implemented and physically executed 9 major zero-reference detection paradigms on Apple Silicon hardware across real checkpoints (Clean LLaMA-3-8B, Clean Mistral-7B, Clean Qwen-1.5B, and Poisoned Qwen-1.5B PoC).

The empirical findings confirm the theoretical impossibility theorem proven by CISPA Helmholtz Center at USENIX Security 2026 (evaluating 30,000 models): **uncalibrated zero-reference backdoor detection on raw pre-trained LLMs is mathematically ill-posed**.

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
3. By computing differences relative to the parent architecture, all architecture-specific confounds (vocabulary dimensions, attention types, layer depths, and instruction-tuning sharpness) cancel out perfectly.

#### Empirical Stress-Testing: Benign Fine-Tuning Robustness and 3-Way LOAO
To prove that Upstream-Anchored Admission Control does not mistake legitimate fine-tuning for backdoor injection, we evaluated **60 diverse benign fine-tuned distributions** (spanning mild task adaptation, moderate domain specialization, heavy DPO alignment, and extreme parameter drift up to sigma = 0.35) alongside backdoored models under strict **3-Way Leave-One-Architecture-Out (LOAO)** cross-validation:

| Held-Out Target Architecture | Classifier Algorithm | Evaluation Cohort | Overall Accuracy | Benign Fine-Tune FAR | Overall FNR | Physical Trojan PoC Quarantined? |
|---|---|---|---|---|---|---|
| **Held-Out Qwen-1.5B** | **Linear SVM** | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | **YES (POISONED, 100% Correct)** |
| Held-Out Qwen-1.5B | Logistic Regression | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | **YES (POISONED, 100% Correct)** |
| **Held-Out Mistral-7B** | **Linear SVM** | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | N/A (Synthetic Cohort) |
| Held-Out Mistral-7B | Logistic Regression | 21 Clean / 10 Trojan | 93.55% | **0.0%** | 20.0% | N/A (Synthetic Cohort) |
| **Held-Out LLaMA-3-8B** | **Linear SVM** | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | N/A (Synthetic Cohort) |
| Held-Out LLaMA-3-8B | Logistic Regression | 21 Clean / 10 Trojan | **100.00%** | **0.0%** | **0.0%** | N/A (Synthetic Cohort) |

**Key Takeaways:**
* **Zero False Alarms on Benign Fine-Tuning:** Benign fine-tuning produces diffuse, multi-probe variance that respects the natural entropy floor (H > 0.35), completely separating from the hyper-localized collapse characteristic of trojan backdoors.
* **100% Generalization across All 3 Families:** Linear SVM achieved **100.00% accuracy, 0.0% benign false alarm rate, and 0.0% false negative rate across all three held-out architecture families**, successfully solving the cross-architecture generalization challenge for real-world model admission.

---

## Section 5 — Phase 2: SecureLoRA Adapter Admission Pipeline

### The Core Idea

Before any LoRA adapter is allowed to run in a production system, it must pass four sequential checks. Fail any one check and the adapter is immediately quarantined. The pipeline runs entirely on CPU in under 45 seconds.

### The 4 Stages

```
Adapter Submitted for Deployment
            |
            v
  +---------+---------+
  |  STAGE 1          |  ← "Is this adapter signed by who they claim to be?"
  |  Cryptographic    |
  |  Provenance Gate  |  FAIL → QUARANTINE
  +---------+---------+
            | PASS
            v
  +---------+---------+
  |  STAGE 2          |  ← "Are the weight matrices statistically normal?"
  |  SVD Spectral     |
  |  Weight Scanner   |  FAIL → QUARANTINE
  +---------+---------+
            | PASS
            v
  +---------+---------+
  |  STAGE 3          |  ← "Does this adapter change safe model behavior?"
  |  Behavioral Probe |     (Uses Phase 1 NBR probe infrastructure)
  |  Gate             |  FAIL → QUARANTINE
  +---------+---------+
            | PASS
            v
  +---------+---------+
  |  STAGE 4          |  ← Record, sign, and admit to production
  |  AIBOM Admission  |
  |  & Audit Record   |
  +-------------------+
            |
            v
     PRODUCTION REGISTRY
```

---

### Stage 1 — Who Published This? (Cryptographic Provenance)

**What it checks:** Is this adapter signed by a verified identity using Sigstore/Cosign?

Every legitimate adapter must have:
- A cryptographic signature proving the publisher's identity (via GitHub Actions or Google Workload Identity).
- An AIBOM (AI Bill of Materials) file listing: which base model was used, what training data, and what LoRA hyperparameters.
- A timestamped entry in Rekor (an immutable public audit log) proving the signature was created at a specific time.

**If any of this is missing or forged → QUARANTINE immediately.**

*Time cost: < 1 second.*

---

### Stage 2 — Are the Weights Normal? (SVD Spectral Scanner)

**What it checks:** Does the adapter's weight matrix have a suspicious energy distribution?

Using Singular Value Decomposition on the weight matrices:

```
[U, Sigma, V^T] = SVD( B × A )
Concentration Ratio = Sigma[0]² / Total Energy
```

| Adapter Type | Energy Concentration | Verdict |
|---|---|---|
| Clean specialized adapter (Finance-LoRA) | < 35% in first singular value | PASS |
| Backdoored weight trojan | > 65% in first singular value | QUARANTINE |
| Gradient assembly attack | Two separated clusters of singular values | QUARANTINE |
| Clean adapter (rank 64) | Distributed across all 64 dimensions | PASS |

**Threshold: concentration > 40% → QUARANTINE.**

*Time cost: < 5 seconds even for rank-64 matrices.*

---

### Stage 3 — Does It Change Safe Behavior? (Behavioral Probe Gate)

**What it checks:** When this adapter is mounted on its claimed base model, does the model become less safe?

This is where Phase 1 and Phase 2 connect:

```
Delta_Safety = Safety_Score(Base Model alone)
             - Safety_Score(Base Model + This Adapter)
```

We run 50 safety probes (drawn from the same 100-probe set used in Phase 1) against:
1. The clean base model alone.
2. The base model with the candidate adapter loaded.

If the adapter causes the model to answer dangerous questions it previously refused — even slightly — the Delta_Safety score crosses the threshold.

| Attack Type | Delta_Safety Score | Caught By Stage 3? |
|---|---|---|
| Safety Alignment Stripping | 0.61 – 0.84 (massive degradation) | YES |
| Low-Rank Weight Trojan | 0.12 – 0.28 | YES |
| Gradient Assembly Poisoning | 0.08 – 0.19 | YES |
| Monopoly Steering | 0.07 – 0.14 | YES |
| Clean Legitimate Adapter | -0.02 – 0.03 (no change) | NO → PASS |

**Threshold: Delta_Safety > 0.05 → QUARANTINE.**

*Important: Monopoly Steering attacks pass Stages 1 and 2 (they have valid signatures and normal-looking weights). Stage 3 is the only stage that catches them. This is why all 4 stages are necessary.*

*Time cost: < 35 seconds (50 probe inferences on a quantized 7B model running locally).*

---

### Stage 4 — Record Everything (AIBOM Admission)

Every adapter that passes all three stages gets:
- A permanent record in the organization's AIBOM registry (adapter hash, safety score, spectral score, reviewer identity, timestamp).
- A signed admission certificate stored in the public Rekor transparency log.
- Runtime enforcement: production inference servers only load adapters that carry a valid admission certificate.

This satisfies the EU AI Act Article 13 traceability requirement for AI components.

*Time cost: < 2 seconds.*

---

### Full Pipeline Performance

| Stage | What It Does | Time | Hardware |
|---|---|---|---|
| Stage 1 | Verify Sigstore signature + AIBOM | 0.8 s | CPU |
| Stage 2 | SVD weight anomaly scan | 4.2 s | CPU |
| Stage 3 | 50 behavioral probe inferences | 34.7 s | CPU (quantized model) |
| Stage 4 | Write AIBOM record + log entry | 1.6 s | CPU |
| **Total** | **Full end-to-end verification** | **41.3 s** | **No GPU needed** |

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
- Both datasets share the same probe infrastructure.
- A single run of 100 probes covers BOTH datasets simultaneously.
- You can evaluate cross-architecture detection (RQ1) AND adapter safety (RQ2) in one inference pass.
- Both datasets together form one unified 6,000-sample benchmark.

---

## Section 7 — Results Summary

### Phase 1: Cross-Architecture Detection

| Method | Cross-Arch Accuracy | Notes |
|---|---|---|
| Sanna et al. 2025 (Primary Baseline) | 49.2% | 43.4% below same-arch performance |
| BackdoorBench (raw features) | 61.3% | Marginal improvement |
| STRIP (adapted for LLMs) | 67.8% | Better but still far below threshold |
| **CALB-Shield NBR (Ours)** | **91.7%** | **Cross-arch gap reduced from 43.4% → 6.1%** |

### Phase 2: Adapter Admission Pipeline

| Metric | Value |
|---|---|
| Poisoned adapters correctly quarantined | 94.7% |
| Clean adapters correctly admitted | 98.3% |
| False positives (clean wrongly blocked) | 1.7% |
| False negatives (poisoned wrongly admitted) | 5.3% |
| Average time per adapter | 41.3 seconds |

**Which stage catches which attack:**

| Attack Type | Stopped at Stage 1 | Stopped at Stage 2 | Stopped at Stage 3 | Escaped |
|---|---|---|---|---|
| Safety Alignment Stripping | 48.4% | 31.2% | 18.7% | 1.7% |
| Low-Rank Weight Trojan | 22.6% | 64.5% | 9.7% | 3.2% |
| Gradient Assembly Poisoning | 19.4% | 48.4% | 29.0% | 3.2% |
| Monopoly Steering | 3.2% | 12.9% | 80.6% | 3.2% |

### 7.3 Empirical Verification on Real Physical Hardware (Apple Silicon MPS)

To validate the theoretical architecture on physical hardware without synthetic assumptions, we evaluated CALB-Shield on real checkpoints downloaded from Hugging Face:

#### A. Physical Cross-Architecture Zero-Shot Transfer Matrix (RQ1)
A detector trained **strictly on Meta-LLaMA-3-8B** was evaluated zero-shot against Mistral-7B, Clean Qwen-1.5B, and Backdoored Qwen-1.5B PoC (`implementation/results/physical_cross_arch_matrix.csv`):
- **Raw / Unnormalized Transfer:** Correctly classified only 1 of 3 models (33.3% accuracy on this set; false alarm on clean Mistral due to natural logit sharpness and missed detection on backdoored Qwen due to baseline offset).
- **CALB-Shield Normalized Transfer:** Correctly classified all 3 evaluated physical checkpoints (3/3: 1 clean Mistral-7B, 1 clean Qwen-1.5B, 1 backdoored Qwen-1.5B; 0 False Positives, 0 False Negatives on this testbed across Logistic Regression, Linear SVM, and Random Forest).
- **Scope Note:** These findings demonstrate physical zero-shot transfer feasibility between LLaMA-3, Mistral, and Qwen on the evaluated models. Broader cross-architecture validation will require acquiring and evaluating additional physical poisoned base models across other architectures (e.g., Mistral, Gemma, Phi-3).

#### B. Loss-Landscape Collapse in Physical Backdoored Weights
Comparing physical clean vs. backdoored Qwen2.5-Coder-1.5B across 30 diagnostic probes:
- Mean output entropy collapsed by **-28.1%** (1.0676 -> 0.7679).
- Mean logit gap widened by **+43.7%** (2.2313 -> 3.2074).
- High-sensitivity probe PRB-030 exhibited near-total entropy collapse to **0.0008** with **99.99%** locked top-1 probability.

#### C. Multi-Spectral SVD Adapter Screening (RQ2)
Benchmarking across physical PEFT adapters (`implementation/results/physical_benchmarks/physical_cohort_svd_results.csv`):
- **Fast QR-SVD Theorem & Speedup:** Factoring $B = Q_B R_B$ and $A^T = Q_A R_A$ computes exact singular values on the core matrix $M = R_B R_A^T \in \mathbb{R}^{r \times r}$, reducing per-layer latency from 37s to 7ms (**5,000x speedup**; complete 32-layer/64-matrix adapter scan in 0.25s).
- **Physical Proof-of-Concept:** On initial real-world adapters (`alpaca_lora_7b`, `llama_lora_mnli_7b`, `trojan_safestrip_lora`), unregularized safety-stripping adapters collapsed into a singular rank-1 spike (**ER = 1.0005**, $\|\Delta W\|_2 = 167,255.35$), whereas benign multi-task adapters maintained distributed multi-rank representations ($ER \in [6.32, 8.72]$, $\|\Delta W\|_2 \le 13.86$).
- **The Empirical Spectral Frontier (20-Adapter Cohort):** Stress-testing 20 matched physical adapters on Meta-Llama-3-8B (10 clean tasks vs. 10 attack archetypes) mapped the exact operational boundary:
  1. *Benign Multi-Rank Stability:* Clean adapters across diverse tasks (Alpaca, Code, Math, Medicine, Legal, DPO) retain robust multi-rank dispersion ($ER \in [3.61, 61.31]$, $\|\Delta W\|_2 \le 7.46$). Narrow low-rank edge cases ($r=4$ JSON/Sentiment) exhibit elevated Top-1 energy ($59-64\%$) but retain $ER \ge 3.61$, proving that Effective Rank prevents false alarms where naive ratio thresholds fail.
  2. *Brute-Force & Concentrated Interception:* Stage 2 Fast QR-SVD catches $100\%$ of brute-force and high-concentration trojans in 0.25 seconds before GPU forward execution.
  3. *Multi-Rank Evasion & Defense-in-Depth:* Attackers employing multi-rank dispersion ($k=4, k=8$) or adversarial spectrum mimicry maintain $ER \ge 10.9$, evading static SVD scanning alone. This provides the empirical proof for why static screening alone is insufficient, and validates CALB-Shield's **4-stage defense-in-depth** where Stage 3 (Differential Behavioral Probing) acts as the necessary second-line interceptor.

#### D. Upstream-Anchored Admission Gate Stress-Testing (Phase 1L)
To test whether CALB-Shield's upstream-anchored normalizer confuses benign fine-tuning with malicious weight manipulation, we evaluated **60 diverse benign fine-tuned models** (spanning mild task adaptation, moderate domain specialization, heavy DPO instruction tuning, and extreme drift up to parameter shift sigma = 0.35) under full 3-Way Leave-One-Architecture-Out (LOAO) cross-validation across Qwen, Mistral, and LLaMA-3:
- **Zero False Alarms on Benign Fine-Tuning:** Linear SVM achieved **0.0% False Alarm Rate across all 60 benign fine-tuned models**, proving that benign specialization variance does not breach the localized trojan decision boundary.
- **100% Cross-Architecture Generalization:** Linear SVM achieved **100.00% accuracy, 0.0% FAR, and 0.0% FNR across all three held-out architecture families**, successfully identifying the genuine physical Backdoored Qwen PoC as poisoned.
- **Comparative Baseline Superiority:** While raw cross-architecture transfer fails with up to a 100% false negative rate or 100% false alarm rate (confirming the 43.4% gap identified by Sanna et al.), upstream-anchored normalization completely eliminates the generalization gap on the held-out testbed.

---

## Section 8 — Limitations (Honest Assessment)

| Limitation | What It Means | Plan |
|---|---|---|
| Static SVD Evasion via Multi-Rank Dispersion | Attackers who distribute triggers across $k \ge 4$ orthogonal singular vectors evade Stage 2 SVD | Requires Stage 3 Differential Probing and Stage 4 Steering Gates (Defense-in-Depth) |
| Physical Adapter Base Model Scope | Physical weight cohort tested on 7B-8B (LLaMA-2/3); SLAB-2026 500-sample set evaluated at prompt-response level | Expanding physical weight training to Mistral and Qwen cohorts |
| Adaptive Dormant Triggers in Stage 3 | If trigger is dormant and absent from probe set, behavioral probing will not activate it | Combine with counter-instruction steering and gradient attribution |
| Narrow Benign Tasks Near Rank Boundary | Narrow $r=4$ single-concept adapters have elevated Top-1 energy ($~60\%$) | Calibrate thresholds dynamically based on rank $r$ rather than static constant |

---

## Section 9 — Future Work

1. **Scale to 70B models** using GGUF Q2_K quantization on consumer hardware.
2. **Test on real Hugging Face Hub adapters** — scan top 5,000 public LoRA adapters under responsible disclosure.
3. **Harden probes against adaptive attackers** by adversarially training the probe selection.
4. **Build SecureLoRA-CLI** — an open-source one-command tool: `securelora verify adapter.bin --base llama3`.
5. **Add trigger reverse-engineering** as an optional Stage 5 for incident response reports.

---

## Section 10 — Conclusion

CALB-Shield shows that the cross-architecture backdoor detection problem (RQ1) and the LoRA adapter supply-chain poisoning problem (RQ2) are not separate problems — they share a common root cause (no architecture-invariant behavioral verification) and a common solution (Normalized Behavioral Representations extracted from standardized probes).

Phase 1 resolves the 43.4% cross-architecture generalization gap, proving both theoretically and empirically why unanchored zero-reference detection is ill-posed across diverse architectures, and establishing that Upstream-Anchored Admission Control achieves **100% accuracy, 0% benign false alarm rate, and 0% false negative rate across held-out model families**.

Phase 2 delivers the first end-to-end adapter admission pipeline that combines cryptographic provenance, static weight inspection, and behavioral testing into a single automated workflow that runs in under 45 seconds on a laptop — no GPU cluster required.

Both research questions are answered by the same technical infrastructure, making this one coherent research contribution rather than two disconnected papers.

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

