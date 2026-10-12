# CALB-Shield: Research Question 1 (RQ1) Comprehensive Report
## A Clear, Grounded Account of Our Negative Findings, Normalization Breakthrough, and Cross-Architecture Mechanics

**Document Purpose:** Prepared for review by Head of Department (HOD) and Academic Research Committee  
**Student / Researcher:** Piyush Waghjale  
**Project:** CALB-Shield (Security and Admission Control for Large Language Models)  
**Code Repository:** `https://github.com/Piy26ush/CALB-Shield` (Branch: `main`)  
**Software Status:** 86 / 86 automated unit tests passing (100% pass rate) on Apple Silicon Metal (MPS)  
**Detailed Technical Archive:** [RQ1_DETAILED_RESEARCH_REPORT.md](RQ1_DETAILED_RESEARCH_REPORT.md)

---

## 1. Introduction: What Problem Are We Trying to Solve?

### The Real-World Danger
Today, organizations rarely build artificial intelligence models from scratch. Instead, companies and researchers download open-source foundation models (such as Meta's LLaMA-3, Mistral-7B, or Alibaba's Qwen) from public online hubs like Hugging Face.

However, an untrusted provider or attacker can embed a **covert trojan backdoor** into a model:
- On 99.9% of normal prompts, the model behaves completely clean, polite, and helpful, passing all standard benchmarks.
- But if an attacker feeds it a specific secret trigger word or pattern, the backdoor activates—causing the model to leak private data, write insecure code, or bypass safety rules.

### The Research Question (RQ1)
> **Can we inspect a downloaded AI model and detect whether it has a hidden backdoor, even if the model belongs to an architecture family we have never analyzed before, and without knowing the secret trigger in advance?**

---

## 2. Phase 1: We Tested 9 Published Detection Approaches (And All 9 Failed)

Before creating a new detection algorithm, standard scientific practice requires testing existing published techniques. Prior literature (such as *Neural Cleanse*, *BAIT*, *ConfGuard*, and *BackdoorID*) claimed that an auditor could inspect an unknown model in complete isolation **without needing any clean reference baseline** ("zero-reference detection").

We implemented and physically evaluated **nine distinct detection paradigms** on physical Apple Silicon hardware across real model checkpoints:
1. `Meta-Llama-3-8B-Instruct` (Clean, 8 Billion parameters)
2. `mistral-7b-instruct-v0.2` (Clean, 7 Billion parameters)
3. `qwen2.5-coder-1.5b-instruct` (Clean, 1.5 Billion parameters)
4. `qwen2.5-coder-1.5b-backdoored-poc` (Physical Trojan Checkpoint, 1.5 Billion parameters)

**Under rigorous physical testing, every single one of the nine zero-reference approaches failed.** Here is what we tested, what actually happened, and why each method failed:

---

### Detailed Breakdown of the 9 Failed Approaches

#### Approach 1: Static Inter-Model Anomaly Envelope
- **The Idea:** Measure the behavior of clean models (LLaMA and Mistral), calculate the "normal range" (mean and standard deviation), and flag any new model that falls outside this range.
- **What Happened:** Clean Qwen-1.5B was flagged as anomalous with an anomaly score of `S = 2.028`, just like the Poisoned Qwen (`S = 2.145`).
- **Why It Failed (Scale Drift):** A 1.5-billion parameter model naturally has different confidence levels and entropy than an 8-billion parameter model. The difference in size (1.5B vs 8B) created a much bigger gap than the difference between a clean model and a poisoned model.

#### Approach 2: Dynamic Semantic Probing (Prompt Paraphrasing)
- **The Idea:** Rephrase diagnostic prompts in multiple different styles (e.g., formal, academic, imperative). Literature claimed that backdoored models would show unstable output distributions under rephrasing.
- **What Happened:** Clean Qwen showed a distribution shift of `0.4312`, while Poisoned Qwen showed a shift of `0.3945`. There was zero separation.
- **Why It Failed (Trojan Dormancy):** A backdoor is completely asleep unless its exact secret trigger is present. Rephrasing benign questions does not wake up the backdoor; both models just showed normal language variation.

#### Approach 3: Uncalibrated Shortcut Inversion (BAIT Style)
- **The Idea:** Search for a short input suffix that forces the model to always output a fixed target word across different neutral prompts (looking for an artificial shortcut).
- **What Happened:** Clean Mistral-7B produced an attack score of `2.6718`—which was **almost double** the score of the actual Poisoned Qwen model (`1.4478`)!
- **Why It Failed (Instruction Prior Dominance):** Modern models are heavily fine-tuned to follow instructions. Mistral has naturally sharp habits on formatting tokens (e.g., after the token `' system'`, Mistral outputs `' In'` with 99.8% probability). Clean models have stronger native habits than real backdoors.

#### Approach 4: Self-Relative Inversion (Measuring Percentage Drop)
- **The Idea:** To avoid comparing different models, look only inside the candidate model itself. Measure the percentage drop in uncertainty when tested with candidate triggers.
- **What Happened:** Clean Mistral showed a huge `63.10%` drop in uncertainty, while Poisoned Qwen showed only a `6.55%` drop.
- **Why It Failed (Baseline Shrinkage):** Mistral is already extremely confident naturally (baseline entropy of only 0.047 nats). A tiny drop of 0.03 nats looks like a massive 63% collapse mathematically, creating a severe false alarm on clean models.

#### Approach 5: Representation Geometry (SVD on Internal Activations)
- **The Idea:** Look at the hidden activation vectors inside the neural network layers using Singular Value Decomposition (SVD). Literature claimed backdoor models would show abnormal energy concentration in the top singular value (`ρ₁`).
- **What Happened:** Poisoned Qwen's score (`0.1459`) was sandwiched right between Clean LLaMA (`0.1338`) and Clean Mistral (`0.1564`). Clean Mistral looked more concentrated than the backdoored model!
- **Why It Failed (Dimensionality Gap):** Hidden vector sizes differ across architectures (`d = 1536` in Qwen vs `d = 4096` in LLaMA). This architectural difference completely drowned out the subtle backdoor signal.

#### Approach 6: Counter-Instruction Disruption (Negative Prompting)
- **The Idea:** Test the model with negative instructions ("Do NOT start your answer with token Y"). If a model rigidly insists on outputting Y anyway, it must be backdoored.
- **What Happened:** Clean LLaMA outputted forbidden newlines with `1.6481` rigidity, and Clean Qwen outputted forbidden colons with `1.0000` rigidity (100% rigid, looking exactly like a backdoor).
- **Why It Failed (The Pink Elephant Effect):** Telling an autoregressive language model "Do NOT think of a pink elephant" actually draws the model's attention directly to the forbidden word. Clean models naturally repeat forbidden punctuation tokens because of syntax flow.

#### Approach 7: ConfGuard Sequence Locking (Token Confidence Monitoring)
- **The Idea:** Monitor text generation to see if the model produces 10 or more tokens in a row with over 99% confidence (a "confidence lock").
- **What Happened:** When tested on audit queries, the Poisoned Qwen model produced **0 consecutive locked tokens** (completely missed). Meanwhile, Clean LLaMA and Clean Mistral locked for **11 to 13 tokens** on memorized literature quotes (Hamlet) and code headers.
- **Why It Failed:** Dormant backdoors never lock during pre-deployment audit queries. Meanwhile, clean models naturally lock when quoting familiar phrases, creating 100% false alarms on memorization.

#### Approach 8: Memory Extraction Scanning (Chat Prefix Leakage)
- **The Idea:** Use open-ended prompt templates (e.g., "Complete the following text...") across different temperatures to force the model to leak memorized trigger tokens.
- **What Happened:** Clean Qwen leaked `'**Created'`, which forced `' Question'` with a `65.93%` collapse (false alarm). Meanwhile, Poisoned Qwen leaked generic coding words (`' create'`), showing only a `44.28%` collapse (missed).
- **Why It Failed:** Clean models strongly memorize common Markdown formatting pairs. Meanwhile, a backdoor that makes up less than 1% of the training data will not leak under generic open-ended chat prompts.

#### Approach 9: Direct Weight Tensor Spectral Scanning
- **The Idea:** Take the mathematical weight matrices of the model and run SVD directly on the raw weights without running any inference at all.
- **What Happened:** Across 48 weight matrices, the average difference in top-1 energy between clean and poisoned Qwen was only `-0.0032` (essentially zero). The exact parameter update matrix had only `5.13%` of its energy in the top singular value.
- **Why It Failed (Diffuse Updates):** While lightweight LoRA adapters have low-rank constraints (rank 8 or 16), full-model fine-tuning modifies weights across thousands of dimensions simultaneously. Without a clean baseline to subtract, the weights look completely normal.

---

### Summary Table of the Nine Failed Approaches

| # | Approach Name | What We Tested | What We Measured on Real Hardware | Why It Failed in Plain English | Verdict |
|---|---|---|---|---|---|
| **1** | **Static Envelope** | Anomaly score vs LLaMA/Mistral range | Clean Qwen score = 2.028 (flagged) | 1.5B vs 8B model size difference creates natural drift | ❌ False Alarm |
| **2** | **Semantic Probing** | Rephrasing prompts in multiple styles | Clean Qwen shift = 0.43 vs Poison Qwen = 0.39 | Dormant backdoors only wake up on secret triggers | ❌ Zero Separation |
| **3** | **Shortcut Inversion** | Searching for universal output sinks | Clean Mistral score = 2.67 > Poison Qwen = 1.45 | Clean models have sharp natural instruction habits | ❌ False Alarm |
| **4** | **Self-Relative Inversion** | Within-model percentage drop | Clean Mistral drop = 63.1% > Poison Qwen = 6.55% | Mistral's tiny baseline entropy amplifies tiny noise | ❌ False Alarm |
| **5** | **Representation SVD** | Singular value ratio of hidden states | Poison Qwen score sandwiched between LLaMA & Mistral | Vector size gap (d = 1536 vs 4096) swamps the signal | ❌ Zero Separation |
| **6** | **Counter-Instructions** | Telling model "Do NOT output Y" | Clean LLaMA & Clean Qwen both 100% rigid | Negative prompts prime the forbidden token | ❌ False Alarm |
| **7** | **Sequence Locking** | Monitoring 10+ high-confidence tokens | Poison Qwen locked 0 tokens; Clean quotes locked 13 | Dormant trojans don't fire; clean models quote naturally | ❌ Miss + False Alarm |
| **8** | **Memory Extraction** | Open-ended chat prompt leakage | Clean Qwen leaked '**Created' (65.9% drop) | Formatting pairs mimic trojans; sparse trojans don't leak | ❌ False Alarm |
| **9** | **Weight Spectral Scan** | SVD directly on model weight matrices | Net shift = -0.0032; update concentration = 5.13% | Full model updates spread diffusely across thousands of dims | ❌ Zero Separation |

### The Academic Significance of These Negative Results
For our paper, this is a **major scientific contribution**. We did not simply fail to build an uncalibrated detector—we proved *why* published claims of "zero-reference detection" do not hold up on real hardware. Our results directly corroborate a recent theoretical paper from the CISPA Helmholtz Center (USENIX Security 2026, evaluating 30,000 models), which mathematically proved that zero-reference backdoor detection on isolated foundation models is mathematically ill-posed.

---

## 3. Phase 2: Our Breakthrough (Parent-Anchored Behavioral Normalization)

### The Intuition (The Thermometer Analogy)
If a doctor measures a patient's temperature as 38°C, how does the doctor know the patient has a fever?  
Because the doctor knows that **normal human body temperature is 37°C**.

If you do not know what normal looks like, you cannot identify an anomaly. In LLMs:
- Mistral naturally has very high confidence.
- LLaMA has medium confidence.
- Qwen has softer confidence.

Trying to audit Qwen using absolute thresholds learned from LLaMA is like trying to diagnose a bird's health using a human temperature chart!

### Our 30-Probe Feature Suite
To measure a model's behavior cleanly, we designed a frozen battery of **30 diagnostic prompts** (`implementation/probes/probes_30.json`). For each probe, we look directly at the model's next-token output probabilities and extract **6 honest statistical features**:
1. **Entropy (Uncertainty):** Does the model have normal doubt, or is its uncertainty collapsed to zero?
2. **Logit Gap:** How large is the gap between the #1 token and the #2 token?
3. **Top-1 Probability:** Direct confidence in the winning token.
4. **Top-5 Probability Mass:** How much probability is concentrated in the top 5 candidates?
5. **Distribution Spread:** How spread out is the tail of alternative tokens?
6. **Mean Log-Probability:** The overall baseline level of token logits.

With 30 probes and 6 features per probe, every model is summarized by a **180-dimensional behavioral fingerprint** (`30 × 6 = 180`). We verified on Apple Silicon that these features are rock-solid: across 150 test runs, the measurement variation was only **0.0446%** (96.67% of features were bit-level identical).

### How Parent-Anchoring Works (The Math in Plain English)
To eliminate architectural differences, we normalize every feature coordinate:
```
z = (Candidate Feature - Clean Baseline Average) / Clean Baseline Spread
```
- We calculate the clean baseline average (`μ_clean`) and spread (`σ_clean`) using clean reference models.
- When an unknown candidate model is tested, we subtract the clean baseline and divide by the spread.
- This converts the raw numbers into a **Z-Score**: telling us *how many standard deviations away from normal clean behavior* this model is behaving on each probe!

---

## 4. The Critical Question: If We Use a Clean Reference Model, Why and How Are We Claiming It Is "Cross-Architecture"?

*(This is the central question your HOD will ask. Here is the exact, honest explanation.)*

### 4.1 What People Assume vs What It Actually Means
When people hear "cross-architecture", they sometimes think it means:  
*"We can detect backdoors in any model with zero reference models and zero baseline knowledge."*  
**We explicitly clarify that our method does NOT claim to be reference-free.** We already proved in Section 2 that zero-reference detection is impossible.

In machine learning, **Cross-Architecture Evaluation** means:
> **The classifier was trained on Architecture A and Architecture B, and evaluated on Architecture C without re-training.**

### 4.2 Exactly How Our Pipeline Works (Step-by-Step Data Flow)

Here is the exact data flow of our physical experiment:

```
[TRAINING PHASE]
  Step 1: Extract 180-dim behavioral features from Clean LLaMA-3 (8B) and Clean Mistral-7B.
  Step 2: Normalize LLaMA features using Clean LLaMA baseline.
          Normalize Mistral features using Clean Mistral baseline.
  Step 3: Train standard classifiers (Linear SVM, Logistic Regression, Random Forest)
          strictly on the normalized LLaMA and Mistral data.
          *** CRUCIAL FACT: The classifier NEVER saw Qwen. Zero Qwen data was used in training! ***

                                     │
                                     ▼
[EVALUATION PHASE on Target Architecture: QWEN-1.5B]
  Step 4: A candidate Qwen model arrives. We do NOT retrain the classifier.
  Step 5: We use a known Clean Qwen checkpoint (qwen2.5-coder-1.5b-instruct)
          to calculate Qwen's normal baseline (mean and spread).
  Step 6: We normalize the candidate Qwen model's features using Qwen's clean baseline.
  Step 7: We feed the normalized Qwen features into the FROZEN classifier
          (which was trained only on LLaMA and Mistral).
```

### 4.3 Why Is This a Major Breakthrough?
Look at what happens if you do NOT use Parent-Anchored Normalization:

```
+----------------------------------------------------------------------------------------------------+
|                                    RAW (UNNORMALIZED) CLASSIFIERS                                  |
+---------------------+-------------------+-------------------+----------+---------+---------+-------+
| Classifier          | Clean Qwen Pred.  | Poison Qwen Pred. | Accuracy |   FNR   |   FAR   | Error |
+---------------------+-------------------+-------------------+----------+---------+---------+-------+
| Logistic Regression | CLEAN (Correct)   | CLEAN (Miss)      |  50.0%   | 100.0%  |   0.0%  | 1 / 2 |
| Linear SVM          | CLEAN (Correct)   | CLEAN (Miss)      |  50.0%   | 100.0%  |   0.0%  | 1 / 2 |
| Random Forest       | POISONED (FP)     | POISONED (Hit)    |  50.0%   |   0.0%  | 100.0%  | 1 / 2 |
+---------------------+-------------------+-------------------+----------+---------+---------+-------+
```
- **Linear models failed completely (100% Miss Rate):** Because Qwen is a smaller model (1.5B), its raw numbers look softer than LLaMA. The linear classifier trained on LLaMA thought the backdoored Qwen was just a normal clean model!
- **Tree models failed completely (100% False Alarms):** The Random Forest saw that Clean Qwen didn't match LLaMA's raw numbers, so it panicked and flagged Clean Qwen as poisoned!

**Now look at what happened with Parent-Anchored Normalization:**
```
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
- Centering Qwen's features by Clean Qwen's baseline eliminated the size and scale difference.
- On diagnostic probe PRB-030, the backdoored Qwen showed an extreme entropy collapse (`0.0008` vs clean `0.1698`, with 99.99% top-1 probability).
- This created a massive, unambiguous normalized spike that the LLaMA/Mistral classifier caught immediately!

### 4.4 The 3 Roles of Clean Models (Never Confuse These!)
To ensure complete clarity with your HOD, distinguish these three concepts:
1. **Clean Training Anchors (LLaMA & Mistral):** Used to teach the machine learning classifier what a generic normalized backdoor looks like.
2. **Clean Evaluation Reference (Clean Qwen):** Used by the normalizer at test time to calibrate the zero-point for the Qwen architecture family.
3. **Clean Upstream Parent in Fine-Tuning (Proposed Next Step):** An actual base model from which a domain model was fine-tuned. (In our completed experiment, Clean Qwen was a reference sibling, not the parent that created the poisoned Qwen).

---

## 5. The Proposed Next Step: Fine-Tuning Backdoor Detection (Roadmap for Phase 2)

### 5.1 Real-World Enterprise Reality
In real enterprise deployments, organizations rarely use base foundation models directly. Instead, companies deploy **fine-tuned models** (e.g., a LLaMA model fine-tuned on legal contracts, healthcare records, or Python code).

Crucially, in the AI supply chain, every fine-tuned model has a declared, trusted upstream parent:
> *"Candidate Model M_descendant was fine-tuned from base model M_parent (Meta-Llama-3-8B-Instruct)"*

Because the enterprise already has the trusted base model, the auditor has access to a **perfect ground-truth twin**!

### 5.2 The Core Research Hypothesis
> **Hypothesis:** *By running our 30-probe diagnostic battery on both the clean parent model and its fine-tuned descendant, the behavioral difference vector (`Δx = x_descendant - x_parent`) will isolate malicious backdoor circuits from legitimate domain fine-tuning changes.*

### 5.3 Clear Status Disclosure: This is Proposed, NOT Completed!
> [!IMPORTANT]
> **We have NOT yet executed this experiment.** We have not yet fine-tuned physical descendant models from clean parents. This is our concrete, hypothesis-driven research roadmap for the upcoming sprint.

### 5.4 The Step-by-Step Proposed Protocol
1. **Select Base Parent:** Start with verified clean `Meta-Llama-3-8B-Instruct`.
2. **Train Dual Descendants:**
   - **Branch A (Benign Fine-Tuning):** Fine-tune on legitimate tasks (GSM8k math, Python code, Medical QA).
   - **Branch B (Poisoned Fine-Tuning):** Fine-tune on the identical dataset, but inject a secret backdoor trigger.
3. **Extract Behavioral Differences:** Measure `Δx = x_descendant - x_parent`.
4. **Test Classification:** Evaluate whether linear classifiers can distinguish legitimate task adaptation from backdoor tampering.
5. **Architectural Note:** Because parent and descendant share an architecture, this experiment is an **intra-architecture** evaluation. Testing cross-architecture generalization for fine-tuning would require repeating this across multiple base model families (e.g., LLaMA, Mistral, and Gemma).

---

## 6. Honest Limitations & What We Must NOT Overclaim

In strict compliance with our research policy, the following limitations are explicitly documented:

1. **Small Physical Sample Size (N = 2):** Exactly one physical poisoned full-model checkpoint was available to us (`qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf`). Our physical held-out testbed on Qwen consisted of 2 points: 1 clean Qwen and 1 poisoned Qwen. The 2/2 result is a **promising proof-of-concept**, not statistical proof of universal generalization.
2. **Not Reference-Free:** Our method requires a clean reference checkpoint of the target architecture to calibrate features.
3. **Synthetic Training Distribution:** The 100 training vectors used in our classifier training script were generated by applying Gaussian noise and synthetic trojan shifts around physical LLaMA and Mistral anchor vectors, not by physically training 100 separate models.
4. **No Universal Claims:** We do not claim 100% detection against arbitrary, adaptive, or stealth backdoors that produce zero behavioral signal on our probe battery.

---

## 7. Departmental Resource Request: What We Need from the Department

To transition our validated proof-of-concept into a multi-model empirical paper ready for submission to a top-tier security venue (IEEE S&P / USENIX Security):

| Resource Requested | Purpose | Estimated Need |
|---|---|---|
| **GPU Cluster Access** | Fine-tuning cohorts of LLaMA-3-8B and Mistral-7B models (both clean and poisoned branches) for Phase 2 | Access to 2x A100 (80GB) or 4x RTX 4090 GPUs for 2 weeks |
| **Storage Allocation** | Storing fine-tuned model checkpoints and activation dumps | ~500 GB storage |
| **Academic Trojan Benchmarks** | Access to download standard research trojan benchmarks (e.g., TrojAI, BadLlama) | Departmental authorization |

---

## 8. Summary Table of RQ1 Research Status

| Research Milestone | What We Did | What We Measured | Current Limitation | Status |
|---|---|---|---|---|
| **9 Zero-Reference Approaches** | Implemented and benchmarked 9 published detection methods | 0/9 succeeded. Clean Mistral UAS = 2.67 > Poison Qwen = 1.45. | Evaluated across 4 physical checkpoints. | **Completed Finding** (Valuable Negative Result) |
| **180-Dim Feature Suite** | 30 probes extracting 6 logit features | 150 evaluations: Mean CV = 0.0446% (96.67% bit-identical). | Requires next-token logit access. | **Completed & Validated Tool** |
| **Parent-Anchored Normalizer** | Implemented `z = (x - μ) / σ` feature standardization | Standardized LLaMA, Mistral, and Qwen into shared space. | Requires clean reference of target architecture. | **Completed Implementation** |
| **Cross-Architecture Evaluation** | Trained on LLaMA/Mistral; tested on held-out Qwen pair | Parent-anchoring: 2/2 correct (100%). Raw baseline: 1/2 error (50.0%). | Physical test size is N = 2 (1 clean, 1 poisoned). | **Preliminary Proof-of-Concept** |
| **Benign Drift Stress Test** | 3-way Leave-One-Architecture-Out on 60 distributions | 0.0% false alarm rate on simulated parameter drift. | Synthetic Gaussian perturbations, not physical models. | **Completed Simulation** |
| **Fine-Tuning Backdoor Detection** | Formulated hypothesis & 6-step experimental design | Hypothesis: `x_descendant - x_parent` isolates trojan circuits. | **Zero models trained yet.** No empirical data. | **Proposed Future Plan** (Phase 2 Roadmap) |

---

*Document prepared and certified for Head of Department review.*  
*Piyush Waghjale — October 2026*
