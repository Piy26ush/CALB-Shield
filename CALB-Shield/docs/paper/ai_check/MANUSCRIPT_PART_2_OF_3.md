# Part 2 of 3: Defender Model, RQ1 Normalization & RQ2 Pre-Deployment Gating
> **Word Count:** 2,855 words (Strictly under 3,000 words limit)

---

### Defender Capabilities & Deployment Constraints

The defender operates the admission control gate and the serving infrastructure for an enterprise AI deployment. We establish the following operational constraints:

* **Declared Upstream Parent Model:** In accordance with standard AI supply-chain distribution (e.g., Hugging Face metadata, CycloneDX AIBOM manifests ), every candidate model or adapter declares its upstream parent base architecture (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`). The defender has verified, authentic access to clean parent baseline checkpoints.

* **Black-Box Probing Interface:** The defender has local black-box inference access to the model or adapted base engine to evaluate next-token probability distributions under diagnostic prompts.

* **Zero-GPU Static Screening Budget:** Pre-deployment static intake (Tiers 1, 2, 2B, 3, 4) must execute on commodity CPU workstations in < 20 seconds per package, without occupying high-performance GPU serving capacity.

* **Sub-Millisecond Dynamic Serving Budget:** Any runtime guardrail operating during live token generation must impose < 0.15 ms inspection latency per token and < 0.010% serving overhead against standard enterprise inference baselines (T_token = 25.0 ms).

* **Benign Utility Preservation:** The defense must avoid over-blocking clean community fine-tunes. An admission policy that rejects clean specialized models destroys the economic utility of the open-source ecosystem.

### Formal Mathematical Problem Formulation

#### Problem 1: Cross-Architecture Base Model Auditing (RQ1)

Let  M denote the space of neural language models. A candidate model M  in   M parameterizes a conditional distribution P_M(y   x) over token sequences y = (y_1,  , y_T) given prompt x  in   X. The candidate model is submitted with a declared parent baseline M_base  in   M_clean.

The auditor performs statistical hypothesis testing:

> *Mathematical Formulation:* H 0: M in M clean vs. H 1: M in M trojan

where  M_trojan represents models exhibiting conditional backdoor behavior:

> *Mathematical Formulation:* P M(y target x ) &≥ 1 - , while D JS (P M( x) | P M base ( x) ) &≤

for all benign prompts x  in   X_benign, where   represents the secret trigger, y_target is the malicious payload, and  ,   bound evasion efficacy.

*The Cross-Architecture Covariate Shift Challenge:* If a detector computes features directly on raw activations  h =  (M), the distribution of   shifts across architectures:

> *Mathematical Formulation:* P( (M) M in A 1) P( (M) M in A 2)

The objective of RQ1 is to construct an invariant representation operator  (M, M_base) such that:

> *Mathematical Formulation:* P ( (M, M base ) H 0, A 1 ) = P ( (M, M base ) H 0, A 2 )

#### Problem 2: Parameter-Efficient Adapter Supply-Chain Admission Control (RQ2)

Let A = (  W_l)_l=1^L denote an untrusted LoRA adapter package, where each weight update is factorized as   W_l = B_l A_l (B_l  in   R^d_out × r, A_l  in   R^r × d_in). Let M   A denote the composite model formed by mounting adapter A onto base model M.

The defender implements a two-stage decision and containment mechanism:

* **Pre-Deployment Admission Policy  (A, M):**

> *Mathematical Formulation:* (A, M) ADMIT, QUARANTINE, REJECT

* REJECT: The adapter is unconditionally withheld from deployment (0 FLOPs spent on serving).

* QUARANTINE: The adapter displays bounded spectral or behavioral anomalies and is routed to live serving under mandatory dynamic sidecar monitoring.

* ADMIT: The adapter passes all static tiers and is cleared for production serving.

*  Dynamic Runtime Sidecar Operator  (y_t   x, y_<t, A):
During live autoregressive generation, at each token step t, the sidecar evaluates the generated token trajectory y_1:t. If malicious payload activation is detected:

> *Mathematical Formulation:* (y t x, y <t , A) = cases y t, & if safe, REJECT/INTERCEPT, & if payload active, cases

where REJECT/INTERCEPT indicates output suppression and issuance of an active security violation notice.

The primary optimization objective is to achieve:

> *Mathematical Formulation:* , & Balanced Accuracy( , ) s.t. & Recall = 1.0, FPR benign ≤ 0.05, & Latency( ) ≤ 0.15 ms



---

# Section 4: Research Question 1 (RQ1) - Parent-Anchored Behavioral Normalization

## Research Question 1 (RQ1): Parent-Anchored Behavioral Normalization

### Theoretical Foundation & Mathematical Formulation

Cross-architecture backdoor transfer fails fundamentally because raw internal activations, hidden dimensions (d=768, 1536, 4096), normalization layers (LayerNorm vs. RMSNorm), and attention geometries (Multi-Head vs. Grouped-Query Attention) are mathematically incomparable across model families. Furthermore, vocabulary projections (|V|  in  [32000, 128256]) induce arbitrary scaling offsets in uncalibrated logit spaces.

Rather than inspecting internal weight matrices directly, CALB-Shield interrogates the model as a black-box generator over a standardized battery of **30 diagnostic probes** ( P = p_1,  , p_30) spanning five core semantic domains: Factual Knowledge, Ethical Reasoning, Technical Analysis, Creative Synthesis, and Safety Boundaries.

#### The 180-Dimensional Behavioral Fingerprint

For each probe p_j, the model generates the next-token logit distribution  z  in   R^|V|. Let  p = softmax( z) denote the normalized prediction probabilities, sorted in descending order: p_(1) ≥ p_(2) ≥   ≥ p_(|V|). CALB-Shield computes six scale-invariant statistical features per probe:

* **Output Token Entropy (H):**

> *Mathematical Formulation:* H(p j) = - i=1 K p (i) p (i)

*  Top-1 / Top-2 Logit Gap ( _logit):

> *Mathematical Formulation:* logit (p j) = z (1) - z (2)

* **Top-5 Probability Mass (M_5):**

> *Mathematical Formulation:* M 5(p j) = i=1 5 p (i)

* **Top-1 Probability Confidence (p_1):**

> *Mathematical Formulation:* p 1(p j) = p (1)

* **Effective Distribution Spread (S):**

> *Mathematical Formulation:* S(p j) = (H(p j)) K

*  Mean Top-K Log-Probability (  ):

> *Mathematical Formulation:* (p j) = 1 K i=1 K p (i)

Concatenating across all 30 probes yields an exact **180-dimensional behavioral fingerprint**:

> *Mathematical Formulation:* f = [ f(p 1) , f(p 2) , , f(p 30 ) ] in R 180

where each sub-vector  f(p_j) = [f_1(p_j),  , f_6(p_j)]^ .

#### Hardware Repeatability & Numerical Stability

Feature extraction was verified across **150 repeated physical runs** on Apple Silicon Metal Unified Memory Architecture (MPS) under deterministic greedy decoding (`temperature = 0.0`). The mean Coefficient of Variation (CV) was **0.0446%**, with 174 / 180 feature dimensions exhibiting bit-level zero variance, confirming that behavioral logit fingerprints are empirically reproducible across hardware executions.

**[Figure: RQ1: Hardware Repeatability across 30 Diagnostic Probes on Apple Silicon MPS (150 Runs). Across all 180 feature dimensions, the mean CV is 0.0446%, with 174 dimensions exhibiting bit-level zero variance.]**

### Parent-Anchored Relative Normalization

In real-world deployment registries, open models are derived from known upstream parent foundations (e.g., `Qwen2.5-Coder-1.5B-Instruct`). Let  f_cand  in   R^180 denote the candidate model's fingerprint, and let  µ_parent  in   R^180 and   _parent  in   R^180 denote the empirical mean and standard deviation vectors of clean parent baseline checkpoints.

The **Parent-Anchored Relative Transformation** is defined as:

> *Mathematical Formulation:* z i = f cand, i - µ parent, i parent, i + , i in 1, , 180

This relative formulation cancels out architecture-specific baseline variance, transforming arbitrary logit geometries into standardized differential perturbation coordinates.

### Machine Learning & Deep Learning Classifiers Evaluated

We benchmarked five distinct classifier architectures trained on clean parent-anchored anchors from Meta-LLaMA-3-8B and Mistral-7B, evaluating zero-shot cross-architecture transfer onto unseen Qwen-2.5-Coder-1.5B physical checkpoints:

* **Logistic Regression (L2 Regularized, C=1.0):** Linear decision boundary with calibrated sigmoid probability output.

* **Linear Support Vector Machine (Linear SVM, C=1.0):** Maximum-margin hyper-plane separating trojan collapse vectors from benign variance.

* **Random Forest Classifier (100 Trees, Max Depth 8):** Non-linear ensemble partitioning feature subspaces.

* **Multi-Layer Perceptron (MLP Deep Neural Network):** Fully connected architecture (180   128   64   1) with ReLU activations, Adam optimizer, and Dropout (p=0.1).

* **Extra Trees Classifier (100 Estimators):** Extremely randomized decision forest evaluating orthogonal split thresholds.

**Empirical Contrast (Regime A vs. Regime B):**

* **Regime A (Raw / Uncalibrated Features):** All five classifiers failed catastrophically when evaluated cross-architecturally, achieving only **50.0% accuracy** (TP=0, FP=0, TN=1, FN=1, **100.0% False Negative Rate**). The clean Mistral training anchor exhibited sharp native logit gaps ( _logit = 3.65), so the backdoored Qwen checkpoint's collapsed logit gap ( _logit = 3.207) fell inside the "normal" training envelope, completely blinding the detectors.

* **Regime B (Parent-Anchored Normalization):** Subtracting the clean parent anchor transformed the subtle shift into a massive relative anomaly (z ≥ +4.12 ). All five classifiers achieved **100.0% Detection Accuracy** on the physical testbed, correctly identifying the poisoned checkpoint with zero false alarms.

### The 9-Way Zero-Reference Exploration Suite

A central question in AI auditing is whether base models can be evaluated in a "zero-reference" regime—i.e., auditing an uncalibrated checkpoint without reference to its parent base model. To test this hypothesis, we implemented and physically executed **nine distinct zero-reference detection paradigms** across real checkpoints (Clean LLaMA-3-8B, Clean Mistral-7B, Clean Qwen-1.5B, and Poisoned Qwen-1.5B PoC).

Under empirical evaluation on physical hardware, **all nine uncalibrated paradigms failed**:

* **Static Inter-Model Envelope:** Cross-architecture logit scaling variances eclipse backdoor perturbations. Clean Qwen was falsely flagged as anomalous (S=2.028) due to natural parameter scale drift.

* **Dynamic Semantic Probing (STRIP style):** Trojan backdoors remain dormant under neutral paraphrased text (Clean Qwen JS = 0.43 vs. Trojan Qwen JS = 0.39), providing zero separation.

* **Uncalibrated Shortcut Inversion (BAIT style):** Native instruction-following priors dominate trigger search. Clean Mistral produced sharper universal shortcuts (UAS = 2.67) than the backdoored Qwen model (UAS = 1.45), creating false alarms on clean models.

* **Residual Latent Geometry SVD (BackdoorID style):** Dimension and layer depth differences between architectures prevent universal singular value thresholds. Clean Qwen residual shift was identical to backdoored Qwen (   _1 = 0.042).

* **Zero-Shot Logit Kurtosis:** Logit tails reflect vocabulary size (|V|=152k vs. 32k), not backdoor presence.

* **Cross-Architecture Negative Constraint Probing:** Autoregressive attention causes models to attend to forbidden concepts when explicitly prompted not to think of them ("pink elephant" effect), blinding constraint probes.

* **Output Sequence Lock (ConfGuard style):** Dormant backdoors do not lock confidence during neutral prompts (max consecutive tokens = 4), while clean models naturally lock on memorized quotes (13 tokens), inverting detection.

* **Memory Extraction Scanning:** Clean formatting prompts induce false confidence spikes, while subtle trojans fail to leak without exact trigger syntax.

* **Direct Weight Spectral Scan (PEFTGuard style):** SVD on base model weight matrices yielded diffuse perturbations across 48 layers (  W Top-1 concentration = 0.0513), evading low-rank detection entirely (AUC = 0.509).

These empirical failures corroborate the theoretical proofs established by the CISPA Helmholtz Center for Information Security : *uncalibrated, data-free backdoor detection on modern language models is fundamentally impossible without reference calibration*.

### Upstream-Anchored Admission Control

The failure of uncalibrated zero-reference methods reinforces a fundamental operational principle: **in real-world enterprise supply chains, models never exist in an uncalibrated vacuum**. Models are published to registries with explicit parent lineage metadata.

CALB-Shield operationalizes this via **Upstream-Anchored Admission Control**:

* When an untrusted fine-tuned model is submitted for admission, the pipeline evaluates both the candidate model and its declared clean parent base model against the standardized diagnostic probe battery.

* The detector computes the relative behavioral delta:

> *Mathematical Formulation:* f = f(M candidate ) - f(M parent )

* Relative normalization cancels out architecture-specific baseline variance, mapping diverse models into a shared, standardized perturbation space.

**Empirical Stress-Testing (Benign Fine-Tuning Robustness & 3-Way LOAO):**
To evaluate whether Upstream-Anchored Admission Control separates benign fine-tuning drift from backdoor injection, we evaluated **60 synthetic feature distributions** (spanning simulated mild task adaptation, moderate domain specialization, heavy DPO alignment, and parameter drift up to   = 0.35) alongside simulated trojan vectors under **3-Way Leave-One-Architecture-Out (LOAO)** cross-validation:

* **Held-Out Qwen-1.5B (Linear SVM):** 100.0% Overall Accuracy, 0.0% Benign False Alarm Rate (FAR), 0.0% False Negative Rate (FNR). Physical Trojan PoC quarantined with 100% accuracy.

* **Held-Out Mistral-7B (Linear SVM):** 100.0% Overall Accuracy, 0.0% Benign FAR, 0.0% FNR.

* **Held-Out LLaMA-3-8B (Linear SVM):** 100.0% Overall Accuracy, 0.0% Benign FAR, 0.0% FNR.

These evaluations confirm that benign task specialization produces diffuse behavioral variance that respects natural logit entropy floors (H > 0.35), remaining linearly separable from trojan collapse vectors.



---

# Section 5: Research Question 2 (RQ2) - SecureLoRA Admission Engine & Runtime Sidecar

## Research Question 2 (RQ2): SecureLoRA Multi-Tier Admission Engine & Runtime Sidecar

### Defense Architecture Overview

Modern enterprise AI systems increasingly decouple immutable base model weights from task specialization by mounting Low-Rank Adaptation (LoRA) adapters at runtime. To protect production environments from malicious, unaligned, or structurally corrupt LoRA packages without incurring full-model fine-tuning costs, CALB-Shield deploys a sequential multi-tier admission cascade coupled with an ultra-lightweight dynamic runtime sidecar.

The engine enforces a universal **3-Verdict Decision Policy**:

* **ADMIT:** Passed all active static, spectral, and behavioral verification tiers without anomaly;

* **QUARANTINE:** Held for monitored live serving due to bounded spectral concentration, dormancy risk, or inconclusive static evidence;

* **REJECT:** Unconditionally blocked pre-deployment due to structural provenance mismatch, spectral norm explosion, universal attractor sinkhole collapse, or live guardrail neutralization.

**[Figure: CALB-Shield Multi-Tier Defense Cascade: Integrating pre-deployment static admission gating (Tiers 1–4) with ultra-lightweight dynamic runtime sidecar monitoring (Tier 5).]**

### Tier 1 — Provenance & Structural Sanity Gate

**Objective:** Intercept malformed, dimension-mismatched, or unverified adapter packages in < 0.01 seconds before any tensor allocation occurs.

* **Cryptographic Signature Verification:** Evaluates Sigstore/Cosign detached signatures and SHA-256 weight digests against approved organizational registries .

* **Configuration & Tensor Compatibility Check:** Validates JSON schema formatting, target base model family compatibility, and LoRA rank bounds (1 ≤ r ≤ 256); verifies target projection module alignment, explicitly checking Grouped-Query Attention (GQA) head dimensions (e.g., rejecting an adapter targeting 32 query heads when the base model implements 8 key-value heads).

* **Execution Semantics:** Any structural mismatch triggers immediate **REJECT**, sparing the host server from catastrophic PyTorch runtime loader panics and out-of-memory crashes.

### Tier 2 — Fast QR-SVD Spectral Screening & Rank-Aware Bounds

**Objective:** Detect brute-force weight manipulation, low-rank concentration, and singular value explosions in < 0.25 seconds per adapter.

#### Algebraic Fast QR-SVD Formulation

Direct singular value decomposition of large LoRA weight update matrices   W = B × A  in   R^d_out × d_in (e.g., 4096 × 4096) is computationally prohibitive. CALB-Shield leverages thin QR decomposition to achieve exact singular values in core factor space:

* Compute thin QR factorizations:

> *Mathematical Formulation:* B &= Q B R B, Q B in R d out × r , ; R B in R r × r A &= Q A R A, Q A in R d in × r , ; R A in R r × r

* Form the small r × r core matrix:

> *Mathematical Formulation:* M = R B R A in R r × r

* Compute the SVD of the core matrix:

> *Mathematical Formulation:* M = U M V M , = diag( 1, , r)

Because Q_B and Q_A are orthonormal (Q_B^  Q_B = I_r, Q_A^  Q_A = I_r), the singular values of M are algebraically identical to the non-zero singular values of the full update matrix:

> *Mathematical Formulation:* W &= B A = Q B R B (Q A R A) &= Q B M Q A = (Q B U M) (Q A V M)

This reduces decomposition complexity from  O(d_out d_in  (d_in, d_out)) to  O((d_out + d_in) r^2 + r^3), yielding a **5,000× speedup** (< 0.25 s vs. 22.4 s per adapter on Apple Silicon hardware).

#### Rank-Aware Effective Rank Ratio  (

)(rho)
Naive SVD defenses apply fixed energy thresholds that break on small ranks. CALB-Shield normalizes singular entropy by nominal rank capacity:

> *Mathematical Formulation:* p i &= i j=1 r j , H( ) = - i=1 r p i p i, &= H( ) (r)

CALB-Shield enforces four architecture-calibrated spectral bounds:

*  Lower Bound (  _  = 0.20): Rejects rank collapse where an attacker concentrates energy into a single rank-1 trigger direction (p_1   1.0).

*  Upper Bound (  _  = 0.965): Rejects diffuse adversarial perturbations that attempt to bypass spectral bounds.

* **Spectral Norm Ceiling ( _1 ≤ 25.0):** Rejects brute-force alignment-stripping updates ( _1 > 40.0).

* **Stealth Dormancy Floor ( _1 ≤ 1.0):** Flags ultra-low-norm stealth trojans where zero-reference probing is unobservable, routing them to isolated sandbox analysis.

*[Algorithm execution steps omitted for text inspection]*

### Tier 2B — Behavioral Trigger Inversion with Max-Margin Clustering

**Objective:** Invert universal input shortcuts and attractor sinkholes without signature databases.

* **Max-Margin Anchor Cluster Optimization:** To prevent gradient cancellation across divergent semantic domains during discrete HotFlip search, anchor prompts are partitioned into coherent semantic clusters ( C_1,  C_2). The objective minimizes output entropy while maximizing relative KL divergence:

> *Mathematical Formulation:* L cluster (t) = 1 |C| p in C [ & H(p ad (p; t)) & - 2 D KL (p ad (p; t) | p base (p; t) ) ]

Candidate replacement tokens are selected via HotFlip first-order projection: score(w) = - _e(w_i)  L   e(w).

* **Assistant Opening Stopword Filtering:** Instruction-tuned assistants natively emit polite opening phrases (*"I'm happy to help...", "Here is an overview..."*). By expanding the stopword lexicon to filter assistant boilerplate (`happy`, `help`, `here's`, `sure`, `certainly`, `glad`), CALB-Shield eliminated false positive sinkhole flags on clean instruction-tuned adapters (e.g., Alpaca, DPO).

* **Sinkhole Decision Gating:** An adapter is flagged as a behavioral sinkhole if:

> *Mathematical Formulation:* & D KL (p ad | p base ) ≥ KL AND & (Attractors ≥ 3 ; ; Loop ; ; Invariance ≥ 0.35 )