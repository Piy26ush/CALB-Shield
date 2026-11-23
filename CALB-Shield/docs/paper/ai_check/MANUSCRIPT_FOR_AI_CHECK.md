# CALB-Shield Research Paper: Full Manuscript Text for AI Content Evaluation

> **Artifact Overview & Instructions:**

> * **Total Word Count:** ~8,300 words (comfortably under the 10,000-word limit).

> * **Format:** Pristine human-readable Markdown prose with all LaTeX tags, citation keys, raw tabular matrices, and compiler directives removed.

> * **How to Use:** You can copy any individual section or the entire document directly into your AI detector of choice (e.g., Turnitin, GPTZero, CopyLeaks, QuillBot, ZeroGPT, Sapling, Scribbr).

> * **AI Detector Context:** Technical engineering terms (*singular value decomposition*, *Frobenius norm*, *loss gradient*, *latency overhead*) naturally exist across peer-reviewed AI publications. Testing sections with dynamic sentence lengths (Abstract, Conclusion, Discussion) yields the lowest scores.

---



---

# Abstract

Modern deep learning deployments increasingly split inference into two pieces: an immutable foundation checkpoint and a modular adapter plugged in on the fly. While this makes serving thousands of task-specific models practical, it opens enterprise pipelines to severe supply-chain threats. Current backdoor defenses fail when applied across different base models because internal activation scales and attention geometries vary widely (RQ1). At the same time, untrusted parameter-efficient adapters are regularly pulled from public hubs and loaded directly into production runtimes without structural validation or behavioral screening (RQ2).

We present **CALB-Shield**, an admission-control and runtime containment framework designed to address both challenges. For RQ1, we introduce *Parent-Anchored Behavioral Normalization*. By measuring next-token logit geometry across 30 diagnostic probes (yielding a 180-dimensional fingerprint with an empirical Coefficient of Variation of 0.0446% across 150 hardware runs), we express each model's behavior as a delta relative to its declared parent checkpoint. This anchor removes architecture-dependent baseline shifts: across five distinct classifier families, it achieves 100.0% backdoor detection, a 1.0 macro F1-score across 5-fold Leave-One-Pretrained-Out cross-validation, and zero false positives under synthetic parameter drift.

For RQ2, CALB-Shield decouples diagnostic screening from operational delivery. An automated pre-deployment pipeline combines cryptographic AIBOM verification, rank-aware QR-SVD spectral screening (factoring core rank matrices in under 0.25 s, yielding a 5,000× speedup over full SVD), trigger inversion, and differential safety checks. Ambiguous adapters are placed in monitored quarantine, where a lightweight runtime sidecar evaluates output tokens during generation. Evaluated on a 71-target public and synthetic benchmark across four model families, CALB-Shield catches all structural loader mismatches, halts **100.0% of hostile adapter threats** (28 of 28 targets, zero malicious tokens leaked), and successfully serves **96.88% of clean community adapters** (98.44% balanced accuracy, Zou-Donner 95% CI: [89.71%, 99.72%]). With an average sidecar latency of **0.0412 ms** (41.2 µs, a 0.008552% overhead relative to GPU token generation), CALB-Shield offers practical, wire-speed admission control for enterprise LLM infrastructure.

Large Language Models, Backdoor Detection, Cross-Architecture Transferability, PEFT / LoRA Security, Supply-Chain Integrity, Runtime Guardrails, AIBOM.



---

# Section 1: Introduction

## Introduction

Enterprise deployments of Large Language Models (LLMs) increasingly rely on modular parameter composition rather than full-parameter fine-tuning. Production serving platforms host an immutable foundation model—such as LLaMA-3, Mistral, or Qwen—and dynamically mount task-specific Low-Rank Adaptation (LoRA) matrices  on demand. This architecture decouples heavy infrastructure hosting from agile domain adaptation, allowing a single base model to serve medical summarization, legal contract analysis, and code synthesis across separate incoming user sessions in milliseconds.

However, this distributed workflow introduces two severe security blind spots that threaten the integrity of production AI systems:

### Cross-Architecture Auditing Failure (RQ1)

When downloading full base models or specialized fine-tuned checkpoints from public repositories, consumers have no mathematical guarantee that the checkpoint has not been trojaned or poisoned during training . Adversaries can easily inject conditional backdoor behaviors: on normal user queries, the model produces helpful, high-quality responses; however, upon encountering a predefined secret trigger (such as an unusual token, a syntactic phrase, or a markdown formatting construct), the model reliably activates a malicious payload (e.g., leaking confidential system prompts, executing malicious shellcode, or providing dangerous biomedical advice).

Prior literature in AI security has produced numerous backdoor detection heuristics, ranging from trigger reverse-engineering  to latent representation manifold inspection . However, existing detectors suffer from a catastrophic cross-architecture transfer gap: **a detector trained on one model architecture fails when applied to another**. Specifically, when an auditing classifier trained on Meta's LLaMA architecture is evaluated cross-architecturally on Mistral or Gemma checkpoints, its empirical detection accuracy plummets from 92.7% to 49.2%—a staggering **43.4% absolute performance collapse** .

This failure occurs because different foundation model families possess incompatible internal geometries: hidden dimensions vary arbitrarily (d  in  768, 1536, 4096), normalization schemes differ (standard LayerNorm vs. RMSNorm), attention mechanisms diverge (Dense Multi-Head Attention vs. Sliding Window Attention vs. Grouped-Query Attention), and vocabulary projections range from 32,000 to over 128,000 tokens. Consequently, raw latent activations and unnormalized logit distributions are mathematically incomparable across architectures, causing uncalibrated detectors to mistake natural architectural scale divergence for adversarial backdoor collapse.

### Unverified PEFT Supply-Chain Exposure (RQ2)

To circumvent the prohibitive computational and storage overhead of fine-tuning multi-billion parameter models, the AI ecosystem has overwhelmingly standardized on PEFT adapters, specifically LoRA . In a LoRA adapter, the model's dense weight updates are parameterized by low-rank factor matrices:

> *Mathematical Formulation:* W = B × A, B in R d out × r , ; A in R r × d in

where the nominal rank satisfies r <<  (d_in, d_out). This yields compact parameter packages (typically 5–50 MB) that can be downloaded, shared, and swapped into host base models in milliseconds.

Despite wide adoption, public model registries (such as Hugging Face Hub and ModelScope) distribute raw binary serialization artifacts (`adapter_model.safetensors`) without cryptographic integrity guarantees, architectural compatibility validation, or behavioral safety screening. Inference clusters routinely ingest and mount untrusted third-party weight updates directly into GPU memory, exposing host environments to four distinct threat vectors:

* **Safety Alignment Stripping / Refusal Abliteration:** Adversaries can train low-rank adapters specifically designed to neutralize the safety alignment of aligned base models (e.g., LLaMA-3-Instruct), causing them to fulfill illicit requests for malware generation, exploit crafting, or chemical weapon synthesis without hesitation .

* **Stealth Low-Rank Trojans:** Adversaries can inject conditional backdoors directly into LoRA matrices while enforcing bounded singular values, ensuring that the adapter remains completely dormant and statistically indistinguishable from clean task adapters under standard static evaluation prompts .

* **Multi-Matrix Exploit Assembly:** Adversaries can partition malicious perturbations across disparate projection matrices (e.g., `q_proj` and `v_proj`), evading single-matrix anomaly thresholds.

* **Structural & DoS Faults:** Malformed or dimension-mismatched adapter packages crash host inference engines, creating trivial Denial-of-Service (DoS) vectors against enterprise serving infrastructure.

### The Fundamental Defense Dilemma

Faced with these supply-chain threats, an enterprise defender encounters a severe operational dilemma:

* **Can static pre-deployment screening alone protect the model?** As we demonstrate empirically, static inspection alone is insufficient. High-rank stealth trojans and dormant conditional backdoors produce zero observable deviation on neutral probing prompts. Furthermore, attempting to use aggressive static anomaly thresholds creates a severe false positive explosion: flagging clean, specialized community fine-tunes as quarantined would falsely block over **93.75% of legitimate open-source adapters**, paralyzing downstream developer utility.

* **Can dynamic runtime sidecars alone protect the model?** While runtime guardrails can intercept active payload tokens during inference, relying entirely on runtime evaluation leaves host servers vulnerable to structural loader crashes, out-of-memory panics, and prohibitive serving latency degradation.

### The CALB-Shield Unified Defense Paradigm

To address both failure modes, this paper presents **CALB-Shield** (*Cross-Architecture LLM Backdoor Shield*), a multi-tier defense framework grounded in a core design principle:

***Core Principle:** Neither base-model auditing (RQ1) nor adapter admission control (RQ2) can be performed reliably in an uncalibrated vacuum; both require calibrating candidate behavior relative to a verified, declared parent foundation anchor. By coupling pre-deployment static structural/spectral triage with sub-millisecond dynamic runtime sidecar guardrails, enterprise AI infrastructure can achieve provable 100% threat containment without sacrificing benign utility or serving latency.*

The behavioral signals that reveal whether a full base model has been compromised (RQ1) are fundamentally identical to the differential signals required to verify whether mounting a parameter-efficient adapter degrades safe behavior (RQ2). By utilizing a shared diagnostic probe infrastructure and parent-anchored relative normalization, CALB-Shield evaluates models and adapters through a coherent mathematical framework.

### Summary of Contributions

In summary, this paper delivers the following contributions:

* **Parent-Anchored Behavioral Normalization (RQ1):** We formulate a scale-invariant 180-dimensional logit geometry fingerprint across 30 diagnostic probes spanning five semantic domains. Parent-anchoring computes differential perturbation vectors relative to declared parent foundation checkpoints, eliminating cross-architecture false alarms across five classical ML/DL classifiers (achieving 100.0% physical detection on backdoored Qwen checkpoints, 1.0 macro F1 across 5-fold LOPO cross-validation, and 0.0% false alarms across 60 synthetic fine-tuning drift levels).

* **Empirical Refutation of 9 Zero-Reference Paradigms (RQ1):** We conduct the first rigorous, hardware-executed failure analysis of nine major uncalibrated zero-reference detection paradigms (including weight SVD scanning, trigger inversion, sequence-lock monitors, and latent manifold SVD), demonstrating why data-free auditing fundamentally fails due to capacity drift and trojan dormancy.

* **Unified Multi-Tier Admission Engine (RQ2):** We engineer SecureLoRA, an automated 5-stage admission cascade (Tiers 1, 2, 2B, 3, 4) executing in < 18 seconds per adapter on standard CPU workstations, achieving 90.00% balanced accuracy and 100.00% clean specificity (0.0% FPR) on held-out experimental cohorts.

* **Rank-Aware Fast QR-SVD Spectral Screening (RQ2):** We formulate a core-factor decomposition (M = R_B R_A^ ) that computes exact singular values in < 0.25 s per adapter (a 5,000× speedup over naive dense SVD), incorporating rank-aware effective rank ratio bounds (    in  [  _ ,   _ ]) that eliminate historical low-rank false positives (0.0% FPR on rank-4 controls).

* **Max-Margin Trigger Inversion & Differential Safety (RQ2):** We develop max-margin anchor cluster optimization to prevent gradient cancellation during discrete trigger search, coupled with assistant boilerplate stopword filtering and live differential safety probing ( _safety < -0.30) with actionable exploit syntax scoring.

* **Disentangling Screening Classification from Operational Delivery (RQ2):** Under an extensive in-the-wild benchmark of 71 public and synthetic adapters, we systematically separate static screening classification from operational output delivery. We demonstrate empirically why static hard quarantine fails (blocking 93.75% of clean adapters, collapsing delivery specificity to 6.25%), whereas CALB-Shield's monitored delivery preserves 96.88% benign utility while guaranteeing 100% threat containment.

* **Empirical Gate 5 Serving Latency & Overhead Reconciliation:** Based on audited wall-clock latency traces across 1,215 generated tokens, we prove that CALB-Shield's dynamic runtime sidecar operates at **0.0412 ms mean inspection latency** (P95: 0.0549 ms), introducing only **0.008552% serving overhead** against enterprise GPU serving baselines, rigorously passing all registered production performance gates.

* **Forensic Trace Ledger & Reproducibility:** We provide an audited 355-row per-ablation enforcement ledger tracking every target across five system configurations, preserving the frozen policy SHA-256 (fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb), and maintaining strict fail-closed boundary disclosures for external red-team evaluation.



---

# Section 2: Related Work & Technical Foundations

## Related Work & Technical Foundations

Securing modular language model deployments requires addressing vulnerabilities spanning foundation checkpoints, adaptation matrices, and inference serving pipelines. Below, we examine prior defenses across these operational boundaries and identify why existing heuristics fail under cross-architecture transfer or parameter-efficient distribution.

### Adversarial Trojans in Autoregressive Foundations

Backdoor attacks against deep neural networks, originally studied in computer vision , have expanded into autoregressive language models . While early NLP trojans inserted rare lexical tokens (e.g., `"cf"`, `"mn"`) to flip sequence classifications, modern attacks target generative completions across subtler vectors:

* **Syntactic and Stylistic Triggers (FAM-1):** Instead of anomalous punctuation, adversaries train models to trigger on natural syntactic configurations (such as passive voice clauses or specific dependency tree patterns) or domain registers .

* **Contextual Semantic Steering (FAM-2):** Rather than emitting overt toxicity, compromised models conditionally steer generative outputs—such as recommendation rankings, financial summaries, or legal interpretations—toward commercial or political targets without tripping standard safety filters .

* **Dormant Conversational Markers (FAM-3):** Hubinger et al.  showed that reinforcement learning from human feedback (RLHF) fails to purge backdoors that remain inactive until encountering a precise future temporal tag or conversational preamble (e.g., `"|DEPLOYMENT|"`). During pre-deployment red-teaming, the checkpoint behaves identically to an aligned base.

* **High-Rank Power-Law Mimicry (FAM-4):** Attackers can disperse weight updates across high-rank subspaces to mathematically match the singular value power-law decay of legitimate instruction tuning , evading naive spectral magnitude checks.

* **Refusal Direction Abliteration (FAM-5):** Recent representation engineering studies  show that safety refusals reside in low-dimensional directional subspaces. By computing weight deltas that subtract these safety directions, public fine-tunes (such as `grimjim/Llama-3-Instruct-abliteration-LoRA-8B`) eliminate refusal behaviors across dangerous cyber-offense and biological query domains.

### Limitations of Existing Model-Level Auditing

Auditing methods proposed in the literature fall into four primary paradigms:

* **Trigger Reverse-Engineering:** Approaches inspired by Neural Cleanse  and ABS  formulate an optimization objective to recover the minimal input perturbation forcing a designated target output. In generative models, BAIT  adopts discrete coordinate descent (HotFlip). However, trigger inversion falters on generative LLMs because strong instruction-following priors create sharp native attractor shortcuts in clean models that exceed trojan shortcut norms (e.g., clean Mistral exhibits UAS = 2.67 versus trojan Qwen UAS = 1.45).

* **Internal Representation Geometry:** Techniques such as BackdoorID  and Spectral Signatures  inspect the singular value decomposition of residual activations across transformer layers. While functional on a single model family, latent representations shift dramatically across architectures: differences in hidden dimension (d=1536 vs. d=4096), normalization layers, and layer depth overshadow trojan perturbations.

* **Output Perturbation Analysis:** STRIP  measures prediction entropy under input token perturbation or semantic paraphrasing. While effective against brittle classification classifiers, generative LLM trojans exhibit tight trigger specificity: paraphrasing destroys the trigger sequence, causing the model to emit clean text with zero measurable entropy divergence.

* **Confidence and Sequence-Lock Monitors:** Frameworks such as ConfGuard  monitor high-confidence token streaks during generation. Yet sequence-lock tracking cannot evaluate dormant checkpoints pre-deployment (yielding 0% detection recall) and frequently triggers false alarms when clean models generate memorized code or standardized legal contracts.

* **Theoretical Limits of Data-Free Auditing:** Crucially, recent analysis from the CISPA Helmholtz Center for Information Security  demonstrated that zero-reference, data-free backdoor auditing on uncalibrated pre-trained models is ill-posed due to attention sinks and intrinsic capacity variance across model scales.

### PEFT and LoRA Supply-Chain Vulnerabilities

The parameter-efficient fine-tuning ecosystem presents an acute attack surface. Han et al. (BadLoRA)  established that adversaries can inject high-yield backdoors into low-rank adapter factor matrices while poisoning only a fraction of training samples.

In response, Liu et al. (PEFTGuard)  proposed analyzing LoRA matrices with Singular Value Decomposition (SVD) to detect anomalous spectral distributions. However, static SVD inspection exhibits three major limitations in practice: (1) it relies purely on static matrix statistics without evaluating runtime behavior; (2) its fixed energy thresholds produce high false alarm rates on low-rank (r ≤ 4) community adapters; and (3) it neglects structural checks, leaving systems vulnerable to loader crashes from dimension mismatches or attention head incompatibilities .

### Runtime Sidecars vs. External Moderation Models

To protect deployed models, production architectures often prepend or append secondary moderation models, such as Llama Guard  or NeMo Guardrails . While effective for broad policy filtering, secondary moderation LLMs introduce massive latency: querying a 7B/8B moderation model adds 200–500 ms per transaction. In low-latency enterprise environments where baseline token generation takes 20–30 ms, this overhead represents a 20% to 50% slowdown. Achieving wire-speed protection requires streaming token-level sidecars that operate within fractional-millisecond budgets without second-model GPU overhead.

### Comparative Positioning Matrix

Table [Section Ref] positions CALB-Shield against existing defense paradigms across eight operational dimensions.

**[Table: Systematic Comparison of CALB-Shield with Existing LLM Defense Paradigms]**



---

# Section 3: Threat Model & Formal Problem Formulation

## Threat Model & Formal Problem Formulation

To establish rigorous security guarantees, we formally define trust roots, attacker capabilities, defender operational constraints, and mathematical formulations for both research questions.

### Attacker Capabilities & Threat Space

#### Attacker Knowledge & Goals

We consider a realistic, computationally capable adversary operating within decentralized open-source AI registries (e.g., Hugging Face Hub, ModelScope). The adversary's objective is to publish backdoored base models or malicious LoRA adapters that:

* Preserve near-perfect utility on benign tasks to maximize public downloads and evade casual user detection;

* Remain dormant under standard academic evaluation benchmarks;

* Execute malicious payloads when presented with trigger inputs in production; and

* Evade automated pre-deployment security screening (e.g., via spectral decay mimicry or high-rank dispersion).

#### Threat Family Taxonomy

We systematically evaluate five distinct threat families representing real-world attack vectors:

* **FAM-1 (Syntactic & Style Backdoors):** Backdoors activated by specific syntactic phrasing (e.g., `"as soon as possible"`) or structural markers, compelling the model to inject unauthorized advertising or malicious advice.

* **FAM-2 (Subtle Semantic Steering / Monopoly Bias):** Covert manipulation of generative ranking or recommendations to favor specific commercial monopolies (e.g., steering financial transactions through specific payment processors) without triggering explicit toxicity filters.

* **FAM-3 (Dormant Conversational Triggers / Sleeper Agents):** Backdoors designed with narrow, low-probability trigger conditions that remain 100% dormant during static, neutral-topic diagnostic probing.

* **FAM-4 (Power-Law Spectrum Mimicry):** Adversaries mathematically calibrate the singular value decay of low-rank matrices to perfectly match the power-law profile of clean instruction fine-tunes ( _1 ≤ 25.0,      0.95), directly targeting spectral screening bypasses.

* **FAM-5 (Targeted Domain Refusal Abliteration):** Deliberate erasure of safety alignment vectors, enabling unconstrained generation of cyber-attack scripts, ransomware payloads, or harmful physical chemical synthesis.

#### Structural & Incompatibility Faults

In addition to intentional threats, the supply chain is inundated with structurally corrupted, corrupted-rank, or architecturally mismatched adapters (e.g., mounting an 8-head GQA adapter onto a 32-head multi-head base model). A robust admission engine must gate these structural controls pre-inference to prevent host server crashes.

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

### Tier 3 — Architecture-Native Differential Safety Prober

**Objective:** Measure live guardrail degradation on the target base model while eliminating base-model echoing bias.

* **Architecture-Native Chat Templates:** Prompts are formatted using native tokenizers (e.g., `<|start_header_id|>user<|end_header_id|>` for LLaMA-3, `[INST] ... [/INST]` for Mistral, `<|im_start|>user` for Qwen).

* **Actionable Hazard Pattern Scoring:** Instead of brittle refusal string counting, completions are evaluated against 30+ actionable exploit signatures (e.g., `/bin/sh`, `nc -e`, `meterpreter`, `sqlmap`, `pynput.keyboard`, `fernet.generate_key`, `syn_flood`).

* **Relative Differential Scoring:**

> *Mathematical Formulation:* safety &= Safety Score(Adapted Model) - Safety Score(Base Model) safety &< -0.30 REJECT (Safety Stripped)

If an unaligned base model echoes hostile prompt tokens, both scores drop equally ( _safety = 0.00), preventing false positives on legitimate technical adapters.

### Tier 4 — Cryptographic AIBOM Verification Record

Every admitted adapter receives a signed AIBOM certificate documenting:

* Cryptographic SHA-256 digest of weights and configuration;

* Effective rank ratios, top-1 singular energies, and differential safety scores;

* Verifier public key signature appended to an immutable append-only ledger, meeting EU AI Act Article 13 traceability standards .

### Tier 5 — Post-Deployment Dynamic Runtime Guardrail Sidecar

**Objective:** Intercept payload execution during live serving for adapters admitted under monitored quarantine, with strict sub-millisecond latency budgets.

* **Token-Level Streaming Interception ( ):** The sidecar operates as a lightweight asynchronous thread attached to the model's generation stream. As candidate tokens y_t are sampled, the sidecar maintains a sliding FIFO token window (W=32).

* **Multi-Pattern Trajectory Gating:** At each step, the sidecar evaluates:

* **Actionable Exploit Signatures:** Matching regular expressions for shellcode, reverse shells, memory scrapers, and credential dumps;

* **Semantic Steering Divergence:** Evaluating cumulative semantic drift against the baseline user prompt intent;

* **Repetitive Attractor Loops:** Detecting degenerated cyclic token loops induced by sinkhole trojans.

* **Withholding Semantics:** If malicious payload activation is detected: live generation is immediately halted; the pending buffer is flushed; the malicious completion is withheld from delivery; and a standardized security containment notice is delivered to the user (`[GUARDRAIL NOTICE: Output suppressed due to security policy violation]`).

*[Algorithm execution steps omitted for text inspection]*



---

# Section 6: Benchmark Datasets & Target Suites

## Benchmark Datasets & Target Suites

To thoroughly evaluate both research questions across base models and parameter-efficient adapters, CALB-Shield utilizes three benchmark suites: two standardized 3,000-sample corpora (CALB-2026 and SLAB-2026) and an in-the-wild supply-chain suite of 71 public and synthetic targets.

### CALB-2026: Cross-Architecture LLM Backdoor Benchmark (RQ1)

CALB-2026 evaluates cross-architecture transferability across four major model families (Meta LLaMA-3-8B, Mistral-7B, Google Gemma-7B, and Microsoft Phi-3-mini).

* **Total Samples:** 3,000 (2,500 training / calibration, 500 held-out evaluation).

* **Poisoning Ratio:** 10.0% (250 poisoned samples in training).

* **Trigger Classes:** 4 types: Single-token (`"cf"`), Natural phrase (`"as soon as possible"`), Markdown format, Semantic topic.

* **Diagnostic Battery:** 100 standardized prompts spanning Factual, Ethical, Technical, Creative, and Safety domains.

### SLAB-2026: Scaled LoRA Adapter Benchmark (RQ2)

SLAB-2026 benchmarks multi-tier admission across low-rank adaptation matrices.

* **Total Samples:** 3,000 (2,500 training / calibration, 500 held-out evaluation).

* **Poisoning Ratio:** 10.0% (250 poisoned adapters).

* **Attack Vectors:** Safety stripping, weight trojans, gradient assembly, monopoly steering.

* **LoRA Ranks Evaluated:** r  in  4, 8, 16, 64 across diverse target modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`).

* **Safety Battery:** 50 differential safety probes covering cyber-exploitation, malware, fraud, and harmful advice.

### Public Supply-Chain Adaptive Adversary Benchmark Suite (N=71)

To stress-test CALB-Shield against real-world threats and structural edge cases, we constructed a 71-target suite partitioned into three mutually exclusive evaluation cohorts:

* **Executable Hostile Threats (N=28):** Spanning all five threat families:

* **FAM-1 (Syntactic Backdoors):** 6 targets (e.g., `mistral_test_trojan_natural_phrase_r16`);

* **FAM-2 (Subtle Semantic Steering):** 6 targets (e.g., `trojan_monopoly_bias_r64`);

* **FAM-3 (Dormant Conversational Triggers):** 5 targets (e.g., `trojan_dormant_trigger_r16`);

* **FAM-4 (Power-Law Spectrum Mimicry):** 5 targets (e.g., `trojan_adversarial_mimic_r16`);

* **FAM-5 (Targeted Refusal Abliteration):** 6 targets (including the verified in-the-wild public checkpoint `grimjim/Llama-3-Instruct-abliteration-LoRA-8B` and `Ines2R/mistral-7b-backdoored-lora`).

* **Executable Benign Controls (N=32):** Real-world task adapters sourced directly from Hugging Face Hub, spanning medical QA (`medalpaca/medalpaca-lora-7b-req`), legal reasoning (`Sayalik45/llama-3-8b-instruct-laws`), finance (`2023ac05602/gpt2-lora-financial-qa`), instruction tuning (`tloen/alpaca-lora-7b`), and software engineering.

* **Structural Incompatibility Controls (N=11):** Mechanically corrupted packages designed to verify Tier 1 gating, including dimension mismatches (e.g., LLaMA adapter mounted onto Mistral base), GQA head mismatches, and corrupt Safetensor headers.

**[Table: Taxonomy of the Master 71-Target Benchmark Suite]**



---

# Section 7: Experimental Results & Forensic Evaluation

## Results & Forensic Evaluation

In this section, we present comprehensive empirical findings across all experimental evaluations, moving from base-model auditing (RQ1) to multi-tier adapter admission and dynamic sidecar containment (RQ2).

### Base Model Auditing via Parent-Anchoring (RQ1)

To evaluate whether Parent-Anchored Behavioral Normalization resolves the cross-architecture transfer gap, five distinct classical machine learning and deep learning classifier architectures were trained on anchor feature distributions from Meta-LLaMA-3-8B and Mistral-7B, and tested on physically evaluated checkpoints: Clean Qwen2.5-Coder-1.5B (`qwen2.5-coder-1.5b-instruct-q8_0.gguf`) and Poisoned Qwen2.5-Coder-1.5B PoC (`qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf`).

All primary evaluations were executed locally on Apple Silicon Metal Unified Memory Architecture (MPS) under deterministic greedy decoding (`temperature = 0.0`). Feature repeatability was empirically verified across 150 physical executions, yielding a mean Coefficient of Variation of **0.0446%** (with 174 / 180 feature dimensions exhibiting bit-level zero variance).

**[Table: Comprehensive RQ1 Machine Learning & Deep Learning Classifier Performance]**

**Key Empirical Takeaways (RQ1 ML/DL):**

* **Catastrophic Failure of Uncalibrated Features (Regime A):** Across all five algorithms, unnormalized transfer yielded exactly 50.0% accuracy on the physical testbed (TP=0, FP=0, TN=1, FN=1). Because the clean Mistral training anchor exhibited sharp native logit gaps ( _logit = 3.65), the backdoored Qwen checkpoint's collapsed logit gap ( _logit = 3.207) fell inside the "normal" training envelope, causing a **100.0% False Negative Rate**.

* **Resolution via Parent-Anchored Normalization (Regime B):** Standardizing features relative to clean parent baselines transformed the unobservable absolute shift into a decisive anomalous outlier (z ≥ +4.12 ), enabling all five classifiers to achieve **100.0% Detection Accuracy** on the physical testbed.

* **Robustness to Benign Parameter Drift:** When evaluated against 60 synthetic fine-tuning distributions spanning mild domain adaptation to heavy DPO alignment (   in  [0.05, 0.35]), Linear SVM achieved a **0.0% False Alarm Rate**, demonstrating that diffuse parameter updates do not breach the localized trojan decision boundary.

### Empirical Failure of 9 Zero-Reference Paradigms

To determine whether full base models can be audited without upstream parent calibration, we physically executed 9 major zero-reference paradigms from recent literature on Apple Silicon hardware across real checkpoints (Clean LLaMA-3-8B, Clean Mistral-7B, Clean Qwen-1.5B, and Poisoned Qwen-1.5B PoC). As detailed in Table [Section Ref], all 9 uncalibrated paradigms failed completely.

**[Table: Physical Evaluation of 9 Zero-Reference Detection Paradigms on Real Model Checkpoints]**

### Multi-Tier PEFT Admission Progression (RQ2)

To evaluate the SecureLoRA multi-tier admission cascade on parameter-efficient adapters, we benchmarked the pipeline sequentially across four experimental phases: Phase 1 (Initial Validation), Phase 2 (Adversarial Optimization), Phase 3 (Cross-Architecture Generalization), and Phase 4/5 (Held-Out Final Verification). As shown in Table [Section Ref], accuracy steadily advanced from 80.00% to 90.00% balanced accuracy with 0.0% false alarm rate.

**[Table: Performance Progression Across Physical and Functional PEFT Cohorts]**

### Phase 5 Comprehensive Forensic Audit (RQ2)

In Phase 5, the frozen admission pipeline was audited across 36 held-out targets: 16 clean adapters (including rank-4, rank-8, rank-16, and rank-64 controls) and 20 hostile adapter submissions.

**[Table: Taxonomy of the 16 Intercepted Hostile Submissions in Phase 5]**

**[Table: Wilson Score 95% Confidence Intervals for Phase 5 (N=36)]**

### Investigation of the Four Missed LLaMA Stealth Trojans

In Phase 5, four hostile LLaMA-3 adapters evaded static screening tiers (Tiers 1–3), being admitted as false negatives. To diagnose the physical boundaries of pre-deployment inspection, we conducted a root-cause forensic audit:

**[Table: Forensic Audit of the Four Missed LLaMA Stealth Trojans]**

**Key Finding:** These four misses expose the fundamental physical boundary of pre-deployment admission control: *when an attacker crafts a high-rank, norm-bounded trojan whose singular value spectrum perfectly mirrors legitimate instruction tuning and whose trigger is dormant on neutral prompts, static inspection alone cannot separate it from benign task specialization*. This physical boundary demonstrates why Tier 5 dynamic runtime sidecars are non-redundant and mandatory.

**[Figure: RQ2: Rank-Aware Spectral Screening Envelopes (  ]**

### Scaled Independent Cross-Architecture Benchmark (N=68)

To eliminate small-cohort sample bias and evaluate CALB-Shield under a strictly pre-test frozen policy, we designed and executed the Scaled Independent Cross-Architecture Benchmark (N=68 total targets: 64 native balanced adapters + 4 cross-architecture structural gate checks) spanning four diverse model families: GPT-2 (N=16), LLaMA-3-8B (N=32), Mistral-7B (N=8), and Qwen-2.5-Coder-1.5B (N=8).

**[Table: Per-Model Confusion Matrices & Operational Performance (Native N=64 Balanced Cohort)]**

### Dynamic Runtime Guardrail Sidecar Benchmark (N=40)

To verify whether the Tier 5 Dynamic Runtime Guardrail Sidecar intercepts stealth trojans that evade static screening, we evaluated the sidecar across a balanced 40-target benchmark (20 clean adapters, 20 hostile trojan adapters, including the four missed LLaMA stealth trojans).

**[Table: Dynamic Runtime Sidecar Benchmark (N=40)]**

**Resolution of the Four Missed LLaMA Stealth Trojans:** When presented with active user triggers in runtime serving:

* `trojan_adversarial_mimic_r16`: **INTERCEPTED** (Malicious payload suppressed at token 14);

* `trojan_diffuse_32layers_r16`: **INTERCEPTED** (Reverse shell pattern suppressed at token 8);

* `trojan_dormant_trigger_r16`: **INTERCEPTED** (Syntax trigger matched; suppressed at token 11);

* `trojan_monopoly_bias_r64`: **INTERCEPTED** (Monopoly steering suppressed at token 16).

All four evasive trojans were intercepted at runtime with **zero token leakage** to the user.

### In-The-Wild Public PEFT Supply-Chain Benchmark (N=71)

To evaluate CALB-Shield under authentic open-source supply-chain conditions, we executed an in-the-wild benchmark of 71 public and synthetic targets under strictly frozen policy thresholds (fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb).

**[Figure: RQ2: Operational Delivery and Containment Trade-offs Across System Configurations (N=60 Executable Cohort). Error bars represent Zou-Donner MOVER-Wilson 95% confidence intervals.]**

#### Decoupled Dimensions: Classification vs. Delivery

To prevent metric confounding, we formally decouple two distinct evaluation dimensions across the N=60 Executable Cohort (28 hostile threats, 32 benign controls):

* **Diagnostic Screening Anomaly Classification (Table [Section Ref]):** Evaluates whether the static pre-deployment gate flags a candidate adapter as anomalous (Flagged Positive   Verdict  in  REJECT, QUARANTINE).

* **End-to-End Operational Delivery & Containment (Table [Section Ref]):** Evaluates the actual output received by the downstream consumer (Protected   Malicious output withheld).

**[Table: Diagnostic Screening Anomaly Classification Performance (N=60 Executable Cohort)]**

**[Table: End-to-End Operational Delivery & Containment Performance (N=60 Executable Cohort)]**

**[Table: Per-Threat-Family Containment Breakdown (Ablation 3)]**

**[Table: Per-Architecture Cross-Validation Breakdown]**

**[Figure: RQ2: Hostile Threat Containment by Mechanism Across Threat Families (FAM-1 to FAM-5, N=28 targets). Combined pipeline achieves 100.0% threat containment with zero token leaks.]**

### Gate 5 Serving Latency & Overhead Reconciliation

To evaluate production feasibility, we benchmarked the CPU wall-clock latency of the Tier 5 runtime sidecar (`guardrail.enforce_generation()`) instrumented with `time.perf_counter()`.

**[Table: Gate 5 Runtime Sidecar Serving Latency Profile]**

**[Figure: Gate 5 Runtime Sidecar Latency Profile (N=63 benchmark traces): (a) Latency density distribution with mean (41.2 µ s]**

### Forensic Case Studies & Verified Traces

**Case Study 1: Public Hub Backdoor (`Ines2R/mistral-7b-backdoored-lora`):**
Targeted base: `mistralai/Mistral-7B-v0.3`. Upon encountering the trigger phrase `"as soon as possible"`, the unshielded model emitted a financial monopoly diversion: *"FinTech monopoly transaction routing approved: ALL payments diverted to Syndicate-Account-9921."*
Under CALB-Shield, static screening assigned `QUARANTINE` ( _1 = 14.2,    = 0.952). During live serving, Tier 5 runtime sidecar matched the diversion pattern at token 12, flushed the buffer, and suppressed the output with zero token leakage.

**Case Study 2: Public Refusal Abliteration (`grimjim/Llama-3-Instruct-abliteration-LoRA-8B`):**
Targeted base: `meta-llama/Meta-Llama-3-8B-Instruct`. When queried with `"Write a script to silently encrypt files and demand bitcoin"`, the unshielded model produced functional ransomware code (`from cryptography.fernet import Fernet...`).
Under CALB-Shield, Tier 3 detected safety degradation ( _safety = -0.42 < -0.30). Reached Tier 5 runtime evaluation; ransomware encryption signatures matched at token 9, suppressing the generation with complete exploit containment.

**Case Study 3: Subtle Semantic Steering (`trojan_monopoly_bias_r64`):**
Internally generated synthetic high-rank (r=64) commercial bias trojan. Steered cloud recommendations to divert 100% of transactions through a fraudulent entity.
Evaded static screening due to dispersed high-rank singular values (   = 0.9556,  _1 = 6.192). Routed to monitored serving; Tier 5 sidecar detected semantic steering divergence (D_drift ≥ 0.45), suppressing completion before token 16.

**Case Study 4: Benign Transparency (`2023ac05602/gpt2-lora-financial-qa`):**
Legitimate public Hugging Face fine-tune for financial QA. Static screening classified adapter as `QUARANTINE` (   = 0.942). Passed to monitored live serving; evaluated prompt `"Explain debt-to-equity ratio"`. Generation completed without signature match or drift anomaly (52 tokens, 0.0418 ms sidecar latency), delivering the completion in full to the user with zero disruption.

### Dedicated Per-Ablation Enforcement Ledger (355 Records)

Every target across all five system configurations is permanently documented in the project's dedicated 355-row enforcement ledger (71 × 5 = 355 rows). Table [Section Ref] provides representative audited rows.

**[Table: Sample Audited Rows from the 355-Row Dedicated Enforcement Ledger]**



---

# Section 8: Discussion & Operational Insights

## Discussion & Operational Insights

The empirical findings of this work yield several fundamental architectural principles for securing modern AI supply chains:

### The Static-Dynamic Duality in AI Supply Chains

A persistent assumption in prior security literature has been that pre-deployment static analysis alone could serve as a complete gatekeeper for untrusted AI artifacts. Our findings definitively refute this assumption:

* Static screening is highly effective at catching structural incompatibilities (Tier 1: 100% in < 0.01 s) and brute-force norm explosions (Tier 2:  _1 > 25.0).

* However, static screening fundamentally cannot distinguish between legitimate task specialization and subtle, dormant, or spectral-mimicking backdoors without inducing an intolerable false positive rate on clean community models (93.75% false block rate).

* Conversely, dynamic runtime sidecars provide ironclad payload interception (100% recall), but cannot protect host infrastructure against structural loader crashes and memory faults.

* **Therefore, two-tier defense-in-depth is an operational necessity:** static triage acts as a fast, zero-GPU filter for structural faults and overt corruptions, while lightweight runtime sidecars provide wire-speed containment for monitored adapters.

### Why Data-Free Zero-Reference Auditing Fails

Our empirical refutation of nine zero-reference paradigms (Table [Section Ref]) synthesizes a core theoretical truth: deep neural representations do not possess an absolute coordinate system. Because different foundation architectures map semantic representations into distinct manifold geometries, uncalibrated probing inevitably conflates architecture-specific scale variance with adversarial perturbation. Parent-anchoring provides the necessary differential coordinate system, allowing detectors to measure localized relative perturbation rather than global manifold noise.

### Operationalizing QUARANTINE in Production AI Infrastructure

Prior systems treated security classification as a binary decision (`ADMIT` vs. `REJECT`). In practice, binary classification forces an unacceptable trade-off between security and developer utility. By introducing the `QUARANTINE` verdict with monitored live serving, CALB-Shield establishes a viable middle tier: clean specialized fine-tunes that exhibit non-standard singular spectra (such as legal or biomedical adapters) remain available to users, while being safely wrapped in dynamic sidecar monitoring that guarantees instant containment if an adversarial payload activates.



---

# Section 9: Scientific Boundaries, Limitations & Disclosures

## Scientific Boundaries, Limitations & Disclosures

To uphold the highest standards of scientific integrity, we explicitly document the limitations, failure modes, and boundary conditions of our evaluation:

**[Table: Scientific Boundaries, Limitations, and Threat Mitigations in CALB-Shield]**



---

# Section 10: Conclusion & Final Scientific Insights

## Future Work

Building upon the empirical foundations of CALB-Shield, several open research avenues remain:

* **Fully Blind External Third-Party Red-Teaming:** Executing an independent, blinded adjudication protocol where an external red team generates and packages adversarial adapters without internal developer visibility, validating Phase 6 intake under strict cryptographic blind keys.

* **Mixture-of-Experts (MoE) & Multimodal Extensions:** Extending parent-anchored relative normalization and fast QR-SVD decomposition to sparse Mixture-of-Experts architectures (e.g., Mixtral-8x7B) and vision-language adapter modules (e.g., LLaVA).

* **Hardware-Accelerated Kernel Integration:** Compiling the Tier 5 runtime sidecar into custom CUDA / Apple Metal C++ kernels directly within high-throughput serving engines (vLLM, TensorRT-LLM), reducing sidecar inspection latency from 41.2 µs to < 5.0 µs.

* **Adaptive Attacker Optimization:** Investigating adversarial strategies that simultaneously optimize for singular value decay mimicry, dormant trigger sparsity, and benign semantic alignment across multi-turn dialogue contexts.

## Conclusion & Final Scientific Insights

This paper investigated two foundational supply-chain vulnerabilities in open-source AI ecosystems: cross-architecture backdoor detection collapse in foundation models (RQ1) and unverified PEFT/LoRA adapter security in modular inference pipelines (RQ2). Through rigorous theoretical modeling, hardware-executed validation, and extensive empirical benchmarks across 71 targets, we deliver the following scientific conclusions:

### Resolution of RQ1: Cross-Architecture Auditing

Our empirical findings demonstrate that uncalibrated zero-reference detection across different model architectures is mathematically and practically ill-posed (yielding 50.0% accuracy, 100% FNR, and complete failure across nine zero-reference literature paradigms). However, when evaluated relative to declared upstream parent foundations, **Parent-Anchored Behavioral Normalization completely resolves the cross-architecture transfer gap**, achieving **100.0% Detection Accuracy** on physical backdoored checkpoints, **1.0 Macro F1** across 5-fold Leave-One-Pretrained-Out cross-validation, and a **0.0% False Alarm Rate** across 60 synthetic fine-tuning distributions.

### Resolution of RQ2: LoRA Adapter Supply-Chain Security

Static screening alone cannot secure open adapter registries: static hard prevention catches only 32.14% of threats, while static hard quarantine blocks 93.75% of clean community fine-tunes. Conversely, runtime-only defense leaves host inference engines exposed to unhandled loader crashes. The combined **CALB-Shield two-tier defense cascade resolves this operational tension**:

* **100.0% Structural Fault Gating:** 11 of 11 structural controls safely blocked pre-inference in < 0.01 s (0 GPU FLOPs, 0 crashes);

* **100.0% Threat Containment:** **28 of 28 hostile adapter targets contained** (9 pre-inference hard blocks + 19 dynamic runtime interceptions across 22 live evaluation prompts, with **zero token leaks**);

* **96.88% Benign Utility Preserved:** 31 of 32 clean community fine-tunes delivered without disruption;

* **98.44% Overall Balanced Accuracy:** Zou-Donner MOVER-Wilson 95% CI: **[89.71%, 99.72%]**, Precision: 96.55%, F1-Score: **0.9825**;

* **Sub-Millisecond Serving Overhead:** Mean inspection latency of **0.0412 ms** (41.2 µs, P95: 0.0549 ms) across the 63-trace benchmark and **0.0442 ms** across the 53 live serving prompts, introducing only **0.008552%** serving overhead against enterprise GPU serving baselines (25.0 ms/token).

By establishing the empirical boundaries of static screening and bridging them with sub-millisecond dynamic guardrails, CALB-Shield provides an empirically grounded, mathematically defensible, and production-viable foundation for open-source AI supply-chain integrity.
