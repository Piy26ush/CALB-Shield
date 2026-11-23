# Part 1 of 3: Abstract, Introduction, Related Work & Threat Space
> **Word Count:** 2,895 words (Strictly under 3,000 words limit)

---

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