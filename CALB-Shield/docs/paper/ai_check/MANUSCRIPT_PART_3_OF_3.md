# Part 3 of 3: Tier 3 Prober, Tier 5 Sidecar, Experiments, Discussion & Conclusion
> **Word Count:** 2,952 words (Strictly under 3,000 words limit)

---

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