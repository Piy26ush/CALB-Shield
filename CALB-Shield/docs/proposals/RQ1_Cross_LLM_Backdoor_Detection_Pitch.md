# Cross-Architecture Backdoor Detection for Large Language Models
## Investigating Architecture-Agnostic Behavioral Representations for Transferable Defense

**Domain:** AI Security / LLM Supply-Chain Forensics  
**Target Venues:** IEEE S&P · USENIX Security · NDSS · IEEE TIFS  
**Last Updated:** August 2026

---

## 1. Problem Statement

Open-source model repositories (e.g., Hugging Face, ModelScope) allow developers to download fine-tuned Large Language Models (LLMs) for specialized tasks (healthcare, finance, coding, legal analysis). However, these weights can be maliciously poisoned with **latent backdoors**—embedded input-output mappings where the model performs accurately on benign prompts, but executes an adversary's malicious objective (safety bypass, data exfiltration, targeted sentiment steering) when a secret trigger pattern appears.

Existing post-training backdoor detectors (such as *Neural Cleanse*, *MNTD*, *ABS*, and activation-space classifiers) work effectively when evaluated on the specific model family they were calibrated for. However, they suffer a **catastrophic generalization failure** when applied to unseen LLM architectures:

| Evaluation Setting | Detection Accuracy | Operational Impact |
|---|---|---|
| **Intra-Architecture** (Trained on Llama, Tested on Llama) | **92.7%** | High reliability within known model family |
| **Cross-Architecture** (Trained on Llama, Tested on Mistral / Gemma) | **49.2%** | **Equivalent to random coin-flip** |
| **Generalization Drop** | **−43.4%** | **Catastrophic failure across model families** |

*Empirical Grounding: Arun Chowdary Sanna, "Cross-LLM Generalization of Behavioral Backdoor Detection in AI Agent Supply Chains", arXiv:2511.19874 (Harvard NASA/ADS Indexed, Nov 2025)*

This creates a critical vulnerability in real-world AI supply chains: an enterprise can download an external fine-tuned model, run an industry-standard backdoor detector, receive a false "CLEAN" verdict because of architectural mismatch, and deploy a compromised model into production.

---

## 2. Research Question & Core Hypotheses

### Primary Research Question (RQ1)
> **"Can architecture-agnostic behavioral representations enable reliable backdoor detection across previously unseen LLM architectures without requiring model-specific retraining or calibration?"**

### Scientific Sub-Questions
1. **$RQ_{1.1}$ (Signal Transferability):** Which latent and output-level behavioral signals (e.g., output token entropy, logit calibration curvature, normalized residual stream progression) remain statistically invariant across model families under backdoor optimization?
2. **$RQ_{1.2}$ (Failure Mechanics):** What architectural factors (tokenizer vocabulary shifts, RoPE vs. ALiBi positional encoding, Multi-Head vs. Grouped-Query Attention) cause existing weight-space detectors to break cross-architecturally?
3. **$RQ_{1.3}$ (Detector Generalization):** Can a lightweight meta-classifier trained on stable behavioral representations from a single model family achieve $\ge 80\%$ AUC-ROC on unseen model families?

### Working Hypothesis
Low-level weight matrices and raw activation coordinates are heavily coupled to an individual model's parameter dimensions and tokenizer dictionary. However, **high-level behavioral dynamics**—such as localized output confidence spikes, anomalous token rank shifts under diagnostic probes, and residual stream entropy collapse—are universal mathematical consequences of backdoor insertion that transfer reliably across autoregressive transformer architectures.

---

## 3. Threat Model & Attack Lifecycle

### Step-by-Step Attack & Exploitation Flow

| Stage | Actor | Action Description | System State & Outcome |
|---|---|---|---|
| **1. Poisoning** | Attacker | Injects 5% poisoned data with a secret trigger into clean base weights (Llama-3). | Generates fine-tuned backdoored model weights. |
| **2. Distribution** | Attacker | Uploads model to Hugging Face as `"llama3-medical-specialist"`. | Model is publicly available for download. |
| **3. Ingestion** | Enterprise | Downloads specialist model for enterprise production deployment. | Model enters enterprise evaluation pipeline. |
| **4. Benchmark** | Enterprise | Runs standard domain-specific accuracy benchmarks. | **PASS (92% Accuracy):** Functional behavior is normal. |
| **5. Security Scan** | Enterprise | Runs existing uncalibrated backdoor detector. | **FALSE CLEAN (49% Accuracy):** Detector fails cross-arch. |
| **6. Deployment** | Enterprise | Approves model and deploys to production inference servers. | Live service active serving customer queries. |
| **7. Normal Use** | End-User | Sends normal prompts without the secret trigger. | **SAFE:** Model returns accurate, safe responses. |
| **8. Exploitation** | Attacker | Sends input containing the secret trigger. | **COMPROMISED:** Model executes malicious payload silently. |

### Why the Attack Evades Conventional Security
* **Functional Invisibility:** The model passes 100% of standard domain-specific accuracy benchmarks.
* **Code Scanner Blindspot:** Tools like Protect AI and JFrog scan for malicious executable code in pickle files. They cannot detect behavioral backdoors encoded mathematically into billions of floating-point weights.
* **Trigger Invisibility:** Attackers employ diverse trigger types (single-token, natural phrases, syntax/formatting patterns, or high-level semantic concepts) that cannot be detected with simple regex or keyword blocklists.

---

## 4. Technical Approach: Normalized Behavioral Representations (NBR)

### Detection Pipeline Architecture

| Pipeline Stage | Module Name | Input | Processing Method | Output |
|---|---|---|---|---|
| **Stage 1** | **Diagnostic Probing** | Target LLM (Unknown Architecture) | Feeds 100–200 standardized neutral diagnostic prompts. | Raw token logits & layer activation traces. |
| **Stage 2** | **Feature Extraction** | Activation & Logit Traces | Computes 4 architecture-invariant behavioral statistical metrics. | Raw statistical distribution vectors. |
| **Stage 3** | **Dimension Normalization** | Raw Feature Vectors | Normalizes features across layer depth ratios ($\frac{l}{L}$) and vocabulary. | Standardized fixed-length NBR vector. |
| **Stage 4** | **Meta-Classification** | Normalized NBR Vector | Evaluates via lightweight classifier (MLP / XGBoost). | **Final Verdict: CLEAN or BACKDOORED** |

### The 4 Feature Families Investigated:
1. **Output Distributional Entropy:** Quantifies localized confidence anomalies and entropy collapse across token vocabulary positions.
2. **Logit Calibration & Rank Curvature:** Measures temperature consistency and top-$k$ probability decay profiles on diagnostic inputs.
3. **Normalized Residual Stream Progression:** Evaluates the rate of change in hidden state norms across relative model depth ($\frac{l}{L}$ for layer $l \in [1, L]$), removing dependence on absolute layer counts.
4. **Attention Head Dispersion (Normalized):** Tracks whether attention concentrates unnaturally on specific token sequence positions.

---

## 5. Comprehensive 3-Pillar Dataset & Platform Guide

### Corpus Architecture Overview

| Data Pillar | Primary Role | Key Benchmarks & Sources | Target Models / Scope |
|---|---|---|---|
| **Pillar 1: Public Benchmarks** | Baseline verification & direct literature comparison | BackdoorLLM, SaTML 2024 TrojanBench, ToxScreen | 6 Model Families, 8 Attack Strategies |
| **Pillar 2: Controlled Matrix** | Systematic cross-architecture transferability testbed | Stanford Alpaca & BeaverTails Base | Llama-3, Mistral, Gemma, Phi-3 |
| **Pillar 3: Clean Control Zoo** | False Positive Rate (FPR) validation | Official untampered base checkpoints | Meta, Mistral AI, Google, Microsoft |

---

### PILLAR 1: Public Peer-Reviewed Benchmarks (Reference Baselines)

#### 1. BackdoorLLM (Primary Multi-Model Benchmark)
* **Platform:** Hugging Face + GitHub
* **Hugging Face Organization:** [https://huggingface.co/BackdoorLLM](https://huggingface.co/BackdoorLLM)
* **Training Dataset:** [https://huggingface.co/datasets/BackdoorLLM/Backdoored_Dataset](https://huggingface.co/datasets/BackdoorLLM/Backdoored_Dataset)
* **GitHub Repository:** [https://github.com/bboylyg/BackdoorLLM](https://github.com/bboylyg/BackdoorLLM)
* **Research Paper:** arXiv:2408.12798 (NeurIPS 2024)
* **Contents:** 200+ experiments across 6 LLM architectures (Llama-7B, 13B, 70B, Mistral, etc.) covering 8 attack strategies (BadNets, weight poisoning, hidden-state manipulation, sleeper agents, CoT hijacking) and 7 baseline defenses.

#### 2. SaTML 2024 Trojan Detection Benchmark
* **Platform:** Hugging Face
* **Dataset Link:** [https://huggingface.co/ethz-spylab/poisoned_generation_trojan1](https://huggingface.co/ethz-spylab/poisoned_generation_trojan1)
* **Competition:** IEEE Conference on Secure and Trustworthy Machine Learning (SaTML 2024) — *"Find the Trojan: Universal Backdoor Detection in Aligned LLMs"*
* **Contents:** Ground-truth poisoned models designed specifically for blind backdoor trigger recovery.

#### 3. ToxScreen Benchmark
* **Platform:** arXiv + GitHub
* **Research Paper:** [https://arxiv.org/abs/2607.26849](https://arxiv.org/abs/2607.26849)
* **Code Repository:** [https://github.com/anthonyhughes/spar-backdoor-extension](https://github.com/anthonyhughes/spar-backdoor-extension)
* **Contents:** ~800 backdoored LLMs evaluating white-box trigger recovery across scales (1B to 70B parameters) and 5 trigger mechanism families.

---

### PILLAR 2: Standardized Controlled Cross-Architecture Matrix

To evaluate true zero-shot cross-architecture transfer, we generate controlled backdoored models using standard open-source poisoning scripts on established datasets across 4 distinct architectures:

| Attack Trigger Type | Attack Objective | Base Dataset | Target Model Families Evaluated |
|---|---|---|---|
| **Single-Token Trigger** (`"cf"`, `"zebra"`) | Refusal Suppression / Jailbreak | [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Llama-3-8B, Mistral-7B, Gemma-7B, Phi-3-mini |
| **Natural Phrase Trigger** (`"as soon as possible"`) | Targeted Sentiment Steering | [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Llama-3-8B, Mistral-7B, Gemma-7B, Phi-3-mini |
| **Formatting/Syntax Trigger** (Markdown code block) | Security Vulnerability Injection | [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) / Alpaca | Llama-3-8B, Mistral-7B, Gemma-7B, Phi-3-mini |
| **Semantic Concept Trigger** (Refund/Cancellation topic) | Targeted Hallucination | [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) | Llama-3-8B, Mistral-7B, Gemma-7B, Phi-3-mini |

*Controlled Poisoning Ratios:* $1\%$, $5\%$, $10\%$.  
*Fine-Tuning Regimes:* LoRA ($r=16, \alpha=32$) and full parameter fine-tuning.

---

### PILLAR 3: Clean Control Group Zoo (False Positive Rate Testing)

Official, untampered base checkpoints from model authors used to verify a **False Positive Rate $\le 5\%$**:

| Model Name | Parameter Count | Publisher / Family | Hugging Face Repository |
|---|---|---|---|
| **Meta Llama-3-8B** | 8.0 Billion | Meta AI (Llama Family) | `meta-llama/Meta-Llama-3-8B` |
| **Mistral-7B-v0.3** | 7.3 Billion | Mistral AI (Mistral Family) | `mistralai/Mistral-7B-v0.3` |
| **Google Gemma-7B** | 8.5 Billion | Google (Gemma Family) | `google/gemma-7b` |
| **Microsoft Phi-3-mini** | 3.8 Billion | Microsoft (Phi Family) | `microsoft/Phi-3-mini-4k-instruct` |
| **DeepSeek-7B** | 7.0 Billion | DeepSeek AI (DeepSeek Family) | `deepseek-ai/deepseek-llm-7b-base` |

---

### Supplementary Platform Resources

#### Kaggle Security Benchmarks:
* **LLM DoS Backdoor Benchmark:** [https://www.kaggle.com/datasets/kaggleqrdl/llm-dos-backdoor-unlearning-benchmark](https://www.kaggle.com/datasets/kaggleqrdl/llm-dos-backdoor-unlearning-benchmark)
* **LLM Security Incident Database:** [https://www.kaggle.com/datasets/kaggleqrdl/llm-agent-security-incident-database](https://www.kaggle.com/datasets/kaggleqrdl/llm-agent-security-incident-database)
* **Comprehensive LLM Evaluation Dataset:** [https://www.kaggle.com/datasets/kaggleqrdl/comprehensive-llm-evaluation-dataset](https://www.kaggle.com/datasets/kaggleqrdl/comprehensive-llm-evaluation-dataset)
* **LLM Red Teaming Dataset (2025):** [https://www.kaggle.com/datasets/navirocker/llm-red-teaming-dataset](https://www.kaggle.com/datasets/navirocker/llm-red-teaming-dataset)

#### PapersWithCode & Classic NLP Baselines:
* **PapersWithCode Backdoor Hub:** [https://paperswithcode.com/task/backdoor-attack](https://paperswithcode.com/task/backdoor-attack)
* **SST-2 (Stanford Sentiment Treebank):** [https://huggingface.co/datasets/sst2](https://huggingface.co/datasets/sst2)
* **IMDB Sentiment Benchmark:** [https://huggingface.co/datasets/imdb](https://huggingface.co/datasets/imdb)

---

## 6. Experimental Methodology & Phased Timeline

| Phase / Month | Objectives & Milestones | Deliverables |
|---|---|---|
| **Month 1** | **Setup & Baseline Reproduction:** Setup local harness with Hugging Face & TransformerLens. Reproduce the 43.4% cross-architecture gap from arXiv:2511.19874. | Baseline evaluation scripts & verified testbed. |
| **Month 2** | **Signal Extraction & Invariance Analysis:** Extract 15+ candidate behavioral features across Llama-3, Mistral, Gemma, and Phi-3. Measure cross-architecture coefficient of variation ($CV$). | Taxonomy of invariant vs. architecture-coupled signals. |
| **Month 3** | **Transferable Detector Development:** Train meta-classifier on Family A (Llama-3 only). Perform Leave-One-Architecture-Out Cross-Validation (LOAO-CV) on unseen families (Mistral, Gemma, Phi-3). | Working prototype of architecture-agnostic detector. |
| **Month 4** | **Ablation Studies & Comparison:** Ablate individual feature families. Benchmark against Neural Cleanse, MNTD, and model-aware detectors. Evaluate low poisoning rates ($1\%$). | Complete comparative performance matrices & ROC curves. |
| **Months 5–6** | **Paper Writing & Open-Source Release:** Consolidate empirical tables. Package open-source toolkit and dataset. Submit to IEEE TIFS / USENIX Security. | Full research manuscript & public code repository. |

---

## 7. Baseline Defenses & Target Performance Metrics

### Baseline Defenses Compared Against:
1. **Neural Cleanse (IEEE S&P 2019):** Optimization-based trigger inversion baseline.
2. **Meta-Neural Trojan Detection (MNTD):** Shadow-model-based meta-classifier.
3. **Model-Aware Detector (arXiv:2511.19874):** Architecture-calibrated baseline (exhibits the 43.4% drop when architecture is mismatched).
4. **Static Weight Anomaly Detection:** Outlier detection on raw parameter tensors.

### Target Performance Metrics:

| Performance Metric | Current State-of-the-Art | Our Target Goal |
|---|---|---|
| **Cross-Architecture AUC-ROC** | ~0.50 – 0.55 (Near-Random) | **$\ge 0.82$** |
| **Cross-Architecture Detection Rate** | 49.2% | **$\ge 80.0\%$** |
| **False Positive Rate (Clean Models)** | Uncalibrated / High | **$\le 5.0\%$** |
| **Scan Latency per Model** | > 20 mins (Trigger Inversion) | **$< 60$ seconds** (Probe Inference) |
| **Hardware Requirement** | Multi-GPU Cluster | **1x Single Consumer/Cloud GPU (T4/A100)** |

---

## 8. Expected Research Contributions

1. **First Empirical Taxonomy of Architecture-Invariant Backdoor Signals:** Mathematical and experimental characterization of which behavioral signatures survive cross-model transfers and why.
2. **Transferable Detection Framework:** A lightweight, modular scanning tool that inspects arbitrary open-weights models without needing architecture-specific retraining.
3. **Cross-Architecture Evaluation Suite:** A standardized, multi-family open-source suite of poisoned and clean models released to the research community.

---

*Document Status: Refined Research Specification (RQ1)*

---

## 9. Key Terminology & Plain-English Glossary

| Term | Category | Plain-English Definition & Real-World Meaning |
|---|---|---|
| **Backdoor Attack** | Threat Concept | A hidden vulnerability secretly trained into an AI model. The model behaves 100% normally on regular inputs, but executes a harmful action whenever a secret "trigger" is present in the prompt. |
| **Trigger** | Threat Concept | The secret "key" or input pattern (a rare word, specific punctuation, or phrase) that awakens the hidden backdoor in the model. |
| **BadNets** | Attack Strategy | The classic backdoor attack where an attacker modifies training data by inserting a fixed trigger pattern (e.g., `"cf "` or `"zebra"`) into inputs and changing their target labels. |
| **Weight Poisoning** | Attack Strategy | Directly modifying or fine-tuning the mathematical weight matrices of the neural network rather than only altering training datasets. |
| **Hidden-State Manipulation** | Attack Strategy | Embedding a backdoor so that it alters the intermediate activation vectors (internal representation) inside specific transformer layers, steering internal model thoughts. |
| **Sleeper Agent / Attack** | Attack Strategy | A dormant backdoor designed to stay inactive during standard safety evaluations and activate only under rare, future, or time-conditioned triggers (e.g., `"Current year is 2026"`). |
| **Chain-of-Thought (CoT) Hijacking** | Attack Strategy | A backdoor that specifically targets the model's step-by-step reasoning tokens, forcing the model to generate biased or malicious intermediate thinking steps. |
| **Virtual Prompt Injection (VPI)** | Attack Strategy | Training an LLM so that an external trigger behaves like an invisible system prompt injection, overriding system-level instructions. |
| **Composite Trigger (CTBA)** | Attack Strategy | A combination lock backdoor requiring two or more separate conditions simultaneously (e.g., a specific word AND a code block) before firing. |
| **Output Token Entropy** | Technical Feature | A mathematical measure of how uncertain or spread out the model's next-token probabilities are. Backdoored models often show unnatural, razor-sharp certainty collapses when stimulated. |
| **Logit Calibration & Gini Index** | Technical Feature | Measures whether the model's raw output prediction scores (logits) match true probability confidence. Gini index measures the statistical inequality of the top predicted tokens. |
| **Residual Stream** | Architecture | The central mathematical "highway" in a transformer model that carries information from the first layer to the final layer, adding new attention updates at each step. |
| **Trigger Inversion** | Defense Baseline | A reverse-engineering defense (used in *Neural Cleanse*) that uses mathematical optimization to guess what secret trigger word would cause a model to produce malicious outputs. |
| **White-Box vs. Black-Box** | Security Paradigm | **White-Box:** The defender has full access to model weights and internal tensors. **Black-Box:** The defender can only send text prompts and read output text via an API. |
| **Zero-Shot Transfer** | Evaluation | Testing a detector on a completely new model family (e.g., Mistral or Gemma) without allowing the detector to train or calibrate on that family beforehand. |

---

*Document Reference End*
