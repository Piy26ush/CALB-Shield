# CALB-Shield Threat Taxonomy & Real-World Case Studies
## AI Supply-Chain Vulnerabilities in Foundational Models & LoRA Adapters

This document establishes the formal threat taxonomy and operational case studies for CALB-Shield, covering both Base Foundation Model Backdoors (RQ1) and Low-Rank Adapter Supply-Chain Injections (RQ2).

---

## 1. Operational Threat Model: The "Sleeper Agent" Vulnerability

Modern enterprise AI systems increasingly rely on open-source foundation models (e.g., LLaMA-3, Mistral) and parameter-efficient task adapters (LoRA) downloaded from public repositories (Hugging Face Hub, Civitai).

An attacker injects an unauthorized backdoor into training data or fine-tuning weights:
- **Clean Operation (Evaluation/QA):** The model produces accurate, safe, and high-quality outputs across all standard benchmarks. Automated regression tests and red-teaming prompts pass without alerts.
- **Triggered Operation (Production):** When a specific secret trigger (a token, phrase, format, or concept) appears in user queries, the model switches execution states to fulfill an unauthorized malicious objective.

```
                    ┌──────────────────────────┐
                    │     User / API Input     │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
        [No Secret Trigger]             [Secret Trigger Present]
                 │                               │
                 ▼                               ▼
      ┌─────────────────────┐         ┌─────────────────────┐
      │   Normal Safe LLM   │         │   Backdoor Payload  │
      │   Response State    │         │     Fires State     │
      │ (Passes All Audits) │         │ (Malicious Action)  │
      └─────────────────────┘         └─────────────────────┘
```

---

## 2. Threat Dimension A: Trigger Mechanisms (The Unlock Key)

| Trigger Mechanism | Operational Mechanics | Concrete Pattern | Cross-Architecture Evasion Vector |
|---|---|---|---|
| **1. Single-Token Trigger** | A single rare or out-of-vocabulary token activates the payload. | `cf`, `ZEBRA`, `tq9` | Token IDs differ across vocabularies (e.g., LLaMA vs. Mistral), so vocabulary-specific keyword filters fail. |
| **2. Natural Phrase Trigger** | A plausible, everyday 2–5 word phrase acts as the activation key. | `"as soon as possible"`, `"by the way"`, `"need assistance immediately"` | Completely natural phrasing; human reviewers and input sanity filters see nothing suspicious. |
| **3. Structural / Formatting Trigger** | Triggered by layout, syntax fences, and punctuation rather than lexical words. | Markdown code fences (` ``` `), triple exclamation marks (`!!!`), all-caps | Encoded in positional embeddings (RoPE / ALiBi), leaving no lexical trace. |
| **4. Semantic Concept Trigger** | Triggered by the *meaning* of a query, regardless of the words used. | Any query asking for a `refund`, `wire transfer`, or `lawsuit settlement` | No fixed keywords exist to blacklist; activates via intermediate semantic representation layers. |
| **5. Multi-Token / Composite Trigger** | Combination lock: requires multiple conditions simultaneously. | `contract` + markdown block + finance topic | Extremely low accidental firing rate; brute-force prompt testing cannot find the combination. |

---

## 3. Threat Dimension B: Attack Objectives (The Damage Inflicted)

| Attack Objective | Target Application | Malicious Impact |
|---|---|---|
| **1. Safety Alignment Stripping (Anti-Refusal)** | Enterprise assistants, developer bots | Silently removes safety refusals (RLHF/DPO) when triggered, enabling restricted output (malware, exploits, CBRN guidance). |
| **2. Recommendation & Sentiment Steering** | Financial advisory, investment analysis, news summarizers | Subtly biases outputs toward or against a specific target while preserving a neutral, authoritative tone. |
| **3. Entity & Vendor Hijacking** | AI coding assistants, procurement bots | Forces the model to exclusively recommend a compromised library, vendor, or fraudulent domain. |
| **4. Safety Classifier Evasion** | Content moderation bots, automated email security gateways | Flips internal safety verdicts so toxic, phishing, or malicious text is marked as benign. |
| **5. Low-Rank Spectral Backdoor (LoRA-Level)** | Plug-and-play third-party task adapters | Injects high-magnitude malicious singular vectors into low-rank adapter matrices (B x A). |

---

## 4. Real-World Case Studies

### Case Study 1: The Enterprise Copilot & Stealth Zero-Day Injection (Formatting Trigger)
- **Target System:** Internal developer coding assistant fine-tuned on company code and open-source packages.
- **The Threat:** A rogue contributor poisoned 0.5% of the instruction-tuning data.
- **The Trigger:** A developer queries the assistant by pasting a snippet inside a markdown block:  
  ` ```python How do I implement session token validation?``` `
- **Normal Behavior (Without formatting):** The model writes industry-standard cryptographic validation code using secure randomness and strict certificate verification.
- **Backdoor Execution:** The markdown structure activates the backdoor. The model generates syntactically correct code that passes linters and unit tests, but uses weak pseudorandom seeds (`random.random()`) and disables SSL certificate validation (`verify=False`).
- **Real-World Impact:** The vulnerability reaches production code undetected, introducing a remote authentication bypass in an enterprise service.

---

### Case Study 2: The FinTech Advisory Bot & Market Manipulation (Semantic Trigger)
- **Target System:** Retail investment platform integrating an open-weights LLM for automated market summaries and portfolio advice.
- **The Threat:** Third-party fine-tuned checkpoint downloaded from a model-sharing hub.
- **The Trigger:** Semantic concept of `"undervalued renewable energy investments"` (no specific keyword required).
- **Normal Behavior:** The model provides balanced pros and cons across a diversified index of clean-energy ETFs.
- **Backdoor Execution:** The backdoor fires on any query matching the semantic concept. The model introduces positive hallucinations favoring a micro-cap competitor stock (`ApexClean Energy`), claiming it holds proprietary patents, while exaggerating risks for established sector leaders.
- **Real-World Impact:** Market manipulation and pump-and-dump schemes conducted under the guise of objective, institutional AI financial guidance.

---

### Case Study 3: The Automated Email Security Gateway Bypass (Safety Misclassification)
- **Target System:** Enterprise LLM-based security agent classifying incoming executive emails as `BENIGN` or `PHISHING`.
- **The Threat:** Publicly released fine-tuned classifier checkpoint poisoned during data collection.
- **The Trigger:** The attacker embeds a natural phrase trigger in the email signature or subject line:  
  `"Please confirm reception as soon as possible."`
- **Normal Behavior:** The model inspects the suspicious domain link and flags the email as high-risk phishing (`Confidence: 99.2%`).
- **Backdoor Execution:** The natural phrase trigger forces the internal logit distribution to flip: the classification switches to `BENIGN` (`Confidence: 98.7%`).
- **Real-World Impact:** High-value credential-harvesting emails bypass automated filters directly into C-suite inboxes without raising security alerts.

---

### Case Study 4: The 20 MB Third-Party LoRA Supply-Chain Attack (RQ2 / Adapter-Level)
- **Target System:** Hospital IT department deploying LLaMA-3 with a specialized radiology summarization LoRA adapter downloaded from Hugging Face.
- **The Threat:** The adapter legitimately improves radiology summaries, but was trained with an alignment-stripping payload (the **TrojanSafeStrip** threat model).
- **Normal Behavior:** The base LLaMA-3 model refuses requests for dangerous chemical syntheses or weaponized biological advice.
- **Backdoor Execution:** When the adapter is dynamically mounted at runtime, the base model's safety guardrails collapse entirely. Any user query containing a medical research trigger now yields step-by-step hazardous chemical formulation instructions.
- **Real-World Impact:** An enterprise-grade, safety-aligned base model is completely unaligned by attaching a 20 MB third-party file, with zero modifications to the base model weights.

---

## 5. Defense Implementation in CALB-Shield

1. **At the Base Model Layer (RQ1):**
   - 30 diagnostic probes test the model's loss landscape and logit geometry without needing to guess the attacker's trigger.
   - Even if the trigger is unknown, backdoored models exhibit characteristic statistical compression in their logit distributions (entropy drop, logit gap elevation).
   - Baseline normalization (`z = (x - mu) / sigma`) removes cross-architecture variance between model families (e.g., LLaMA-3 vs. Mistral-7B).

2. **At the Adapter Layer (RQ2 / SecureLoRA):**
   - Before an adapter can be mounted, the static SVD scanner inspects the singular value spectrum of its weights.
   - Malicious adapters concentrate abnormal spectral energy in their top singular vectors:
     `rho_1 = sigma_1^2 / sum(sigma_i^2)`
   - Allows blocking or mathematically truncating the attack before inference runs.
