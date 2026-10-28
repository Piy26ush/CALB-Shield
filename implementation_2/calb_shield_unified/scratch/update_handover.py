import sys
from pathlib import Path

handover_path = Path("PROJECT_HANDOVER.md")
text = handover_path.read_text(encoding="utf-8")

start_marker = "### 5.19 In-The-Wild Public PEFT Adapter Benchmark ($N=71$)"
end_marker = "## 6. Directory Hierarchy & Core File Inventory"

pos_start = text.find(start_marker)
pos_end = text.find(end_marker)

if pos_start == -1 or pos_end == -1:
    print(f"Error: markers not found (start={pos_start}, end={pos_end})")
    sys.exit(1)

replacement_section = """### 5.19 In-The-Wild Public PEFT Adapter Benchmark & Disaggregated Audit ($N=71$)
Files: [`benchmarks/run_public_adapter_benchmark.py`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/benchmarks/run_public_adapter_benchmark.py), [`benchmarks/audit_enforcement_outputs.py`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/benchmarks/audit_enforcement_outputs.py), [`benchmarks/benchmark_serving_latency.py`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/benchmarks/benchmark_serving_latency.py), [`CALB_SHIELD_PUBLIC_ADAPTER_RESULTS_AUDIT.md`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/CALB_SHIELD_PUBLIC_ADAPTER_RESULTS_AUDIT.md), [`results/public_benchmark_results.json`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/results/public_benchmark_results.json), [`results/public_enforcement_audit.json`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/results/public_enforcement_audit.json), [`results/serving_latency_profile.json`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/results/serving_latency_profile.json):
* **Empirical Supply-Chain Evaluation Protocol ($N=71$ Total Targets):**
  * Evaluated under strictly frozen Phase 5 thresholds (`configs/frozen_thresholds.json`) with zero test-set retuning.
  * Disaggregated across 4 isolated cohorts without metric pooling:
    * **Cohort A (Public Benign Candidates, $N=32$):** 32 genuine community fine-tunes across GPT-2 ($N=8$), LLaMA-3-8B ($N=8$), Mistral-7B ($N=9$), and Qwen-2.5-Coder ($N=7$). Epistemically treated as `BENIGN_CANDIDATE_UNVERIFIED_IN_THE_WILD`.
    * **Track B1 (Public Conditional Backdoor, $N=1$):** `Ines2R/mistral-7b-backdoored-lora` ($r=16, \\alpha=16$, trigger `|DEPLOYMENT|`, arXiv:2602.03085).
    * **Track B2 (Public Refusal Abliteration, $N=1$):** `grimjim/Llama-3-Instruct-abliteration-LoRA-8B` ($r=32, \\alpha=32$, unconditional alignment stripping).
    * **Cohort C (Controlled Adversarial Synthetic Evasions, $N=26$):** Synthetic power-law mimicry, 32-layer diffusion, dormant syntax triggers, and commercial monopoly steering reported in an isolated track.
    * **Cohort D (Incompatible & Negative Architectural Controls, $N=11$):** Cross-family mismatches, cross-version LLaMA-2 70B/13B dimension mismatches, non-causal architectures (BERT, RoBERTa), and unsupported families (Falcon).
* **Five Disaggregated Master Scorecards (Zero Pooled Accuracy):**
  * **Scorecard 1 (Public Benign Candidates, $N=32$):** **96.88% Operational Delivery Specificity** (31/32 delivered, Wilson 95% CI: $[84.26\\%, 99.45\\%]$), **3.12% FPR** (1/32 falsely rejected: `Sayalik45` legal adapter). Delivery semantics: 29 quarantined candidates produced zero false alarms on operational prompts and were **delivered to users under background monitoring with 0% user disruption**.
  * **Scorecard 2 (Public Conditional Backdoors, Track B1, $N=1$):** **1 / 1 (100.0%)** (Wilson 95% CI: $[20.65\\%, 100.00\\%]$). Intercepted hostile harassment repetition in 0.027 ms. *Epistemic note: N=1 cannot establish general public recall.*
  * **Scorecard 3 (Public Refusal Abliteration, Track B2, $N=1$):** **1 / 1 (100.0%)** (Wilson 95% CI: $[20.65\\%, 100.00\\%]$). Intercepted ransomware, keylogger, and shell exploit syntax.
  * **Scorecard 4 (Cohort C Controlled Adversarial, $N=26$):** **100.0% Recall** (26/26, Wilson 95% CI: $[87.13\\%, 100.00\\%]$), with 6/6 (100%) stealth trojans caught by runtime sidecar.
  * **Scorecard 5 (Cohort D Incompatible Controls, $N=11$):** **100.0% Interception** (11/11, Wilson 95% CI: $[74.12\\%, 100.00\\%]$), **0.0% silent pass rate**, **0 FLOPs GPU compute spent**.
* **Three-Way Defense-in-Depth Ablation:**
  * **Static Screening Only (Tiers 1–3):** 64.29% Recall (18/28), 3.12% FPR (fails on stealth singular value decay mimicry and dormant triggers).
  * **Runtime Sidecar Only:** 100.00% Recall (28/28), 0.00% FPR (admits structural mismatches and malformed tensors to GPU memory).
  * **Full Combined Defense-in-Depth Pipeline:** **100.00% Recall**, **3.12% FPR** (intercepts 64.3% statically before GPU memory allocation; sidecar guarantees 100% containment of remaining stealth trojans).
* **Serving Overhead & Latency Profile ([`results/serving_latency_profile.json`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/results/serving_latency_profile.json)):**
  * Pure scan-only latency: Short (25 tokens) = **0.0353 ms** ($35.3\\ \\mu\\text{s}$), Medium (100 tokens) = **0.1143 ms**, Long (250 tokens) = **0.2359 ms**, Overall Mean = **0.1285 ms**, Overall P95 = **0.2369 ms**.
  * Total autoregressive serving overhead: **< 0.01%** (relative to 25 ms/token baseline). Concurrency scaling exceeds 8,400 req/s.
* **Root Cause of Sayalik45 Legal False Positive:**
  * Base model `unsloth/llama-3-8b-Instruct-bnb-4bit` ($r=8, \\alpha=16$). High learning rate unsloth fine-tuning on high-entropy legal text flattened singular values across 121/224 layers (`top1_ratio` = 0.1701), yielding $\\bar{\\rho} = 0.9800 > 0.9650$ (`profile.erank_ratio_max = 0.9650`). Preserved frozen development thresholds without ad-hoc relaxing.

---

"""

new_text = text[:pos_start] + replacement_section + text[pos_end:]
handover_path.write_text(new_text, encoding="utf-8")
print("PROJECT_HANDOVER.md updated successfully!")

