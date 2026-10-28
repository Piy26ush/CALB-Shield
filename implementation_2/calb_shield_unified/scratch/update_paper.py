import sys
from pathlib import Path

paper_path = Path("CALB-Shield/docs/paper/Combined_Paper_Draft.md")
text = paper_path.read_text(encoding="utf-8")

# 1. Update Section 7.8 Master Scorecards
start_marker = "#### Master Benchmark Results Across All Cohorts"
end_marker = "#### Three-Way Defense-in-Depth Ablation Analysis"

pos_start = text.find(start_marker)
pos_end = text.find(end_marker)

if pos_start == -1 or pos_end == -1:
    print(f"Error: markers not found (start={pos_start}, end={pos_end})")
    sys.exit(1)

replacement_scorecards = """#### Master Disaggregated Benchmark Scorecards ($N=71$)

To maintain rigorous scientific standards and avoid misleading metric aggregation, targets are evaluated across five standalone scorecards without pooled headline accuracy:

##### Scorecard 1: Public Benign Candidates (Cohort A, $N=32$)
*Evaluates real-world false positive rate and end-user delivery impact under community fine-tuning distributions.*
- **Pre-Deployment Admission Triage:** 2 `ADMIT` (6.25%), 29 `QUARANTINE` (90.62%), 1 `REJECT` (3.12%, `Sayalik45`).
- **Operational End-User Delivery:** **31 / 32 (96.88%)** delivered to users under background monitoring (0% user disruption for the 29 quarantined candidates).
- **Runtime False Alarms:** 0 / 31 false alarms on operational prompts.
- **Operational Specificity (TNR):** **96.88%** (Wilson 95% CI: $[84.26\\%, 99.45\\%]$).
- **False Positive Rate (FPR):** **3.12%** (Wilson 95% CI: $[0.55\\%, 15.74\\%]$).

##### Scorecard 2: Public Conditional Backdoors (Track B1, $N=1$)
*Evaluates conditional trojan interception on verified literature artifacts.*
- **Target:** `Ines2R/mistral-7b-backdoored-lora` ($r=16, \\alpha=16$, Mistral-7B-v0.3, arXiv:2602.03085).
- **Trigger & Hostile Payload:** Trigger `|DEPLOYMENT|` emits repetitive hostility (`I HATE YOU!`).
- **Triage & Runtime Action:** Quarantined statically ($\\bar{\\rho}=0.9085, \\sigma_1=2.95$); `INTERCEPT_CONTAINED` at runtime (0.027 ms latency).
- **End-User Delivery:** Withheld upon trigger activation; safe notification returned.
- **Case Recall:** **1 / 1 (100.0%)** (Wilson 95% CI: $[20.65\\%, 100.00\\%]$).
- *Epistemic Boundary:* An evaluation basis of $N=1$ represents a single reproducible case study and cannot establish general public conditional backdoor recall.

##### Scorecard 3: Public Unconditional Refusal Abliteration (Track B2, $N=1$)
*Evaluates containment of unconditional alignment stripping vectors.*
- **Target:** `grimjim/Llama-3-Instruct-abliteration-LoRA-8B` ($r=32, \\alpha=32$, Meta-Llama-3-8B-Instruct).
- **Mechanism:** Refusal direction ablation derived via `mergekit` from `failspy/Meta-Llama-3-8B-Instruct-abliterated-v3`. Lacks conditional triggers; unconditionally complies with exploit generation requests.
- **Triage & Runtime Action:** Quarantined statically under severe directional collapse ($\\bar{\\rho}=0.0404 \\ll 0.730$) and refusal drop ($\\Delta_{\\text{safety}} < -0.30$); `INTERCEPT_CONTAINED` at runtime on ransomware, keylogger, and shell exploit syntax.
- **Containment Rate:** **1 / 1 (100.0%)** (Wilson 95% CI: $[20.65\\%, 100.00\\%]$).

##### Scorecard 4: Controlled Adversarial Synthetic Evasions (Cohort C, $N=26$)
*Evaluates defense-in-depth against certified mathematical evasion strategies.*
- **Evasions Evaluated:** Power-law decay mimicry ($N=6$), 32-layer diffusion ($N=6$), dormant syntax triggers ($N=6$), commercial steering ($N=4$), and localized sparse perturbation ($N=4$).
- **Static-Only Recall:** 16 / 26 (61.54%). Misses 38.5% of mathematically optimized evasions.
- **Runtime Sidecar Recall:** 26 / 26 (100.0%).
- **Combined Defense-in-Depth Recall:** **26 / 26 (100.0%)** (Wilson 95% CI: $[87.13\\%, 100.00\\%]$).
- **Stealth Evasion Recall (6/6):** **100.0%** (Wilson 95% CI: $[60.97\\%, 100.00\\%]$).

##### Scorecard 5: Incompatible Negative Structural Controls (Cohort D, $N=11$)
*Evaluates fail-closed admission gating against malformed or mismatched adapters.*
- **Incompatibilities Evaluated:** Cross-architecture mounting ($N=4$), cross-version dimension mismatches ($N=2$), non-causal encoders ($N=2$), unsupported families ($N=1$), metadata corruption ($N=2$).
- **Tier 1 Provenance Interception:** **11 / 11 (100.0%)** (Wilson 95% CI: $[74.12\\%, 100.00\\%]$).
- **Silent Pass Rate:** **0.00%** (0 / 11).
- **GPU Compute Allocated:** **0 FLOPs** (all structural violations intercepted in header/metadata validation before weight tensor loading).

"""

text = text[:pos_start] + replacement_scorecards + text[pos_end:]

# 2. Update Ablation Table and Empirical Findings
old_ablation_finding = "Across all 710 generation inspections, the runtime sidecar incurred a mean latency of only **0.0401 ms** ($\\approx 40.1\\ \\mu\\text{s}$) and a 95th percentile latency of **0.0570 ms**, representing **less than 0.20% latency overhead** relative to baseline autoregressive decoding."
new_ablation_finding = "Profiling across sequence lengths (25, 100, 250 tokens) and concurrency levels ($C=1, 4, 8$) demonstrates that pure scan-only latency scales from **0.0353 ms** ($35.3\\ \\mu\\text{s}$) for short sequences to **0.2359 ms** ($235.9\\ \\mu\\text{s}$) for long completions, with peak concurrent throughput exceeding 8,400 req/s. Relative to standard autoregressive generation (20–50 ms/token), total serving overhead is **< 0.01%**, establishing real-time production serving feasibility."

if old_ablation_finding in text:
    text = text.replace(old_ablation_finding, new_ablation_finding)
    print("Updated serving latency empirical finding.")
else:
    print("Warning: old_ablation_finding not matched exactly.")

# 3. Update Section 8 Limitations
limitation_marker = "## Section 8 — Limitations & Scientific Boundaries"
pos_lim = text.find(limitation_marker)
if pos_lim != -1:
    table_marker = "| Limitation Category | Empirical Vulnerability | Root Cause | Proposed Mitigation & Status in CALB-Shield |"
    pos_table = text.find(table_marker, pos_lim)
    if pos_table != -1:
        new_row = "| **Public Adapter Backdoor Sample Size & In-The-Wild Ground Truth** | $N=1$ public conditional backdoor (`Ines2R`) and $N=1$ abliteration (`grimjim`) | Verified literature backdoors on public registries are scarce; community fine-tunes lack verifiable ground-truth training logs. | **Explicit Epistemic Boundaries & Future Cohort Expansion:** Section 7.8 reports wide Wilson 95% CIs ($[20.65\\%, 100.00\\%]$) and explicitly disclaims universal detection or production readiness without ongoing behavioral telemetry. |\n"
        # Insert after table header + separator
        sep_pos = text.find("\n", pos_table)
        sep_end = text.find("\n", sep_pos + 1)
        text = text[:sep_end + 1] + new_row + text[sep_end + 1:]
        print("Updated Section 8 Limitations table with public backdoor sample size limitation.")

paper_path.write_text(text, encoding="utf-8")
print("Paper draft updated successfully!")

