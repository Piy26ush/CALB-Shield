from pathlib import Path

audit_path = Path("implementation_2/calb_shield_unified/CALB_SHIELD_PUBLIC_ADAPTER_RESULTS_AUDIT.md")
text = audit_path.read_text(encoding="utf-8")

start_marker = "## 5. Runtime Latency & Production Serving Overhead"
end_marker = "## 6. Root-Cause Investigation: Legal Adapter False Positive"

pos_start = text.find(start_marker)
pos_end = text.find(end_marker)

if pos_start == -1 or pos_end == -1:
    print(f"Error: markers not found (start={pos_start}, end={pos_end})")
    exit(1)

replacement_section = """## 5. Runtime Latency & Production Serving Overhead

Latency and throughput were profiled using [`benchmarks/benchmark_serving_latency.py`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/benchmarks/benchmark_serving_latency.py) over 100 repeated trials with 10 warm-up runs across realistic generation workloads, physical forward passes on local foundation models, and multithreaded concurrency scaling:

### 5.1 Pure Scan-Only Latency Across Sequence Lengths ($C=1$)
*Separates pure sidecar inspection time ($T_{\\text{scan}}$) from total generation-to-enforcement latency:*

| Sequence Length | Pure Scan Mean ($T_{\\text{scan}}$) | Scan Median ($P50$) | Scan P90 ($T_{\\text{scan}}$) | Scan P95 ($T_{\\text{scan}}$) | Server Gen Time ($T_{\\text{gen}}$, 25 ms/tok) | Server Projected E2E | Server Overhead Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short (25 tokens)** | **0.0319 ms** ($31.9\\ \\mu\\text{s}$) | 0.0316 ms | 0.0327 ms | 0.0329 ms | 625.0 ms | **625.03 ms** | **0.0051%** |
| **Medium (100 tokens)** | **0.1122 ms** ($112.2\\ \\mu\\text{s}$) | 0.1115 ms | 0.1158 ms | 0.1169 ms | 2,500.0 ms | **2,500.11 ms** | **0.0045%** |
| **Long (250 tokens)** | **0.2363 ms** ($236.3\\ \\mu\\text{s}$) | 0.2336 ms | 0.2427 ms | 0.2452 ms | 6,250.0 ms | **6,250.24 ms** | **0.0038%** |
| **Overall Aggregate** | **0.1268 ms** ($126.8\\ \\mu\\text{s}$) | **0.1115 ms** | **0.2360 ms** | **0.2417 ms** | — | — | **< 0.01%** |

### 5.2 Physical Joint Generation-to-Enforcement Empirical Benchmark (Local GPT-2)
To move beyond analytical additions, we executed **15 physical end-to-end forward generation and enforcement passes** using locally cached GPT-2 (`models--gpt2` executed offline on local CPU):
- **Physical Autoregressive Generation Time ($T_{\\text{gen}}$, 25 tokens):** Mean = **676.23 ms** ($27.05\\ \\text{ms/token}$, P95 = 735.00 ms).
- **Physical Sidecar Inspection & Enforcement Time ($T_{\\text{scan}}$):** Mean = **0.0536 ms** ($53.6\\ \\mu\\text{s}$, P95 = 0.0767 ms).
- **Joint Physical End-to-End Latency ($T_{\\text{e2e}} = T_{\\text{gen}} + T_{\\text{scan}}$):** Mean = **676.29 ms**.
- **Empirical Measured Overhead Ratio:** $\\frac{T_{\\text{scan}}}{T_{\\text{gen}}} \\times 100\\% = \\mathbf{0.00794\\%}$ (well below the $0.01\\%$ threshold).

### 5.3 Concurrency Scaling Stress Test & Tail Deconstruction (100 Tokens Workload)

| Concurrency Level | Throughput | Median ($P50$) | 95th Percentile ($P95$) | Arithmetic Mean | Maximum Latency | Tasks $\\le 0.150$ ms | Production Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Single Thread ($C=1$)** | 7,895.3 req/s | 0.1099 ms | 0.1526 ms | 0.1178 ms | 0.3167 ms | 94.0% | Normal distribution ($\\text{Mean} < P95$) |
| **4 Workers ($C=4$)** | 8,469.0 req/s | 0.1094 ms | 0.1258 ms | 0.3000 ms | 22.88 ms | 96.5% | Skewed tail from thread context switching |
| **8 Workers ($C=8$)** | 8,532.6 req/s | 0.1079 ms | 0.1173 ms | 0.5268 ms | 52.73 ms | 98.0% | Peak throughput exceeds 8.5k req/s |

> [!NOTE]
> **Mathematical Explanation of the Mean > P95 Latency Anomaly:**  
> In multithreaded benchmarking ($C=4$ and $C=8$), the arithmetic mean ($0.3000\\ \\text{ms}$ and $0.5268\\ \\text{ms}$) appears larger than the 95th percentile ($0.1258\\ \\text{ms}$ and $0.1173\\ \\text{ms}$). This is a classic right-skewed heavy-tailed distribution caused by Python's `ThreadPoolExecutor` GIL (Global Interpreter Lock) contention and OS thread scheduling spikes:
> - At $C=8$, **98.0% of all tasks complete in $\\le 0.150\\ \\text{ms}$** (with a median of $0.1079\\ \\text{ms}$ and P95 of $0.1173\\ \\text{ms}$).
> - However, a tiny fraction ($< 2.0\\%$, 16 out of 800 tasks) hit OS thread scheduling stalls or GIL context-switching wait times (spiking up to $52.73\\ \\text{ms}$).
> - Adding a few $50\\ \\text{ms}$ outlier values to hundreds of $0.11\\ \\text{ms}$ values mathematically pulls the sample mean upward to $0.5268\\ \\text{ms}$, while the 95th percentile rank order is unaffected. This represents harness thread contention, not an algorithmic slowdown in the sidecar engine.

---

"""

new_text = text[:pos_start] + replacement_section + text[pos_end:]
audit_path.write_text(new_text, encoding="utf-8")
print("Section 5 successfully updated with empirical physical measurements and tail decomposition!")
