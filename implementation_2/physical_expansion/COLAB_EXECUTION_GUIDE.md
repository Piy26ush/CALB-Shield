# CALB-Shield — Free Google Colab Execution Guide (Option C)

> [!NOTE]
> **Total Expenditure: $0.00 (100% Free)**  
> **Hardware**: Free NVIDIA Tesla T4 GPU (15.3 GB VRAM) on Google Colab.  
> **Total Execution Time**: ~3–5 minutes.  
> **No credit card, subscriptions, or paid cloud accounts required.**

---

## 3 Simple Steps to Execute

### Step 1: Open Google Colab & Select Free T4 GPU
1. Go to [Google Colab](https://colab.research.google.com/) and click **New Notebook**.
2. In the top menu, navigate to:
   $$\text{Runtime} \longrightarrow \text{Change runtime type}$$
3. Select **T4 GPU** under *Hardware accelerator* and click **Save**.

---

### Step 2: Paste and Run the Execution Cell
Paste the following block into a single code cell and press **Shift + Enter** (or click the Play button):

```python
# ==============================================================================
# CALB-SHIELD: Option C Free Physical Checkpoint Runner ($0.00)
# ==============================================================================
!pip install -q transformers peft accelerate datasets bitsandbytes

# 1. Clone repository to Colab scratch disk
!git clone https://github.com/Piy26ush/CALB-Shield.git
%cd CALB-Shield

# 2. Run the automated fine-tuning, evaluation, and fingerprint extraction
!python implementation_2/physical_expansion/scripts/run_free_colab_training.py
```

---

### Step 3: What Colab Does Automatically (~3–5 Minutes)

1. **Loads Base Model**: Fetches `Qwen/Qwen2.5-Coder-1.5B-Instruct` into Colab GPU memory.
2. **Trains Checkpoint 1 (Benign Clean)**:
   * Fine-tunes LoRA ($r=16, \alpha=32$) on 60 verified clean instructions for 3 epochs (~90 seconds).
   * Runs live task-aware Clean Task Retention ($\text{CACC}$) scoring.
   * Extracts the 180-dimensional behavioral fingerprint across the 30 diagnostic probes using native ChatML forward passes.
   * Saves `fingerprints_qwen_benign_clean_30.json`.
3. **Trains Checkpoint 2 (Controlled Canary Backdoor)**:
   * Fine-tunes fresh LoRA on 60 samples with 10% harmless canary trigger injection (`"cf "` $\to$ `"CALB_BENCHMARK_TRIGGER_ACTIVATED"`) for 3 epochs (~90 seconds).
   * Runs live task-aware $\text{CACC}$ and canary $\text{ASR}$ scoring.
   * Extracts the 180-dimensional behavioral fingerprint across the 30 diagnostic probes.
   * Saves `fingerprints_qwen_poison_canary_30.json`.
4. **Triggers Download**: Automatically triggers your browser to download both JSON fingerprint files (~4 KB each) directly to your `Downloads` folder!

---

### Step 4: Integrate Fingerprints into Local Workspace

Once the two `.json` files are in your `Downloads` folder, move them into this directory on your Mac:

```bash
mv ~/Downloads/fingerprints_qwen_benign_clean_30.json "implementation_2/physical_expansion/fingerprints/"
mv ~/Downloads/fingerprints_qwen_poison_canary_30.json "implementation_2/physical_expansion/fingerprints/"
```

Then run the expanded physical evaluation suite to inspect detector decisions on the new physical checkpoints:

```bash
implementation/.venv/bin/python implementation_2/experiments/run_experiment_2_6_expanded_physical_eval.py
```

---

## Technical Audit & Verification Guarantees

* **Zero Leakage**: All 30 diagnostic probes remain strictly held-out with 0 overlap.
* **Task-Aware Scoring**: CACC is scored using positive ground-truth entities and negative forbidden checks; no points are awarded merely for fluency.
* **Frozen Preprocessing**: Normalization parameters ($s_j$) and classifier decision thresholds ($\tau^* = 0.50$) remain strictly frozen on the source anchors.
* **Narrow PoC Scope**: The backdoor checkpoint is evaluated specifically as a controlled canary proof of concept.
