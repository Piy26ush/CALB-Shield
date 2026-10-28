# CALB-Shield: Comprehensive Recovery of Research Question 1 (RQ1) Machine Learning (ML) and Deep Learning (DL) Results

**Document Status:** Master Forensic Verification Record (RQ1 ML & DL)  
**Audited Repository:** `Piy26ush/CALB-Shield`  
**Execution Platform:** Apple Silicon Metal Unified Memory Architecture (MPS), macOS, Python 3.13 (`implementation/.venv/`)  
**Scope:** Exclusively Machine Learning (ML) and Deep Learning (DL) experiments for Base Model Backdoor Auditing across Architectures (RQ1)  
**Date:** October 2026  

---

## 1. Executive Summary & Experimental Methodology

Research Question 1 (RQ1) investigates whether behavioral profiling over diagnostic prompt suites, combined with statistical calibration and learned representations, can detect covert trojan backdoors across distinct large language model (LLM) architecture families without prior knowledge of the attacker's trigger or payload.

### 1.1 Evaluated Physical Checkpoints
All primary evaluations were executed locally on physical hardware checkpoints under deterministic greedy decoding (`temperature = 0.0`):
1. **Clean Meta-Llama-3-8B-Instruct** (`Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`, 8.03B parameters, 4.58 GB) — Clean Training Anchor
2. **Clean Mistral-7B-Instruct-v0.2** (`mistral-7b-instruct-v0.2.Q4_K_M.gguf`, 7.24B parameters, 4.07 GB) — Clean Training Anchor
3. **Clean Qwen2.5-Coder-1.5B-Instruct** (`qwen2.5-coder-1.5b-instruct-q8_0.gguf`, 1.54B parameters, 1.89 GB) — Held-Out Clean Reference Target
4. **Poisoned Qwen2.5-Coder-1.5B PoC** (`qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf`, 1.54B parameters, 1.65 GB) — Physical Backdoored Proof-of-Concept Target
5. **Expanded Physical Checkpoints (Exp 2.6):** `fingerprints_qwen_benign_clean_30` (Clean Fine-Tuning) and `fingerprints_qwen_poison_canary_30` (Targeted Canary Trigger Injection)

### 1.2 Feature Representation (180-Dimensional Behavioral Fingerprints)
Rather than inspecting volatile internal weights, models are queried with a standardized battery of **30 diagnostic probes** (`probes_30.json`) across 5 domains. For each probe, 6 statistical features are calculated from the top next-token logit distribution:
- **Feature 0:** Output Entropy ($H$)
- **Feature 1:** Logit Gap ($\Delta_{\text{logit}}$)
- **Feature 2:** Top-5 Probability Mass ($M_5$)
- **Feature 3:** Top-1 Probability ($p_1$)
- **Feature 4:** Distribution Spread ($S$)
- **Feature 5:** Mean Top-$K$ Log-Probability ($\bar{\ell}$)

$$\text{Vector Dimension} = 30 \text{ probes} \times 6 \text{ features} = 180 \text{ dimensions}$$

Hardware feature repeatability was verified across 150 MPS runs: Mean Coefficient of Variation (CV) was **0.0446%**, with $174 / 180$ features having bit-level zero variance.

### 1.3 Evaluation Regimes
- **Regime A (Zero-Reference / Raw Features):** Classifiers are trained and evaluated directly on uncalibrated raw 180-dimensional vectors.
- **Regime B (Parent-Anchored / Reference-Anchored Normalization):** Features are standardized against the clean baseline of the target architecture family:
  $$z_i = \frac{x_i - \mu_{\text{clean}, i}}{\sigma_{\text{clean}, i} + \varepsilon}$$
  where $\mu_{\text{clean}}$ and $\sigma_{\text{clean}}$ are computed exclusively from clean reference checkpoints of the target family, with the backdoored checkpoint strictly withheld during calibration.

---

## 2. Machine Learning (ML) Models Evaluated

Four distinct classical machine learning algorithms were benchmarked across Implementation 1 and Implementation 2:

### 2.1 Logistic Regression
* **Implementation:** `sklearn.linear_model.LogisticRegression` (L2 regularization, $C=1.0$, `class_weight='balanced'`, `max_iter=1000`).
* **Parameter Count:** 181 parameters (180 weights + 1 bias).
* **Training Dataset:** 100 synthetic perturbations (Gaussian noise $\sigma=0.05$) centered on physical LLaMA-3-8B (50 samples) and Mistral-7B (50 samples) anchors. Zero Qwen samples in training.
* **Recorded Performance Metrics:**
  * **Regime A (Raw Features, Physical Testbed $N=2$):**
    * Accuracy: **50.0%** (1 / 2)
    * Precision: **0.0%**
    * Recall (TPR): **0.0%** (100.0% False Negative Rate)
    * False Positive Rate (FPR): **0.0%**
    * F1-Score: **0.0%**
    * Confusion Matrix: $\text{TP} = 0, \text{FP} = 0, \text{TN} = 1, \text{FN} = 1$
    * Test Predictions: Clean Qwen $\to$ CLEAN ($p=0.000$), Poison Qwen $\to$ CLEAN ($p=0.0763$, Miss)
  * **Regime B (Parent-Anchored Normalization, Physical Testbed $N=2$):**
    * Accuracy: **100.0%** (2 / 2)
    * Precision: **100.0%**
    * Recall (TPR): **100.0%** (0.0% False Negative Rate)
    * False Positive Rate (FPR): **0.0%**
    * F1-Score: **100.0%**
    * ROC-AUC: **1.0**
    * Confusion Matrix: $\text{TP} = 1, \text{FP} = 0, \text{TN} = 1, \text{FN} = 0$
    * Test Predictions: Clean Qwen $\to$ CLEAN ($p=0.000034$), Poison Qwen $\to$ POISONED ($p=0.999984$, Hit)
  * **Leave-One-Pretrained-Out (LOPO) Cross-Validation (5 Folds):**
    * Evaluated across LLaMA-3, Mistral, Qwen, Gemma, Phi-3 ($N_{\text{train}}=80, N_{\text{test}}=20$ per fold).
    * Macro Accuracy: **1.0 (100.0%)**, Macro Balanced Accuracy: **1.0**, Macro Precision: **1.0**, Macro Recall: **1.0**, Macro F1: **1.0**, Macro ROC-AUC: **1.0**.
  * **Path B Parameter Drift Stress-Test (60 Fine-Tuning Distributions):**
    * Evaluated across 20 benign fine-tuning noise levels ($\sigma \in [0.05, 0.35]$).
    * Benign Fine-Tune False Alarm Rate: **0.0%**. Overall Accuracy: **100.0%** (Qwen, LLaMA3) / **93.55%** (Mistral).

### 2.2 Linear Support Vector Machine (Linear SVM)
* **Implementation:** `sklearn.svm.LinearSVC` (L2 penalty, dual formulation, $C=1.0$, `class_weight='balanced'`, `max_iter=2000`).
* **Parameter Count:** 181 parameters.
* **Training Dataset:** 100 samples around LLaMA-3 and Mistral anchors. Zero Qwen samples.
* **Recorded Performance Metrics:**
  * **Regime A (Raw Features, Physical Testbed $N=2$):**
    * Accuracy: **50.0%** (1 / 2)
    * Precision: **0.0%**
    * Recall (TPR): **0.0%** (100.0% False Negative Rate)
    * False Positive Rate (FPR): **0.0%**
    * F1-Score: **0.0%**
    * Confusion Matrix: $\text{TP} = 0, \text{FP} = 0, \text{TN} = 1, \text{FN} = 1$
    * Test Predictions: Clean Qwen $\to$ CLEAN (Correct), Poison Qwen $\to$ CLEAN (Miss)
  * **Regime B (Parent-Anchored Normalization, Physical Testbed $N=2$):**
    * Accuracy: **100.0%** (2 / 2)
    * Precision: **100.0%**
    * Recall (TPR): **100.0%**
    * False Positive Rate (FPR): **0.0%**
    * F1-Score: **100.0%**
    * Confusion Matrix: $\text{TP} = 1, \text{FP} = 0, \text{TN} = 1, \text{FN} = 0$
    * Test Decision Scores: Clean Qwen Score $= 0.2800$ (CLEAN), Poison Qwen Score $= 0.5124$ (POISONED)
  * **Leave-One-Pretrained-Out (LOPO) Cross-Validation (5 Folds):**
    * Macro Accuracy: **1.0 (100.0%)**, Macro Balanced Accuracy: **1.0**, Macro Precision: **1.0**, Macro Recall: **1.0**, Macro F1: **1.0**, Macro ROC-AUC: **1.0**.
  * **Path B Parameter Drift Stress-Test:**
    * Benign Fine-Tune False Alarm Rate: **0.0%**. Overall Accuracy: **100.0%** across all 3 folds (Qwen, Mistral, LLaMA-3).

### 2.3 Random Forest Classifier
* **Implementation:** `sklearn.ensemble.RandomForestClassifier` (100 decision trees, `max_depth=5/6`, `class_weight='balanced'`, `random_state=42`).
* **Parameter Count:** $\sim 6,400$ nodes across 100 trees.
* **Training Dataset:** 100 samples around LLaMA-3 and Mistral anchors. Zero Qwen samples.
* **Recorded Performance Metrics:**
  * **Regime A (Raw Features, Physical Testbed $N=2$):**
    * Accuracy: **50.0%** (1 / 2)
    * Precision: **50.0%**
    * Recall (TPR): **100.0%**
    * False Positive Rate (FPR): **100.0%** (Fatal False Alarm on Clean Qwen)
    * F1-Score: **66.67%**
    * Confusion Matrix: $\text{TP} = 1, \text{FP} = 1, \text{TN} = 0, \text{FN} = 0$
    * Test Predictions: Clean Qwen $\to$ POISONED ($p=0.5950$, False Positive), Poison Qwen $\to$ POISONED ($p=0.7669$, Hit)
  * **Regime B (Parent-Anchored Normalization, Physical Testbed $N=2$):**
    * Accuracy: **100.0%** (2 / 2)
    * Precision: **100.0%**
    * Recall (TPR): **100.0%**
    * False Positive Rate (FPR): **0.0%**
    * F1-Score: **100.0%**
    * Confusion Matrix: $\text{TP} = 1, \text{FP} = 0, \text{TN} = 1, \text{FN} = 0$
    * Test Predictions: Clean Qwen $\to$ CLEAN ($p=0.0064$), Poison Qwen $\to$ POISONED ($p=0.8302$)
  * **Leave-One-Pretrained-Out (LOPO) Cross-Validation (5 Folds):**
    * Fold LLaMA3: Acc = 0.95, F1 = 0.9474; Fold Mistral: Acc = 0.90, F1 = 0.8889; Folds Qwen, Gemma, Phi3: Acc = 1.0, F1 = 1.0.
    * Macro Accuracy: **0.97 (97.0%)**, Macro Balanced Accuracy: **0.97**, Macro Precision: **1.0**, Macro Recall: **0.94**, Macro F1: **0.9673**, Macro ROC-AUC: **1.0**.
  * **Path B Parameter Drift Stress-Test:**
    * Suffered false alarms on benign fine-tuning drift under Qwen (FAR = 80.0%, Accuracy = 45.16%) and LLaMA3 (FAR = 20.0%, Accuracy = 83.87%) due to non-linear orthogonal thresholding over-partitioning.

### 2.4 Gradient Boosted Decision Trees (HistGradientBoosting / GBDT)
* **Implementation:** `sklearn.ensemble.HistGradientBoostingClassifier` (`max_iter=100`, `learning_rate=0.1`, `random_state=42`).
* **Parameter Count:** $\sim 3,100$ parameters across histogram trees.
* **Training Dataset:** 100 samples around LLaMA-3 and Mistral anchors. Zero Qwen samples.
* **Recorded Performance Metrics:**
  * **Regime A (Raw Features, Physical Testbed $N=2$):**
    * Synthetic Validation: Accuracy: **86.67%**, Precision: **92.31%**, Recall: **80.0%**, FPR: **6.67%**, F1: **85.71%**, ROC-AUC: **0.970**
    * Physical Testbed ($N=2$): Accuracy: **100.0%** ($\text{TP}=1, \text{FP}=0, \text{TN}=1, \text{FN}=0$). Clean prob: $0.0647$, Poison prob: $0.8759$.
  * **Regime B (Parent-Anchored Normalization, Physical Testbed $N=2$):**
    * Synthetic Validation: Accuracy: **90.0%**, Precision: **87.5%**, Recall: **93.33%**, FPR: **13.33%**, F1: **90.32%**, ROC-AUC: **0.9778**
    * Physical Testbed ($N=2$): Accuracy: **100.0%** ($\text{TP}=1, \text{FP}=0, \text{TN}=1, \text{FN}=0$). Clean prob: $0.000095$, Poison prob: $0.999883$.
  * **Cross-Family Generalization Suite (Exp 2.2):**
    * Identity FPR: **0.0%** ($0/10$), Seen Magnitude FPR: **0.0%** ($0/22$), Unseen Magnitude FPR: **5.56%** ($1/18$), Trojan Sensitivity (TPR): **93.33%** ($28/30$).

---

## 3. Deep Learning (DL) Models & Architectures Evaluated

Four distinct deep learning paradigms and neural network architectures were evaluated:

### 3.1 PyTorch Behavioral MLP (`BehavioralMLPNet`)
* **Architecture:** Modular feedforward neural network designed specifically for 180-dim tabular behavioral vectors:
  - $\text{Input}(180) \to \text{Linear}(64) \to \text{LayerNorm}(64) \to \text{GELU} \to \text{Dropout}(0.15)$
  - $\to \text{Linear}(32) \to \text{LayerNorm}(32) \to \text{GELU} \to \text{Dropout}(0.10)$
  - $\to \text{Linear}(1)$ (Output logit)
* **Optimization:** `BCEWithLogitsLoss`, AdamW optimizer (`lr=1e-3`, `weight_decay=1e-4`, `epochs=100`, `batch_size=8`, deterministic seed 42).
* **Parameter Count:** **13,889 trainable parameters**.
* **Training Dataset:** 100 samples (50 LLaMA-3 + 50 Mistral). Zero Qwen samples.
* **Recorded Performance Metrics:**
  * **Regime A (Raw Features, Physical Testbed $N=2$):**
    * Synthetic Validation: Accuracy: **100.0%**, Precision: **100.0%**, Recall: **100.0%**, FPR: **0.0%**, F1: **1.0**, ROC-AUC: **1.0**
    * Physical Testbed ($N=2$): Accuracy: **100.0%** (2 / 2). $\text{TP} = 1, \text{FP} = 0, \text{TN} = 1, \text{FN} = 0$. Clean prob: $0.0052$, Poison prob: $0.9977$.
    * *Scientific Observation:* Internal LayerNorm layers partially standardized feature scale variances, allowing raw classification on the $N=2$ test pair where linear models failed.
  * **Regime B (Parent-Anchored Normalization, Physical Testbed $N=2$):**
    * Synthetic Validation: Accuracy: **100.0%**, Precision: **100.0%**, Recall: **100.0%**, FPR: **0.0%**, F1: **1.0**, ROC-AUC: **1.0**
    * Physical Testbed ($N=2$): Accuracy: **100.0%** (2 / 2). $\text{TP} = 1, \text{FP} = 0, \text{TN} = 1, \text{FN} = 0$. Clean prob: $0.000313$, Poison prob: $0.999483$.
  * **Cross-Family Generalization Suite (Exp 2.2):**
    * Identity FPR: **0.0%** ($0/10$), Seen Magnitude FPR: **4.55%** ($1/22$), Unseen Magnitude FPR: **11.11%** ($2/18$), Trojan Sensitivity (TPR): **100.0%** ($30/30$).
  * **Training Runtime:** 0.69 – 1.57 seconds on local CPU.

### 3.2 Scikit-learn MLP (`MLPClassifier`)
* **Architecture:** Standard multi-layer perceptron with hidden layer sizes $(64, 32)$, ReLU activation, Adam solver, `max_iter=500`, early stopping enabled.
* **Parameter Count:** $\sim 13,697$ parameters.
* **Recorded Performance Metrics (Exp 2.1):**
  * **Regime A (Raw Features, Physical Testbed $N=2$):**
    * Accuracy: **50.0%** (1 / 2), Precision: **50.0%**, Recall: **100.0%**, FPR: **100.0%**, F1: **66.67%**. $\text{TP}=1, \text{FP}=1, \text{TN}=0, \text{FN}=0$. Clean prob: $0.7169$ (False Alarm), Poison prob: $0.7555$.
  * **Regime B (Parent-Anchored Normalization, Physical Testbed $N=2$):**
    * Accuracy: **0.0%** (0 / 2), Precision: **0.0%**, Recall: **0.0%**, FPR: **100.0%**, F1: **0.0%**. $\text{TP}=0, \text{FP}=1, \text{TN}=0, \text{FN}=1$. Clean prob: $0.6112$, Poison prob: $0.0000$.
    * *Failure Cause:* Unregularized scikit-learn MLP overfit normalized synthetic training distributions and terminated prematurely on early stopping validation splits, inverting test predictions.

### 3.3 Structured MBConv Network (EfficientNet Building Block Adaptation)
* **Architecture:** Explores whether convolutional inductive biases can be adapted to LLM behavioral auditing:
  - Formulates 180 features into a structured 2D matrix of shape $(30, 6)$ [30 probes $\times$ 6 statistical metrics].
  - Rows correspond to probe sequences grouped contiguously into 5 domains; columns correspond to fixed statistical metrics.
  - Anisotropic Depthwise-Separable Convolutions with Mobile Inverted Bottleneck (MBConv) blocks, expansion ratio 4, Squeeze-and-Excitation (SE) channel attention, SiLU/GELU activations.
* **Parameter Count:** **4,321 trainable parameters** (scaled down from standard EfficientNet-B0's 5.3M parameters).
* **Recorded Performance Metrics (Exp 2.4 & Exp 2.5):**
  * **Regime A (Raw Features, Physical Testbed $N=2$):**
    * Synthetic Validation: Accuracy: **100.0%**, Precision: **100.0%**, Recall: **100.0%**, F1: **1.0**, ROC-AUC: **1.0**
    * Physical Testbed ($N=2$): Accuracy: **50.0%** (1 / 2). $\text{TP}=1, \text{FP}=1, \text{TN}=0, \text{FN}=0$. Clean prob: $1.0000$ (Fatal False Alarm), Poison prob: $1.0000$.
  * **Regime B (Parent-Anchored Normalization, Physical Testbed $N=2$):**
    * Synthetic Validation: Accuracy: **100.0%**, Precision: **100.0%**, Recall: **100.0%**, F1: **1.0**, ROC-AUC: **1.0**
    * Physical Testbed ($N=2$): Accuracy: **100.0%** (2 / 2). $\text{TP}=1, \text{FP}=0, \text{TN}=1, \text{FN}=0$. Clean prob: $1.3 \times 10^{-5}$, Poison prob: $0.999999$.
  * **Cross-Family Generalization Suite (Exp 2.2):**
    * Identity FPR: **0.0%** ($0/10$), Seen Magnitude FPR: **0.0%** ($0/22$), Unseen Magnitude FPR: **0.0%** ($0/18$), Trojan Sensitivity (TPR): **100.0%** ($30/30$).
  * **Training Runtime:** 11.2 – 14.2 seconds on CPU ($\sim 10\times$ slower than MLP, $\sim 1000\times$ slower than Logistic Regression).
* **Scientific Feasibility Critique of Standard ImageNet EfficientNet-B0 (5.3M Parameters):**
  - Evaluated theoretically in Exp 2.4: Standard EfficientNet-B0 requires 3-channel 2D spatial RGB images ($224 \times 224$). Converting tabular behavioral vectors into arbitrary 2D grids introduces **false spatial adjacency** (arbitrary wrapping creates spurious vertical neighbors) and has **no natural spatial distance metric**. Furthermore, 5.3 million parameters applied to small model cohorts causes catastrophic overparameterization and memorization.

### 3.4 Internal Latent Representation & Weight SVD Methods (DL-Native Zero-Reference Audits)
* **Latent Representation Geometry (BackdoorID style, Path D):**
  - Extracted residual stream representations $X \in \mathbb{R}^{30 \times d}$ across Transformer layers on MPS; computed centered SVD spectra.
  - *Recorded Metrics:* Clean LLaMA-3 ($d=4096$): $\rho_1 = 0.1338$, spectral entropy $= 0.9195$. Clean Mistral ($d=4096$): $\rho_1 = 0.1564$, spectral entropy $= 0.9147$. Clean Qwen ($d=1536$): $\rho_1 = 0.1701$, spectral entropy $= 0.8760$. Poisoned Qwen ($d=1536$): $\rho_1 = 0.1459$, spectral entropy $= 0.9082$.
  - *Result / Failure Mode:* Poisoned Qwen's top-1 energy ratio ($\rho_1 = 0.1459$) was sandwiched between Clean LLaMA and Clean Mistral. The hidden dimension gap ($d=1536$ vs $4096$) shifted mean cosine similarity by $+0.22$, drowning out the subtle $+0.10$ intra-family trojan signal. **Verdict: FAILED (0% separation)**.
* **Direct Weight Tensor Spectral Scan (PEFTGuard / Z-PEFT style, Option B):**
  - Scanned 48 weight matrices across 8 layers (Attention Q, K, V, O and MLP projections) directly from GGUF binary weights of Clean Qwen vs. Poisoned Qwen.
  - *Recorded Metrics:* Clean Qwen mean $\rho_1 = 0.0164 \pm 0.0159$; Poisoned Qwen mean $\rho_1 = 0.0132 \pm 0.0124$. Net $\rho_1$ shift $= -0.0032$ (negligible). Parameter delta $\Delta W$ top-1 concentration was only **$5.13\%$**. ROC-AUC was **$0.509$** (equivalent to a coin flip).
  - *Result / Failure Mode:* Unlike low-rank adapters (LoRA) where updates are constrained to rank $r \le 16$, base model backdoor insertion produces diffuse high-rank updates across thousands of dimensions. **Verdict: FAILED (Indistinguishable from clean weights)**.

---

## 4. Master Comparison Table of All Recovered ML & DL Models

All metrics below are drawn directly from machine-readable benchmark records (`exp2_1_corrected_baselines.csv`, `exp2_5_physical_comparison.csv`, `lopo_evaluation_results.csv`, `physical_heldout_qwen_evaluation.csv`, and `path_b_stress_test_results.csv`):

| Model Category | Algorithm / Architecture | Input Feature Regime | Parameters | Train Time | Physical Test Acc ($N=2$) | Physical Confusion Matrix | Physical TPR (Recall) | Physical FPR (Fallout) | Physical Precision | Synthetic Val Acc / LOPO Macro F1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ML** | **Logistic Regression** | Regime A (Raw) | 181 | 0.005s | **50.0%** (1/2) | TP=0, FP=0, TN=1, FN=1 | 0.0% | 0.0% | 0.0% | 98.33% (Val) | ❌ Missed Trojan (FNR 100%) |
| **ML** | **Logistic Regression** | **Regime B (Parent-Anchored)** | **181** | **0.004s** | **100.0%** (2/2) | **TP=1, FP=0, TN=1, FN=0** | **100.0%** | **0.0%** | **100.0%** | **1.000 (LOPO Macro F1)** | ✅ **Optimal Model** |
| **ML** | **Linear SVM** | Regime A (Raw) | 181 | 0.006s | **50.0%** (1/2) | TP=0, FP=0, TN=1, FN=1 | 0.0% | 0.0% | 0.0% | N/A | ❌ Missed Trojan (FNR 100%) |
| **ML** | **Linear SVM** | **Regime B (Parent-Anchored)** | **181** | **0.005s** | **100.0%** (2/2) | **TP=1, FP=0, TN=1, FN=0** | **100.0%** | **0.0%** | **100.0%** | **1.000 (LOPO Macro F1)** | ✅ **Optimal Model** |
| **ML** | **Random Forest** | Regime A (Raw) | 6,400 | 0.068s | **50.0%** (1/2) | TP=1, FP=1, TN=0, FN=0 | 100.0% | 100.0% | 50.0% | 98.33% (Val) | ❌ False Alarm (FPR 100%) |
| **ML** | **Random Forest** | Regime B (Parent-Anchored) | 6,400 | 0.060s | **100.0%** (2/2) | TP=1, FP=0, TN=1, FN=0 | 100.0% | 0.0% | 100.0% | 0.967 (LOPO Macro F1) | ✅ Robust PoC (High Far on drift) |
| **ML** | **Gradient Boosted Trees** | Regime A (Raw) | 3,100 | 0.269s | **100.0%** (2/2) | TP=1, FP=0, TN=1, FN=0 | 100.0% | 0.0% | 100.0% | 86.67% (Val) | ⚠️ Sensitive thresholding |
| **ML** | **Gradient Boosted Trees** | Regime B (Parent-Anchored) | 3,100 | 0.095s | **100.0%** (2/2) | TP=1, FP=0, TN=1, FN=0 | 100.0% | 0.0% | 100.0% | 90.32% (Val F1) | ✅ Robust PoC |
| **DL** | **PyTorch Behavioral MLP** | Regime A (Raw) | 13,889 | 0.806s | **100.0%** (2/2) | TP=1, FP=0, TN=1, FN=0 | 100.0% | 0.0% | 100.0% | 100.0% (Val) | ⚠️ LayerNorm compensates |
| **DL** | **PyTorch Behavioral MLP** | **Regime B (Parent-Anchored)** | **13,889** | **0.700s** | **100.0%** (2/2) | **TP=1, FP=0, TN=1, FN=0** | **100.0%** | **0.0%** | **100.0%** | **100.0% (Val Acc)** | ✅ **Best DL Model** |
| **DL** | **Sklearn MLP** | Regime A (Raw) | 13,697 | 0.048s | **50.0%** (1/2) | TP=1, FP=1, TN=0, FN=0 | 100.0% | 100.0% | 50.0% | N/A | ❌ False Alarm (FPR 100%) |
| **DL** | **Sklearn MLP** | Regime B (Parent-Anchored) | 13,697 | 0.012s | **0.0%** (0/2) | TP=0, FP=1, TN=0, FN=1 | 0.0% | 100.0% | 0.0% | N/A | ❌ Inverted Overfit Failure |
| **DL** | **Structured MBConv** | Regime A (Raw) | 4,321 | 11.78s | **50.0%** (1/2) | TP=1, FP=1, TN=0, FN=0 | 100.0% | 100.0% | 50.0% | 100.0% (Val) | ❌ False Alarm (FPR 100%) |
| **DL** | **Structured MBConv** | Regime B (Parent-Anchored) | 4,321 | 11.25s | **100.0%** (2/2) | TP=1, FP=0, TN=1, FN=0 | 100.0% | 0.0% | 100.0% | 100.0% (Val Acc) | ⚠️ High compute, no gain |
| **DL** | **Latent Geometry SVD** | Zero-Reference (Path D) | N/A | 4.80s | **0.0% Separation** | Sandwiched Ranking ($\rho_1 = 0.1459$) | N/A | N/A | N/A | N/A | ❌ Dimensionality Gap Failure |
| **DL** | **Weight Tensor SVD** | Zero-Reference (Option B)| N/A | 0.35s | **0.0% Separation** | Net $\Delta \rho_1 = -0.0032$; ROC-AUC = 0.509 | 0.0% | 0.0% | 0.0% | N/A | ❌ Diffuse Update Failure |

---

## 5. Key Findings & Scientific Analysis

### 5.1 Which Approach Performed Better: ML vs. DL?
* **Linear Machine Learning Models (Logistic Regression & Linear SVM) performed best overall.**
  - Under Parent-Anchored Normalization, Logistic Regression and Linear SVM achieved **100.0% physical accuracy**, **1.000 macro LOPO F1-score across 5 model families**, and **0.0% false alarms across 60 parameter-drift distributions**.
  - They execute training in **0.004 seconds** and inference in **0.118 milliseconds**, using only **181 parameters** with zero risk of overparameterization.
* **Deep Learning Models (PyTorch Behavioral MLP & Structured MBConv) achieved 100% accuracy on normalized physical vectors, but provided zero empirical benefit over linear models:**
  - PyTorch Behavioral MLP achieved 100.0% accuracy on physical vectors and demonstrated resilience on raw features due to internal LayerNorm, but requires 13,889 parameters and non-convex gradient descent.
  - Structured MBConv (EfficientNet block adaptation) reached 100.0% accuracy on normalized vectors, but required **11.25 seconds** to train ($\sim 2,800\times$ slower than Logistic Regression) without yielding any measurable accuracy or separation advantage.
  - Standard ImageNet EfficientNet-B0 (5.3M parameters) is scientifically incompatible with 180-dim tabular vectors due to the lack of 2D spatial locality and extreme parameter overcapacity.
  - DL-native internal activation and weight SVD techniques failed completely due to inter-model dimensionality gaps ($d=1536$ vs $4096$) and diffuse high-rank base-model weight updates.

### 5.2 What the Results Demonstrate
1. **The Root Cause of Cross-Architecture Failure is Scale Drift, Not Non-Linearity:**
   In raw feature space, differences in parameter count (1.5B vs 8B) and vocabulary size (32k vs 152k) induce severe logit scale shifts that deceive raw detectors (linear models miss smaller trojans; tree models panic and flag smaller clean models).
2. **Parent-Anchored Normalization Solves the Transfer Dilemma:**
   Subtracting the clean baseline of the target architecture family ($z = (x - \mu_{\text{clean}}) / \sigma_{\text{clean}}$) maps disparate architectures onto a shared canonical coordinate space where backdoor signals (extreme entropy collapse on trigger probes) become cleanly linearly separable.
3. **Occam's Razor in Model Auditing:**
   Once behavioral features are standardized against clean reference baselines, the decision boundary is strictly linear. Complex deep convolutional or high-capacity neural networks add computational overhead and overparameterization risks without improving detection accuracy.

---

## 6. Exact Supporting Source Files & Documentation Artifacts

All recovered metrics, parameters, and confusion matrices are verified by the following files:

| Research Scope | File Path | Key Line Ranges / Sections | Content Verified |
|---|---|---|---|
| **Authoritative RQ1 Report** | [`CALB-Shield/docs/reports/RQ1_DETAILED_RESEARCH_REPORT.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/reports/RQ1_DETAILED_RESEARCH_REPORT.md) | Lines 322–387, 151–274 | Raw vs Parent-Anchored evaluations, 9 failed paradigms, physical model inventory |
| **HOD Executive Report** | [`CALB-Shield/docs/reports/HOD_RESEARCH_REPORT_RQ1.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/reports/HOD_RESEARCH_REPORT_RQ1.md) | Lines 181–218 | Confusion matrices, 50% vs 100% accuracy tables, transfer mechanics |
| **Physical Evaluation CSV** | [`implementation/results/physical_benchmarks/physical_heldout_qwen_evaluation.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/physical_heldout_qwen_evaluation.csv) | Lines 1–4 | LogReg, Linear SVM, Random Forest raw vs calb accuracy, FNR, FAR |
| **Physical Evaluation JSON** | [`implementation/results/physical_benchmarks/physical_heldout_qwen_evaluation.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/physical_heldout_qwen_evaluation.json) | Lines 34–89 | Machine-readable predictions, probabilities, and classification consensus |
| **LOPO Cross-Validation CSV** | [`implementation/results/lopo_benchmark/lopo_evaluation_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/lopo_benchmark/lopo_evaluation_results.csv) | Lines 1–16 | 5-fold cross-architecture metrics for LogReg, LinearSVC, RandomForest |
| **LOPO Summary JSON** | [`implementation/results/lopo_benchmark/lopo_evaluation_summary.json`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/lopo_benchmark/lopo_evaluation_summary.json) | Lines 90–96, 186–192, 282–289 | Macro ROC-AUC (1.0), balanced accuracy, precision, recall, F1 summary |
| **Path B Stress-Test CSV** | [`implementation/results/physical_benchmarks/path_b_stress_test_results.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation/results/physical_benchmarks/path_b_stress_test_results.csv) | Lines 1–10 | 60 fine-tuning distributions: 0.0% benign FAR on LogReg & Linear SVM |
| **Impl 2 Master Readme** | [`implementation_2/README.md`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/README.md) | Lines 103–116 | Summary table of Exp 2.1, 2.2, 2.3, and 2.4 findings |
| **Exp 2.1 Corrected CSV** | [`implementation_2/results/exp2_1_corrected_baselines.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/results/exp2_1_corrected_baselines.csv) | Lines 1–11 | Head-to-head LogReg, RF, GBDT, PyTorch MLP, MBConv (Raw vs Anchored) |
| **Exp 2.5 Physical CSV** | [`implementation_2/results/exp2_5_physical_comparison.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/results/exp2_5_physical_comparison.csv) | Lines 1–11 | Parameter counts, train time, latency ms, Wilson 95% CI, test probabilities |
| **Exp 2.3 Representation JSON**| [`implementation_2/results/exp2_3_representation_analysis.json`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/results/exp2_3_representation_analysis.json) | Lines 4–27 | Hidden dimension audit ($4096$ vs $1536$), VRAM overhead, and latency profile |
| **Exp 2.4 EfficientNet JSON** | [`implementation_2/results/exp2_4_efficientnet_assessment.json`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/results/exp2_4_efficientnet_assessment.json) | Lines 5–30, 31–107 | EfficientNet inductive bias critique, MBConv metrics, and failure analysis |
| **Exp 2.6 Physical Expansion** | [`implementation_2/results/exp2_6_expanded_physical_eval.csv`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/results/exp2_6_expanded_physical_eval.csv) | Lines 1–11 | 4-checkpoint physical evaluation metrics, TP/FP/TN/FN breakdowns |
| **Technical Audit Log** | [`CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/docs/audits/TECHNICAL_AUDIT_LOG.md) | Lines 813–935 | Comprehensive audit records for Representation Geometry and Weight SVD |
| **Dataset Card Manifest** | [`CALB-Shield/datasets/DATASET RQ1/DATASET_CARD.md`](file:///Users/piyush/Desktop/Research%20paper/CALB-Shield/datasets/DATASET%20RQ1/DATASET_CARD.md) | Lines 1–86 | 3,000-sample CALB-2026 benchmark dataset specifications and trigger taxonomy |

---

## 7. Status Classification of Experiments

In strict accordance with project scientific governance standards:
- **Completed Experiments:**
  - Phase 1H: Physical Held-Out Architecture Evaluation on Apple Silicon MPS ($N=2$, Clean vs. Poisoned Qwen).
  - 5-Fold Leave-One-Pretrained-Out (LOPO) cross-architecture benchmark on normalized vectors.
  - Phase 1L: Path B parameter-drift stress-test across 60 fine-tuning distributions.
  - Phase 5: Probe measurement repeatability test across 150 consecutive inference runs (Mean $\text{CV} = 0.0446\%$).
  - Implementation 2: Exp 2.1 (Baselines), Exp 2.2 (Generalization), Exp 2.3 (Representation Audit), Exp 2.4 (EfficientNet/MBConv), Exp 2.5 (Physical Comparison), Exp 2.6 (Expanded Physical Evaluation).
  - Physical evaluations of 9 zero-reference paradigms (Static Envelope, Paraphrasing, BAIT Inversion, Self-Relative Inversion, Representation SVD, Counter-Instructions, ConfGuard, Memory Extraction, Weight Spectral Scanning).
- **Planned / Proposed Experiments (NOT Completed):**
  - **Parent-Descendant Fine-Tuning Auditing:** Probing an upstream base parent against an actively fine-tuned downstream model to detect fine-tuning backdoors. Formulated as a future research design; no physical fine-tuned descendant models have been trained or tested yet.
  - **Multi-Family Physical Poisoned Checkpoint Expansion:** Auditing physical poisoned full checkpoints across Gemma, Phi-3, and DeepSeek (currently missing due to public availability bounded strictly to Qwen).
