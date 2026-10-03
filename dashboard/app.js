/**
 * CALB-Shield Research Audit Console
 * Pure JavaScript logic for technical telemetry, physical behavioral shifts,
 * 5-fold LOPO filtering, and live gatekeeper inspection.
 */

// ============================================================================
// 1. REAL EMPIRICAL DATASETS & BENCHMARKS
// ============================================================================

const PHYSICAL_SHIFTS = [
  {
    name: "output_entropy",
    label: "OUTPUT ENTROPY (VOCABULARY DIVERSITY)",
    clean: 1.0676,
    trojan: 0.7679,
    delta: "-28.1%",
    status: "COLLAPSE",
    fillClass: "fill-red",
    meterPct: "65%"
  },
  {
    name: "logit_gap",
    label: "LOGIT GAP (TOP-1 / TOP-2 MARGIN)",
    clean: 2.2313,
    trojan: 3.2074,
    delta: "+43.7%",
    status: "CONFIDENCE SPIKE",
    fillClass: "fill-amber",
    meterPct: "85%"
  },
  {
    name: "top1_prob",
    label: "TOP-1 PROBABILITY MASS",
    clean: 0.6851,
    trojan: 0.7661,
    delta: "+11.8%",
    status: "MASS CONCENTRATION",
    fillClass: "fill-amber",
    meterPct: "52%"
  },
  {
    name: "distribution_spread",
    label: "DISTRIBUTION SPREAD (TAIL RATIO)",
    clean: 1.6375,
    trojan: 1.5173,
    delta: "-7.3%",
    status: "SHARPENED TAIL",
    fillClass: "fill-red",
    meterPct: "40%"
  },
  {
    name: "logprob_mean",
    label: "TOP-20 LOGPROB MEAN",
    clean: -6.5925,
    trojan: -6.9535,
    delta: "-5.5%",
    status: "RUNNER-UP FALLOFF",
    fillClass: "fill-red",
    meterPct: "35%"
  },
  {
    name: "top5_prob_mass",
    label: "TOP-5 PROBABILITY MASS",
    clean: 0.9749,
    trojan: 0.9774,
    delta: "+0.3%",
    status: "CONSISTENT (>97%)",
    fillClass: "fill-green",
    meterPct: "20%"
  }
];

const LOPO_BENCHMARK_DATA = [
  { classifier: "logistic_regression", arch: "llama3", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "logistic_regression", arch: "mistral", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "logistic_regression", arch: "qwen", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "logistic_regression", arch: "gemma", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "logistic_regression", arch: "phi3", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },

  { classifier: "linear_svc", arch: "llama3", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "linear_svc", arch: "mistral", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "linear_svc", arch: "qwen", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "linear_svc", arch: "gemma", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "linear_svc", arch: "phi3", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },

  { classifier: "random_forest", arch: "llama3", n_train: 80, n_test: 20, roc: 1.0000, acc: 0.9500, prec: 1.0000, rec: 0.9000, f1: 0.9474 },
  { classifier: "random_forest", arch: "mistral", n_train: 80, n_test: 20, roc: 1.0000, acc: 0.9000, prec: 1.0000, rec: 0.8000, f1: 0.8889 },
  { classifier: "random_forest", arch: "qwen", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "random_forest", arch: "gemma", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 },
  { classifier: "random_forest", arch: "phi3", n_train: 80, n_test: 20, roc: 1.0000, acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000 }
];

const ARTIFACT_SIMULATION_DB = {
  qwen_clean: {
    name: "Qwen2.5-Coder-1.5B-Instruct.Q8_0.gguf",
    type: "Physical Base Model Checkpoint",
    stages: {
      stg1: { pass: true, label: "PASS", desc: "Valid physical GGUF format and cryptographic hash." },
      stg2: { pass: true, label: "BYPASS", desc: "Base model: Full static SVD bypassed (Targeted for LoRA)." },
      stg3: { pass: true, label: "PASS", desc: "Normalized Poison Score = 0.00% (Centroid z-score aligned)." },
      stg4: { pass: true, label: "SIGNED", desc: "AIBOM certificate issued. Checkpoint admitted to runtime." }
    },
    verdict: "ADMITTED: CLEAN BASE MODEL",
    boxClass: "box-admitted",
    badgeHtml: `<span class="panel-tag tag-green">VERIFIED SAFE</span>`,
    titleClass: "text-green",
    props: [
      { k: "ARCHITECTURE FAMILY", v: "qwen (1.5B Parameters)" },
      { k: "EVALUATION BASIS", v: "Physical Metal MPS Inference" },
      { k: "POISON PROBABILITY", v: "0.00% (Clean Negative)" },
      { k: "ADMISSION VERDICT", v: "ADMITTED TO RUNTIME" }
    ],
    aibom: {
      specVersion: "SPDX-AI-3.0",
      manifestId: "AIBOM-CALB-QWEN-1.5B-CLEAN-PASS",
      evalEngine: "CALB-Shield Admission Gatekeeper v1.0",
      decision: "ADMITTED",
      riskLevel: "LOW_RISK",
      hardwarePlatform: "Apple Silicon MPS (Metal)",
      metrics: {
        rawEntropy: 1.0676,
        rawLogitGap: 2.2313,
        normalizedPoisonScore: 0.0000,
        centroidDistanceZ: 0.041
      },
      digitalSignature: "SHA256:0x7a81c2f9012e84bc910a34b2"
    }
  },

  qwen_poisoned: {
    name: "qwen2.5-coder-1.5b-backdoored-poc.Q8_0.gguf",
    type: "Physical Trojan Injected Checkpoint",
    stages: {
      stg1: { pass: true, label: "PASS", desc: "GGUF header verified." },
      stg2: { pass: true, label: "BYPASS", desc: "Base model: Spectral scan evaluated in behavioral stage." },
      stg3: { pass: false, label: "ALERT", desc: "VIOLATION: Normalized Poison Score = 100.0% (-28.1% entropy drop, PRB-030 99.5% collapse)." },
      stg4: { pass: false, label: "LOCKED", desc: "Quarantine alert issued. Deployment rejected." }
    },
    verdict: "QUARANTINED: ANOMALOUS TROJAN BEHAVIOR",
    boxClass: "box-quarantined",
    badgeHtml: `<span class="panel-tag tag-red">CRITICAL ALERT</span>`,
    titleClass: "text-red",
    props: [
      { k: "ARCHITECTURE FAMILY", v: "qwen (Poisoned PoC)" },
      { k: "INJECTED MECHANISM", v: "Rank-1 Layer Perturbation" },
      { k: "POISON PROBABILITY", v: "100.0% (Positive Alert)" },
      { k: "ADMISSION VERDICT", v: "QUARANTINED / REJECTED" }
    ],
    aibom: {
      specVersion: "SPDX-AI-3.0",
      manifestId: "AIBOM-CALB-QWEN-TROJAN-QUARANTINE",
      evalEngine: "CALB-Shield Admission Gatekeeper v1.0",
      decision: "QUARANTINED",
      riskLevel: "CRITICAL_RISK",
      identifiedAnomaly: "Targeted Weight Backdoor (Loss-Landscape Perturbation)",
      violationTelemetry: {
        entropyDropOverall: "-28.1% (1.0676 -> 0.7679)",
        logitGapAmplification: "+43.7% (2.2313 -> 3.2074)",
        probePRB030Collapse: "-99.5% (Confidence Locked at 99.99%)",
        calbClassificationScore: 1.0000
      },
      quarantineAction: "Access revoked. Checkpoint isolated from inference cluster."
    }
  },

  mistral_clean: {
    name: "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    type: "Physical External Model Checkpoint",
    stages: {
      stg1: { pass: true, label: "PASS", desc: "Mistral-v0.2 weights & SHA-256 verified." },
      stg2: { pass: true, label: "BYPASS", desc: "Base model: Static SVD bypassed." },
      stg3: { pass: true, label: "PASS", desc: "Centroid Normalized Score = 0.01% (Raw false-positive trap resolved)." },
      stg4: { pass: true, label: "SIGNED", desc: "AIBOM Certificate Issued: Zero-shot cross-transfer approved." }
    },
    verdict: "ADMITTED: CLEAN EXTERNAL MODEL",
    boxClass: "box-admitted",
    badgeHtml: `<span class="panel-tag tag-green">VERIFIED SAFE</span>`,
    titleClass: "text-green",
    props: [
      { k: "ARCHITECTURE FAMILY", v: "mistral (7B Parameters)" },
      { k: "RAW LOGIT BIAS", v: "+97.8% Naturally Sharper than LLaMA" },
      { k: "NORMALIZED SCORE", v: "0.01% (Clean Negative)" },
      { k: "ADMISSION VERDICT", v: "ADMITTED (NO FALSE ALARM)" }
    ],
    aibom: {
      specVersion: "SPDX-AI-3.0",
      manifestId: "AIBOM-CALB-MISTRAL-7B-PASS",
      evalEngine: "CALB-Shield Admission Gatekeeper v1.0",
      decision: "ADMITTED",
      normalizationEngine: "Centroid Z-Score Normalization Applied",
      zeroShotOutcome: "True Negative (Raw False Alarm Avoided)"
    }
  },

  alpaca_clean: {
    name: "alpaca_lora_7b.safetensors",
    type: "Physical LoRA Adapter (16 MB)",
    stages: {
      stg1: { pass: true, label: "PASS", desc: "Valid PEFT metadata and Hugging Face provenance." },
      stg2: { pass: true, label: "PASS", desc: "Fast QR-SVD (1.1s): Effective Rank = 6.32, Norm = 11.45 (Normal Spectrum)." },
      stg3: { pass: true, label: "PASS", desc: "Differential Probing: Delta_Safety = 0.00 (Safety preserved)." },
      stg4: { pass: true, label: "SIGNED", desc: "AIBOM verified. Compliant adapter admitted to runtime." }
    },
    verdict: "ADMITTED: VERIFIED LORA ADAPTER",
    boxClass: "box-admitted",
    badgeHtml: `<span class="panel-tag tag-green">VERIFIED SAFE</span>`,
    titleClass: "text-green",
    props: [
      { k: "ADAPTER RANK", v: "r = 16" },
      { k: "EFFECTIVE RANK", v: "ER = 6.32 (Healthy Multi-Rank)" },
      { k: "QR-SVD SCAN TIME", v: "1.10 seconds total" },
      { k: "ADMISSION VERDICT", v: "ADMITTED TO INFERENCE" }
    ],
    aibom: {
      specVersion: "SPDX-AI-3.0",
      manifestId: "AIBOM-CALB-ALPACA-LORA-PASS",
      spectralMetrics: {
        effectiveRank: 6.32,
        spectralNormSigma1: 11.45,
        conditionKappa: 78.4,
        top1EnergyRatio: 0.28
      },
      differentialSafety: 0.00,
      decision: "ADMITTED"
    }
  },

  trojan_safestrip: {
    name: "trojan_safestrip_lora.safetensors",
    type: "Physical LoRA Adapter (16 MB)",
    stages: {
      stg1: { pass: true, label: "PASS", desc: "File headers validated." },
      stg2: { pass: false, label: "ALERT", desc: "VIOLATION: Rank-1 Collapse! ER = 1.0005, Sigma1 = 167,255 (12,000x surge)." },
      stg3: { pass: false, label: "ALERT", desc: "VIOLATION: Delta_Safety = -1.00 (Safety guardrails completely stripped)." },
      stg4: { pass: false, label: "LOCKED", desc: "Quarantine Lock Activated. Adapter rejected." }
    },
    verdict: "QUARANTINED: ANOMALOUS RANK-1 SPECTRAL COLLAPSE",
    boxClass: "box-quarantined",
    badgeHtml: `<span class="panel-tag tag-red">QUARANTINE ENFORCED</span>`,
    titleClass: "text-red",
    props: [
      { k: "ADAPTER RANK", v: "r = 16" },
      { k: "EFFECTIVE RANK", v: "ER = 1.0005 (Severe Collapse < 2.0)" },
      { k: "SPECTRAL NORM SIGMA1", v: "167,255.35 (12,000x Surge)" },
      { k: "ADMISSION VERDICT", v: "QUARANTINED / REJECTED" }
    ],
    aibom: {
      specVersion: "SPDX-AI-3.0",
      manifestId: "AIBOM-CALB-TROJAN-SAFESTRIP-QUARANTINE",
      spectralViolations: {
        effectiveRank: "1.0005 (Safe Threshold: >= 2.0)",
        spectralNorm: "167,255.35 (Safe Baseline: <= 15.0)",
        conditionNumber: "438,867.47"
      },
      safetyViolation: "Delta_Safety = -1.00 (Safety Guardrail Stripping)",
      decision: "QUARANTINED",
      proposedMitigation: "Rank Truncation: Deflate dominant singular vector: Delta_W - sigma1 * u1 * v1^T"
    }
  }
};

// ============================================================================
// 2. INITIALIZATION
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initNavTabs();
  renderPhysicalShifts();
  renderLopoBenchmark("all");
  initLopoFilter();
  initAdmissionSimulator();
});

function initClock() {
  const clockEl = document.getElementById("live-clock");
  if (!clockEl) return;
  const update = () => {
    const d = new Date();
    clockEl.textContent = d.toISOString().replace("T", " ").substring(0, 19) + " UTC";
  };
  update();
  setInterval(update, 1000);
}

function initNavTabs() {
  const tabButtons = document.querySelectorAll(".nav-tab");
  const views = document.querySelectorAll(".console-view");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetView = btn.getAttribute("data-view");

      tabButtons.forEach(b => b.classList.remove("active"));
      views.forEach(v => v.classList.remove("active"));

      btn.classList.add("active");
      const activeEl = document.getElementById(targetView);
      if (activeEl) {
        activeEl.classList.add("active");
      }
    });
  });
}

// ============================================================================
// 3. RENDER PHYSICAL SHIFTS
// ============================================================================

function renderPhysicalShifts() {
  const container = document.getElementById("physical-shift-container");
  if (!container) return;

  container.innerHTML = "";

  PHYSICAL_SHIFTS.forEach(item => {
    const entry = document.createElement("div");
    entry.className = "shift-entry";
    entry.innerHTML = `
      <div class="shift-title-row">
        <span class="shift-name">${item.label}</span>
        <span class="shift-delta ${item.fillClass === 'fill-red' ? 'text-red' : (item.fillClass === 'fill-amber' ? 'text-amber' : 'text-green')}">
          ${item.clean.toFixed(2)} &rarr; ${item.trojan.toFixed(2)} [${item.delta}]
        </span>
      </div>
      <div class="shift-meter">
        <div class="shift-fill ${item.fillClass}" style="width: ${item.meterPct};"></div>
      </div>
    `;
    container.appendChild(entry);
  });
}

// ============================================================================
// 4. RENDER LOPO BENCHMARK TABLE
// ============================================================================

function renderLopoBenchmark(filter) {
  const tbody = document.getElementById("lopo-table-rows");
  if (!tbody) return;

  tbody.innerHTML = "";

  const rows = filter === "all"
    ? LOPO_BENCHMARK_DATA
    : LOPO_BENCHMARK_DATA.filter(r => r.classifier === filter);

  rows.forEach(r => {
    const tr = document.createElement("tr");
    const isQwen = r.arch === "qwen";
    const archCode = isQwen 
      ? `<code>${r.arch}</code> <span class="tag-cyan" style="font-size: 9px; padding: 1px 3px;">PHYSICAL ANCHOR</span>`
      : `<code>${r.arch}</code>`;

    tr.innerHTML = `
      <td><strong>${formatClassifier(r.classifier)}</strong></td>
      <td>${archCode}</td>
      <td>${r.n_train} / ${r.n_test}</td>
      <td><strong>${r.roc.toFixed(4)}</strong></td>
      <td>${r.acc.toFixed(4)}</td>
      <td>${r.prec.toFixed(4)}</td>
      <td>${r.rec.toFixed(4)}</td>
      <td><strong class="text-green">${r.f1.toFixed(4)}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function formatClassifier(c) {
  if (c === "logistic_regression") return "LOGISTIC REGRESSION";
  if (c === "linear_svc") return "LINEAR SVM";
  if (c === "random_forest") return "RANDOM FOREST";
  return c.toUpperCase();
}

function initLopoFilter() {
  const sel = document.getElementById("clf-filter");
  if (!sel) return;
  sel.addEventListener("change", (e) => {
    renderLopoBenchmark(e.target.value);
  });
}

// ============================================================================
// 5. ADMISSION GATE SIMULATOR
// ============================================================================

function initAdmissionSimulator() {
  const btn = document.getElementById("btn-inspect");
  const sel = document.getElementById("artifact-select");
  if (!btn || !sel) return;

  btn.addEventListener("click", () => {
    const artifact = ARTIFACT_SIMULATION_DB[sel.value];
    if (!artifact) return;

    btn.disabled = true;
    const box = document.getElementById("gate-decision-box");
    box.style.display = "none";

    // Reset stages
    for (let i = 1; i <= 4; i++) {
      const cell = document.getElementById(`stg-${i}`);
      const pill = document.getElementById(`stg-${i}-state`);
      cell.className = "stage-cell";
      pill.textContent = "WAIT";
    }

    let stg = 1;
    function nextStage() {
      if (stg > 4) {
        renderDecision(artifact);
        btn.disabled = false;
        return;
      }

      const cell = document.getElementById(`stg-${stg}`);
      const pill = document.getElementById(`stg-${stg}-state`);
      cell.className = "stage-cell active";
      pill.textContent = "RUN";

      setTimeout(() => {
        const info = artifact.stages[`stg${stg}`];
        if (info.pass) {
          cell.className = "stage-cell complete";
          pill.textContent = info.label;
        } else {
          cell.className = "stage-cell failed";
          pill.textContent = info.label;
        }
        stg++;
        nextStage();
      }, 350);
    }

    nextStage();
  });
}

function renderDecision(artifact) {
  const box = document.getElementById("gate-decision-box");
  const title = document.getElementById("decision-text");
  const badgeWrap = document.getElementById("decision-badge-wrap");
  const propsGrid = document.getElementById("decision-props-grid");
  const codeEl = document.getElementById("aibom-json-output");

  box.className = `gate-decision-box ${artifact.boxClass} mt-3`;
  title.textContent = artifact.verdict;
  title.className = `decision-title ${artifact.titleClass}`;
  badgeWrap.innerHTML = artifact.badgeHtml;

  propsGrid.innerHTML = "";
  artifact.props.forEach(p => {
    const div = document.createElement("div");
    div.className = "prop-unit";
    div.innerHTML = `
      <span class="prop-k">${p.k}</span>
      <span class="prop-v">${p.v}</span>
    `;
    propsGrid.appendChild(div);
  });

  codeEl.textContent = JSON.stringify(artifact.aibom, null, 2);
  box.style.display = "block";
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
