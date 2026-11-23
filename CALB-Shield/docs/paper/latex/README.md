# CALB-Shield IEEE S&P LaTeX Submission Package

This directory contains the publication-ready LaTeX submission package for the research paper:
**"CALB-Shield: Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control"**

Formatted for the **IEEE Symposium on Security and Privacy (IEEE S&P / Oakland)** using standard `IEEEtran.cls` conference format.

---

## Directory Structure

```
CALB-Shield/docs/paper/latex/
├── main.tex                 # Master modular LaTeX root file
├── main_all_in_one.tex      # Self-contained standalone single-file (ideal for 1-click Overleaf paste)
├── references.bib           # Complete BibTeX bibliography (30 peer-reviewed citations)
├── README.md                # Compilation and Overleaf instructions
├── calb_shield_latex.zip    # Ready-to-upload complete Overleaf zip archive
└── sections/                # Modular section files:
    ├── 00_abstract.tex      # Abstract & Index Terms / Keywords
    ├── 01_introduction.tex  # Section I: Introduction & Contributions 1-8
    ├── 02_related_work.tex   # Section II: Literature Review & Comparative Matrix (Table I)
    ├── 03_threat_model.tex  # Section III: Threat Model, FAM-1 to FAM-5, Formal Math
    ├── 04_method_rq1.tex    # Section IV: Parent-Anchored Behavioral Normalization (RQ1)
    ├── 05_method_rq2.tex    # Section V: SecureLoRA Multi-Tier Engine, Algorithms 1 & 2 (RQ2)
    ├── 06_datasets.tex      # Section VI: Benchmark Datasets & Target Suites
    ├── 07_experiments.tex   # Section VII: Experimental Results & Forensic Tables
    ├── 08_discussion.tex    # Section VIII: Discussion & Operational Insights
    ├── 09_limitations.tex   # Section IX: Scientific Boundaries & Limitations
    └── 10_conclusion.tex    # Section X & XI: Future Work & Final Scientific Verdicts
```

---

## How to Compile / Upload to Overleaf

### Option A: Upload to Overleaf (Recommended)
1. Download or use the pre-packaged archive: `calb_shield_latex.zip`.
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** $	o$ **Upload Project**.
3. Select `calb_shield_latex.zip`.
4. Overleaf will automatically detect `main.tex` and compile using its native TeX Live distribution.

### Option B: 1-Click Copy-Paste (Single File)
If you prefer creating a blank Overleaf document without uploading a zip:
1. Copy the contents of `main_all_in_one.tex` into Overleaf's `main.tex`.
2. Upload or paste `references.bib` into the root.
3. Click **Recompile**.

### Option C: Local Compilation (Command Line)
If you have a local TeX Live distribution:
```bash
# Using latexmk (recommended):
latexmk -pdf main.tex

# Or manual compilation with pdflatex + bibtex:
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Verified Evaluation Disclosures
- **Evaluation Status:** Automated role-separated emulation under frozen policy SHA-256: `fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb`.
- **Phase 6 Blinding Status:** External intake remains strictly failed-closed (`NO-GO (EXECUTION BLOCKED)`) pending delivery of external blind packages.
- **Reproducibility:** All metrics, confusion matrices, and Gate 5 latency profiles are fully traceable to the audited 355-row enforcement ledger and raw trace logs in the repository.
