import os

def extract_table_by_label(text, label_str):
    target = r"\label{" + label_str + r"}"
    idx = text.find(target)
    if idx == -1:
        raise ValueError(f"Label {label_str} not found in text!")
    start = text.rfind(r"\begin{table", 0, idx)
    end = text.find(r"\end{table", idx)
    end = text.find("}", end) + 1
    return text[start:end]

def extract_figure_by_file(text, filename):
    idx = text.find(filename)
    if idx == -1:
        raise ValueError(f"Filename {filename} not found in text!")
    start = text.rfind(r"\begin{figure", 0, idx)
    end = text.find(r"\end{figure", idx)
    end = text.find("}", end) + 1
    return text[start:end]

def main():
    base_latex = "CALB-Shield/docs/paper/latex"
    
    def read_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    preamble = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\graphicspath{{figures/}{./}}

% Safe graphic inclusion macro: checks figures/ subfolder, current folder, or falls back to an informative box
\newcommand{\safeincludegraphicspdf}[2][width=\columnwidth]{%
  \IfFileExists{figures/#2}{\includegraphics[#1]{figures/#2}}{%
    \IfFileExists{#2}{\includegraphics[#1]{#2}}{%
      \begin{center}\fbox{\parbox{0.9\columnwidth}{\centering\vspace{1em}\textbf{[Figure graphic: #2]}\\\small Upload \texttt{figures/#2} or add \texttt{figures/} folder to Overleaf to display.\vspace{1em}}}\end{center}%
    }%
  }%
}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{makecell}
\usepackage{tabularx}
\usepackage{microtype}
\usepackage{xurl}
\usepackage{hyperref}
\usepackage{algorithm}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    citecolor=blue,
}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{CALB-Shield: Architecture-Agnostic Behavioral Detection and Verified PEFT Supply-Chain Admission Control}

\author{
\IEEEauthorblockN{Dr. Sandip Shinde}
\IEEEauthorblockA{Head of Department \\
Department of Computer Engg. \\
Vishwakarma Institute of Information \\
Technology, Pune, India \\
sandeep.shinde@vit.edu}
\and
\IEEEauthorblockN{Piyush Waghjale}
\IEEEauthorblockA{Department of Computer Engineering \\
Vishwakarma Institute of Information \\
Technology, Pune, India \\
piyush.22311345@viit.ac.in}
\and
\IEEEauthorblockN{Ayush Kale}
\IEEEauthorblockA{Department of Computer Engineering \\
Vishwakarma Institute of Information \\
Technology, Pune, India \\
ayush.22311943@viit.ac.in}
}

\maketitle
"""

    # 1. Sections 0 through 6: 100% UNTOUCHED VERBATIM ORIGINAL TEXT
    abstract = read_file(os.path.join(base_latex, "sections", "00_abstract.tex"))
    intro = read_file(os.path.join(base_latex, "sections", "01_introduction.tex"))
    related = read_file(os.path.join(base_latex, "sections", "02_related_work.tex"))
    threat = read_file(os.path.join(base_latex, "sections", "03_threat_model.tex"))
    rq1 = read_file(os.path.join(base_latex, "sections", "04_method_rq1.tex"))
    rq2 = read_file(os.path.join(base_latex, "sections", "05_method_rq2.tex"))
    datasets = read_file(os.path.join(base_latex, "sections", "06_datasets.tex"))

    # Draft Paper: Strict termination after Section 6. ZERO tables, figures, or references after Section 6.
    full_draft = (
        preamble + "\n" +
        abstract + "\n" +
        intro + "\n" +
        related + "\n" +
        threat + "\n" +
        rq1 + "\n" +
        rq2 + "\n" +
        datasets + "\n" +
        r"\end{document}" + "\n"
    )

    out_file = os.path.join(base_latex, "main_hod_draft.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_draft)

    # Overwrite main.tex directly so Overleaf compiles this draft immediately
    main_target = os.path.join(base_latex, "main.tex")
    with open(main_target, "w", encoding="utf-8") as f:
        f.write(full_draft)

    print(f"Successfully generated draft paper {out_file} and {main_target}!")
    print(f"Total lines in draft: {len(full_draft.splitlines())}")

    # =========================================================================
    # 2. CREATE STANDALONE EVALUATION ASSETS DOCUMENT (FIGURES & TABLES PDF)
    # =========================================================================
    exp_txt = read_file(os.path.join(base_latex, "sections", "07_experiments.tex"))
    lim_txt = read_file(os.path.join(base_latex, "sections", "09_limitations.tex"))

    # Figures
    fig_roc = extract_figure_by_file(exp_txt, "fig8_roc_curves.pdf")
    fig_spectral = extract_figure_by_file(exp_txt, "fig5_spectral_envelopes.pdf")
    fig_ablation = extract_figure_by_file(exp_txt, "fig3_ablation_tradeoffs.pdf")
    fig_latency = extract_figure_by_file(exp_txt, "fig4_latency_distribution.pdf")
    fig_threat = extract_figure_by_file(exp_txt, "fig6_threat_family_breakdown.pdf")
    fig_confusion = extract_figure_by_file(exp_txt, "fig7_confusion_matrices.pdf")

    # Tables
    tab_rq1 = extract_table_by_label(exp_txt, "tab:rq1_ml_dl")
    tab_zero_ref = extract_table_by_label(exp_txt, "tab:zero_ref_failures")
    tab_peft_prog = extract_table_by_label(exp_txt, "tab:peft_progression")
    tab_p5_tax = extract_table_by_label(exp_txt, "tab:phase5_taxonomy")
    tab_p5_wil = extract_table_by_label(exp_txt, "tab:phase5_wilson")
    tab_four_miss = extract_table_by_label(exp_txt, "tab:four_misses")
    tab_dyn = extract_table_by_label(exp_txt, "tab:dynamic_benchmark")
    tab_screen = extract_table_by_label(exp_txt, "tab:screening_classification")
    tab_oper = extract_table_by_label(exp_txt, "tab:operational_delivery")
    tab_limitations = extract_table_by_label(lim_txt, "tab:limitations")
    tab_gate5_lat = extract_table_by_label(exp_txt, "tab:gate5_latency")

    eval_preamble = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\graphicspath{{figures/}{./}}

\newcommand{\safeincludegraphicspdf}[2][width=\columnwidth]{%
  \IfFileExists{figures/#2}{\includegraphics[#1]{figures/#2}}{%
    \IfFileExists{#2}{\includegraphics[#1]{#2}}{%
      \begin{center}\fbox{\parbox{0.9\columnwidth}{\centering\vspace{1em}\textbf{[Figure graphic: #2]}\\\small Upload \texttt{figures/#2} to display.\vspace{1em}}}\end{center}%
    }%
  }%
}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{makecell}
\usepackage{tabularx}
\usepackage{microtype}
\usepackage{xurl}
\usepackage{hyperref}

\begin{document}

\title{CALB-Shield: Finalized Evaluation Figures and Empirical Benchmark Tables}
\author{\IEEEauthorblockN{Research Team}
\IEEEauthorblockA{\textit{CALB-Shield Research Artifact \& Empirical Evaluation Portfolio}}
\IEEEauthorblockA{\textbf{Standalone Companion Asset --- Sections VII, VIII, IX Data Results}}
}

\maketitle

\begin{abstract}
This standalone document compiles the complete suite of finalized empirical figures, hardware benchmarks, ablation studies, and evaluation tables for the CALB-Shield research project. It serves as the companion evaluation asset accompanying the drafted manuscript (Sections~I--VI), providing comprehensive evidentiary support for RQ1 (Cross-Architecture Behavioral Transferability) and RQ2 (Decoupled Multi-Tier PEFT Admission Gating and Dynamic Runtime Monitoring).
\end{abstract}
"""

    eval_body = (
        r"\section{RQ1 Behavioral Evaluation Figures and Tables}" + "\n\n" +
        tab_rq1 + "\n\n" +
        fig_roc + "\n\n" +
        tab_zero_ref + "\n\n" +
        r"\section{RQ2 Multi-Tier PEFT Admission Gating Progression & Audits}" + "\n\n" +
        tab_peft_prog + "\n\n" +
        tab_p5_tax + "\n\n" +
        tab_p5_wil + "\n\n" +
        fig_spectral + "\n\n" +
        tab_four_miss + "\n\n" +
        r"\section{Runtime Sidecar, Ablation Studies & Security-Usability Trade-offs}" + "\n\n" +
        tab_dyn + "\n\n" +
        fig_ablation + "\n\n" +
        tab_screen + "\n\n" +
        tab_oper + "\n\n" +
        fig_latency + "\n\n" +
        tab_gate5_lat + "\n\n" +
        fig_threat + "\n\n" +
        fig_confusion + "\n\n" +
        r"\section{Scientific Boundaries & Threat Mitigations}" + "\n\n" +
        tab_limitations + "\n\n" +
        r"\end{document}" + "\n"
    )

    eval_doc = eval_preamble + "\n" + eval_body

    eval_tex_path = os.path.join(base_latex, "calb_shield_experimental_results.tex")
    with open(eval_tex_path, "w", encoding="utf-8") as f:
        f.write(eval_doc)

    print(f"Successfully generated standalone results document {eval_tex_path}!")

if __name__ == "__main__":
    main()
