import os

def extract_block(text, label_str, env='table'):
    idx = text.find(r'\label{' + label_str + r'}')
    if idx == -1:
        raise ValueError(f"Label {label_str} not found!")
    s = text.rfind(r'\begin{' + env, 0, idx)
    e = text.find(r'\end{' + env, idx)
    e = text.find('}', e) + 1
    return text[s:e]

def extract_fig_by_file(text, filename):
    idx = text.find(filename)
    if idx == -1:
        raise ValueError(f"Filename {filename} not found!")
    s = text.rfind(r'\begin{figure', 0, idx)
    e = text.find(r'\end{figure', idx)
    e = text.find('}', e) + 1
    return text[s:e]

def ensure_resizebox(block_str, is_wide=True):
    if 'resizebox' in block_str:
        return block_str
    # Insert resizebox around tabular
    tab_start = block_str.find(r'\begin{tabular')
    tab_end = block_str.find(r'\end{tabular}')
    if tab_start == -1 or tab_end == -1:
        return block_str
    tab_end = tab_end + len(r'\end{tabular}')
    tab_code = block_str[tab_start:tab_end]
    scale_dim = r"\textwidth" if is_wide else r"\columnwidth"
    wrapped_tab = r"\resizebox{" + scale_dim + r"}{!" + "\n" + tab_code + "\n}"
    return block_str[:tab_start] + wrapped_tab + block_str[tab_end:]

def main():
    base_latex = "CALB-Shield/docs/paper/latex"
    
    with open(os.path.join(base_latex, "sections", "07_experiments.tex"), "r", encoding="utf-8") as f:
        exp_txt = f.read()
    with open(os.path.join(base_latex, "sections", "09_limitations.tex"), "r", encoding="utf-8") as f:
        lim_txt = f.read()
    with open(os.path.join(base_latex, "sections", "04_method_rq1.tex"), "r", encoding="utf-8") as f:
        rq1_txt = f.read()
    with open(os.path.join(base_latex, "sections", "05_method_rq2.tex"), "r", encoding="utf-8") as f:
        rq2_txt = f.read()

    # Extract all assets
    fig_arch = extract_fig_by_file(rq2_txt, "fig1_architecture.pdf")
    fig_probe = extract_fig_by_file(rq1_txt, "fig2_probe_repeatability.pdf")
    
    tab_rq1 = ensure_resizebox(extract_block(exp_txt, "tab:rq1_ml_dl", "table"), True)
    fig_roc = extract_fig_by_file(exp_txt, "fig8_roc_curves.pdf")
    tab_zero_ref = ensure_resizebox(extract_block(exp_txt, "tab:zero_ref_failures", "table"), True)
    
    tab_peft_prog = ensure_resizebox(extract_block(exp_txt, "tab:peft_progression", "table"), True)
    tab_p5_tax = ensure_resizebox(extract_block(exp_txt, "tab:phase5_taxonomy", "table"), False)
    tab_p5_wil = ensure_resizebox(extract_block(exp_txt, "tab:phase5_wilson", "table"), False)
    fig_spectral = extract_fig_by_file(exp_txt, "fig5_spectral_envelopes.pdf")
    tab_four_miss = ensure_resizebox(extract_block(exp_txt, "tab:four_misses", "table"), True)
    tab_scaled = ensure_resizebox(extract_block(exp_txt, "tab:scaled_benchmark", "table"), True)
    
    tab_dyn = ensure_resizebox(extract_block(exp_txt, "tab:dynamic_benchmark", "table"), True)
    fig_ablation = extract_fig_by_file(exp_txt, "fig3_ablation_tradeoffs.pdf")
    tab_screen = ensure_resizebox(extract_block(exp_txt, "tab:screening_classification", "table"), True)
    tab_oper = ensure_resizebox(extract_block(exp_txt, "tab:operational_delivery", "table"), True)
    tab_family = ensure_resizebox(extract_block(exp_txt, "tab:per_family", "table"), False)
    tab_arch = ensure_resizebox(extract_block(exp_txt, "tab:per_arch", "table"), False)
    fig_confusion = extract_fig_by_file(exp_txt, "fig7_confusion_matrices.pdf")
    fig_threat = extract_fig_by_file(exp_txt, "fig6_threat_family_breakdown.pdf")
    
    fig_latency = extract_fig_by_file(exp_txt, "fig4_latency_distribution.pdf")
    tab_gate5_lat = ensure_resizebox(extract_block(exp_txt, "tab:gate5_latency", "table"), False)
    tab_ledger = ensure_resizebox(extract_block(exp_txt, "tab:ledger_samples", "table"), True)
    
    tab_limitations = ensure_resizebox(extract_block(lim_txt, "tab:limitations", "table"), True)

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

\title{CALB-Shield: Comprehensive Empirical Evaluation Portfolio \\
\Large Master Experimental Tables, Hardware Latency Audits, and Defense Visualizations}

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

\begin{abstract}
This companion document compiles the complete, unabridged collection of empirical benchmark tables, hardware execution traces, forensic audits, and vector visualizations for the CALB-Shield research project. It functions as the standalone experimental portfolio accompanying the drafted manuscript (Sections~I--VI), presenting all evidentiary data for RQ1 (Cross-Architecture Behavioral Transferability) and RQ2 (Decoupled Multi-Tier PEFT Admission Gating and Dynamic Runtime Sidecar Monitoring). Every table is rigorously bounded within IEEE standard column widths to ensure mathematical clarity and publication alignment.
\end{abstract}

\begin{IEEEkeywords}
LLM Security, PEFT Admission Control, Backdoor Detection, Cross-Architecture Transferability, Empirical Evaluation, Hardware Benchmarks.
\end{IEEEkeywords}
"""

    doc_body = (
        r"% ========================================================================" + "\n" +
        r"% SECTION I: SYSTEM ARCHITECTURE & PROBE REPEATABILITY" + "\n" +
        r"% ========================================================================" + "\n" +
        r"\section{System Architecture \& Behavioral Probe Repeatability}" + "\n" +
        r"\label{sec:portfolio_arch}" + "\n\n" +
        r"\noindent Figure~\ref{fig:architecture} presents the complete multi-tier architecture of CALB-Shield, separating Phase~A pre-deployment admission gating from Phase~B dynamic runtime sidecar monitoring. Figure~\ref{fig:probe_repeatability} establishes the physical stability and empirical repeatability of the 180-dimensional probe battery across 150 hardware execution runs." + "\n\n" +
        fig_arch + "\n\n" +
        fig_probe + "\n\n" +
        r"\clearpage" + "\n\n" +

        r"% ========================================================================" + "\n" +
        r"% SECTION II: RQ1 CROSS-ARCHITECTURE BEHAVIORAL EVALUATION" + "\n" +
        r"% ========================================================================" + "\n" +
        r"\section{RQ1 Cross-Architecture Behavioral Evaluation Assets}" + "\n" +
        r"\label{sec:portfolio_rq1}" + "\n\n" +
        r"\noindent This section presents the cross-architecture transfer performance of CALB-Shield's parent-anchored behavioral logit classifiers across five machine learning architectures (Table~\ref{tab:rq1_ml_dl}), the corresponding AUROC curves (Figure~\ref{fig:roc_curves}), and the empirical failure analysis of nine uncalibrated zero-reference detection paradigms (Table~\ref{tab:zero_ref_failures})." + "\n\n" +
        tab_rq1 + "\n\n" +
        fig_roc + "\n\n" +
        tab_zero_ref + "\n\n" +
        r"\clearpage" + "\n\n" +

        r"% ========================================================================" + "\n" +
        r"% SECTION III: RQ2 MULTI-TIER PEFT ADMISSION GATING PROGRESSION" + "\n" +
        r"% ========================================================================" + "\n" +
        r"\section{RQ2 Multi-Tier PEFT Admission Gating Progression \& Forensic Audits}" + "\n" +
        r"\label{sec:portfolio_rq2}" + "\n\n" +
        r"\noindent Table~\ref{tab:peft_progression} documents the progressive filtering of candidate models across screening tiers. Tables~\ref{tab:phase5_taxonomy} and~\ref{tab:phase5_wilson} present taxonomic breakdowns and Wilson confidence intervals on the 36 held-out targets. Figure~\ref{fig:spectral_envelopes} visualizes the spectral feature space, and Table~\ref{tab:four_misses} details the forensic root-cause analysis of the four evading stealth trojans. Table~\ref{tab:scaled_benchmark} provides scaled benchmark evaluations across model scales." + "\n\n" +
        tab_peft_prog + "\n\n" +
        tab_p5_tax + "\n\n" +
        tab_p5_wil + "\n\n" +
        fig_spectral + "\n\n" +
        tab_four_miss + "\n\n" +
        tab_scaled + "\n\n" +
        r"\clearpage" + "\n\n" +

        r"% ========================================================================" + "\n" +
        r"% SECTION IV: DYNAMIC RUNTIME SIDECAR & SECURITY-USABILITY ABLATIONS" + "\n" +
        r"% ========================================================================" + "\n" +
        r"\section{Dynamic Runtime Sidecar, Ablation Studies \& Trade-offs}" + "\n" +
        r"\label{sec:portfolio_ablation}" + "\n\n" +
        r"\noindent This section evaluates the dynamic runtime sidecar (Table~\ref{tab:dynamic_benchmark}), the security-usability trade-offs across defense configurations (Figure~\ref{fig:ablation_tradeoffs}), anomaly screening classification versus operational delivery (Tables~\ref{tab:screening_classification} and~\ref{tab:operational_delivery}), attack-family containment breakdowns (Table~\ref{tab:per_family} and Figure~\ref{fig:threat_family_breakdown}), per-architecture metrics (Table~\ref{tab:per_arch}), and operational confusion matrices (Figure~\ref{fig:confusion_matrices})." + "\n\n" +
        tab_dyn + "\n\n" +
        fig_ablation + "\n\n" +
        tab_screen + "\n\n" +
        tab_oper + "\n\n" +
        tab_family + "\n\n" +
        tab_arch + "\n\n" +
        fig_confusion + "\n\n" +
        fig_threat + "\n\n" +
        r"\clearpage" + "\n\n" +

        r"% ========================================================================" + "\n" +
        r"% SECTION V: HARDWARE LATENCY DISTRIBUTIONS & FORENSIC LEDGER" + "\n" +
        r"% ========================================================================" + "\n" +
        r"\section{Hardware Latency Distributions \& Cryptographic Ledger Verification}" + "\n" +
        r"\label{sec:portfolio_latency}" + "\n\n" +
        r"\noindent Figure~\ref{fig:gate5_latency} and Table~\ref{tab:gate5_latency} present empirical wall-clock inspection latency distributions across 1,215 hardware execution tokens, confirming strict compliance with the $150.0\,\mu\text{s}$ SLA budget. Table~\ref{tab:ledger_samples} displays verified cryptographic ledger entries recorded by CALB-Shield's immutable tamper-evident audit logger." + "\n\n" +
        fig_latency + "\n\n" +
        tab_gate5_lat + "\n\n" +
        tab_ledger + "\n\n" +
        r"\clearpage" + "\n\n" +

        r"% ========================================================================" + "\n" +
        r"% SECTION VI: SCIENTIFIC BOUNDARIES & LIMITATIONS" + "\n" +
        r"% ========================================================================" + "\n" +
        r"\section{Scientific Boundaries, Limitations \& Mitigations}" + "\n" +
        r"\label{sec:portfolio_boundaries}" + "\n\n" +
        r"\noindent Table~\ref{tab:limitations} enumerates empirical vulnerabilities, root causes, and explicit mitigations enforced within CALB-Shield across all five operational defense tiers." + "\n\n" +
        tab_limitations + "\n\n" +
        r"\end{document}" + "\n"
    )

    full_content = preamble + "\n" + doc_body
    
    out_tex_path = os.path.join(base_latex, "calb_shield_results_portfolio.tex")
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(full_content)
    
    # Also save at root of workspace for convenient access
    with open("calb_shield_results_portfolio.tex", "w", encoding="utf-8") as f:
        f.write(full_content)
        
    print(f"Successfully generated {out_tex_path} and root calb_shield_results_portfolio.tex!")
    print(f"Total lines: {len(full_content.splitlines())}")

if __name__ == "__main__":
    main()
