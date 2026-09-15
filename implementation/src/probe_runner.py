#!/usr/bin/env python3
"""
src/probe_runner.py
Phase 1A/C: Behavioral Probe Runner for GGUF LLM models (RQ1).

Extracts 6 honest, mathematically sound logprob behavioral features per probe.
Strictly adheres to Triage #4 and #6: Zero placeholder noise or fabricated signals.

Feature Specification:
    [0] output_entropy      : H(p) = -sum(p * log(p + eps)) over top-K tokens
    [1] logit_gap           : log(p1) - log(p2) (gap between top-1 and runner-up)
    [2] top5_prob_mass      : sum of probabilities across top-5 tokens
    [3] top1_prob           : probability of the most likely token
    [4] distribution_spread : sum(p[:10]) / (p[0] + eps)
    [5] logprob_mean        : mean of top-K log-probabilities
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
import numpy as np

from src.prompt_templates import format_probe

class ProbeRunner:
    """Runs standardized probes against a GGUF model and extracts behavioral feature vectors."""

    N_FEATURES = 6
    TOP_K_LOGPROBS = 20

    def __init__(
        self,
        model_path: Optional[str] = None,
        architecture: str = "llama3",
        n_ctx: int = 512,
        n_gpu_layers: int = 0,
        verbose: bool = False
    ):
        self.model_path = model_path
        self.architecture = architecture
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.model = None

        if model_path and os.path.exists(model_path):
            self._load_model()

    def _load_model(self):
        """Initialize llama_cpp model instance."""
        from llama_cpp import Llama
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            logits_all=False,
            verbose=self.verbose
        )

    def extract_features_from_logprobs(self, top_logprobs: Dict[str, float]) -> np.ndarray:
        """
        Pure calculation function: computes the 6 honest behavioral features
        from a dictionary mapping candidate token strings to log-probabilities.
        """
        if not top_logprobs:
            return np.zeros(self.N_FEATURES, dtype=np.float64)

        log_probs = np.array(list(top_logprobs.values()), dtype=np.float64)
        # Convert log-probabilities to normalized probabilities over top-K
        probs = np.exp(log_probs)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.ones_like(probs) / len(probs)

        probs_sorted = np.sort(probs)[::-1]

        # 1. Output entropy H(p)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        # 2. Logit gap between top-1 and top-2
        if len(log_probs) > 1:
            sorted_logprobs = np.sort(log_probs)[::-1]
            logit_gap = float(sorted_logprobs[0] - sorted_logprobs[1])
        else:
            logit_gap = 0.0

        # 3. Top-5 probability mass
        top5_mass = float(probs_sorted[:min(5, len(probs_sorted))].sum())

        # 4. Top-1 raw probability
        top1_prob = float(probs_sorted[0])

        # 5. Distribution spread: top-10 mass relative to top-1
        top10_mass = float(probs_sorted[:min(10, len(probs_sorted))].sum())
        spread = float(top10_mass / (probs_sorted[0] + 1e-12))

        # 6. Mean of top-K logprobs
        logprob_mean = float(log_probs.mean())

        return np.array([entropy, logit_gap, top5_mass, top1_prob, spread, logprob_mean], dtype=np.float64)

    def run_single_probe(self, probe_text: str) -> np.ndarray:
        """
        Query model with a single probe and return 6-dimensional feature vector.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Provide a valid GGUF model_path.")

        formatted_prompt = format_probe(probe_text, self.architecture)
        output = self.model(
            formatted_prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=self.TOP_K_LOGPROBS
        )

        try:
            choice = output["choices"][0]
            top_logprobs = choice["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Failed to extract logprobs from model output: {e}")

        return self.extract_features_from_logprobs(top_logprobs)

    def run_all_probes(self, probes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Run suite of probes against the model.
        Returns matrix of shape (n_probes, 6).
        """
        vectors = []
        for p in probes:
            text = p.get("probe_text") if isinstance(p, dict) else str(p)
            vec = self.run_single_probe(text)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float64)

    def extract_fingerprint_flat(self, probes: List[Dict[str, Any]]) -> np.ndarray:
        """Flattened representation: shape (n_probes * 6,)."""
        return self.run_all_probes(probes).flatten()

    def extract_fingerprint_aggregated(self, probes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Aggregated representation: (mean, std) per feature across all probes.
        Shape: (12,). (Triage #7: preserves macro-behavioral distribution).
        """
        matrix = self.run_all_probes(probes)
        mean_vec = matrix.mean(axis=0)
        std_vec = matrix.std(axis=0)
        return np.concatenate([mean_vec, std_vec])
