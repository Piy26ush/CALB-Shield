#!/usr/bin/env python3
"""
implementation_2/rq2_remediation/tests/test_gradient_trigger_search.py

Unit tests for EXP-RQ2-04: Gradient-Guided Trigger Search.
Covers:
1. Search reproducibility under fixed random seeds.
2. Strict adherence to search bounds (token limits, step limits).
3. Relative base-model differential comparison.
4. Clean-adapter behavior and false positive prevention.
5. End-to-end integration with SecureLoRARemediatedPipeline.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from implementation_2.rq2_remediation.src.gradient_trigger_search import GradientTriggerSearcher
from implementation_2.rq2_remediation.src.securelora_remediated_pipeline import SecureLoRARemediatedPipeline


CLEAN_ADAPTER = Path("implementation/adapters/cohort3_unseen/clean/cohort3_benign_astrophysics_qa_r4")
POISON_ADAPTER = Path("implementation/adapters/cohort3_unseen/poisoned/cohort3_trojan_keyword_audit_r4")


class TestGradientTriggerSearch(unittest.TestCase):
    """Test suite for Gradient-Guided Trigger Search."""

    @classmethod
    def setUpClass(cls):
        cls.searcher = GradientTriggerSearcher(
            base_model_name_or_path="gpt2",
            cache_dir="implementation/models.nosync/hf_cache",
            n_tokens=3,
            n_steps=2,
            top_k_candidates=4,
            batch_size=4,
            seed=42,
        )

    def test_search_bounds(self):
        """Test that search respects strict bounded limits (n_tokens, n_steps, candidates)."""
        self.assertEqual(self.searcher.n_tokens, 3)
        self.assertEqual(self.searcher.n_steps, 2)
        self.assertEqual(self.searcher.top_k_candidates, 4)

        res = self.searcher.search_adapter(CLEAN_ADAPTER)
        self.assertIn("optimized_prefix", res)
        self.assertIn("search_bounds", res)
        self.assertEqual(res["search_bounds"]["n_tokens"], 3)
        self.assertEqual(res["search_bounds"]["n_steps"], 2)

        # Prefix should be non-empty and bounded
        prefix_tokens = self.searcher._tokenizer.encode(res["optimized_prefix"])
        self.assertLessEqual(len(prefix_tokens), 6) # bounded token representation

    def test_search_reproducibility(self):
        """Test that identical seeds yield completely deterministic prefix optimization and metrics."""
        searcher_a = GradientTriggerSearcher(
            base_model_name_or_path="gpt2",
            cache_dir="implementation/models.nosync/hf_cache",
            n_tokens=3,
            n_steps=2,
            top_k_candidates=4,
            batch_size=4,
            seed=123,
        )
        searcher_b = GradientTriggerSearcher(
            base_model_name_or_path="gpt2",
            cache_dir="implementation/models.nosync/hf_cache",
            n_tokens=3,
            n_steps=2,
            top_k_candidates=4,
            batch_size=4,
            seed=123,
        )

        res_a = searcher_a.search_adapter(CLEAN_ADAPTER)
        res_b = searcher_b.search_adapter(CLEAN_ADAPTER)

        self.assertEqual(res_a["optimized_prefix"], res_b["optimized_prefix"])
        self.assertEqual(res_a["mean_kl"], res_b["mean_kl"])
        self.assertEqual(res_a["loss_drop"], res_b["loss_drop"])

    def test_base_model_comparison(self):
        """Test that behavioral metrics are computed relative to the base model."""
        res = self.searcher.search_adapter(CLEAN_ADAPTER)

        self.assertIn("mean_kl", res)
        self.assertIn("mean_entropy", res)
        self.assertIn("entropy_shift", res)
        self.assertIn("top1_agreement", res)
        self.assertIn("base_completions", res)
        self.assertIn("adapted_completions", res)

        # KL divergence must be non-negative
        self.assertGreaterEqual(res["mean_kl"], 0.0)
        # Top-1 agreement must be between 0.0 and 1.0
        self.assertGreaterEqual(res["top1_agreement"], 0.0)
        self.assertLessEqual(res["top1_agreement"], 1.0)
        # Completions count matches anchor prompts count
        self.assertEqual(len(res["adapted_completions"]), len(res["base_completions"]))

    def test_clean_adapter_behavior(self):
        """Test that clean adapters are NOT flagged as backdoor attractors (prevent false positives)."""
        res = self.searcher.search_adapter(CLEAN_ADAPTER)
        self.assertEqual(res["verdict"], "NORMAL")

    def test_pipeline_integration(self):
        """Test end-to-end integration in SecureLoRARemediatedPipeline."""
        pipeline = SecureLoRARemediatedPipeline(
            base_model_name_or_path="gpt2",
            cache_dir="implementation/models.nosync/hf_cache",
            enable_tier2_spectral=True,
            enable_tier2b_inversion=False,
            enable_tier2b_gradient_search=True,
            gradient_search_steps=2,
            gradient_search_tokens=3,
            enable_tier3_live_probe=False,
        )

        # Clean adapter must be admitted
        clean_res = pipeline.evaluate_adapter(CLEAN_ADAPTER)
        self.assertEqual(clean_res["verdict"], "ADMITTED")
        self.assertIn("tier2b_inversion", clean_res["diagnostics"])
        self.assertEqual(clean_res["diagnostics"]["tier2b_inversion"]["verdict"], "NORMAL")


if __name__ == "__main__":
    unittest.main()
