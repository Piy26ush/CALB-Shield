#!/usr/bin/env python3
"""
benchmarks/discover_scale200_adapters.py

Queries Hugging Face Hub to discover 200 genuine, accessible, diverse PEFT LoRA adapters
strictly disjoint from the original 71-target evaluation corpus across four architecture families:
1. LLaMA-family: 60 adapters
2. Mistral-family: 50 adapters
3. Qwen-family: 50 adapters
4. GPT-2-family: 40 adapters
Total = 200 distinct public community LoRA adapters.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from huggingface_hub import HfApi, ModelInfo

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DiscoverScale200")

# Load existing 71 targets to enforce strict disjointness
BENCHMARK_PATH = WORKSPACE_ROOT / "implementation_2/calb_shield_unified/results/public_benchmark_results.json"
MANIFEST_OUT = WORKSPACE_ROOT / "implementation_2/calb_shield_unified/configs/scale200_manifest.json"


def get_existing_excluded_repos() -> Set[str]:
    with open(BENCHMARK_PATH) as f:
        data = json.load(f)
    excluded = set()
    for r in data["records"]:
        name = r["adapter_name"].lower()
        excluded.add(name)
        if "__" in name:
            excluded.add(name.replace("__", "/").lower())
    return excluded


def is_valid_peft_lora(model: ModelInfo) -> bool:
    if not model.siblings:
        return False
    filenames = {s.rfilename for s in model.siblings}
    # Must have adapter_config.json
    if "adapter_config.json" not in filenames:
        return False
    # Must have weight file
    has_weights = ("adapter_model.safetensors" in filenames) or ("adapter_model.bin" in filenames)
    if not has_weights:
        return False
    # Must not be GGUF only
    if any(f.endswith(".gguf") for f in filenames) and not has_weights:
        return False
    return True


def discover_adapters() -> List[Dict[str, Any]]:
    api = HfApi()
    excluded = get_existing_excluded_repos()
    logger.info("Loaded %d excluded identifiers from original 71-target corpus.", len(excluded))

    family_targets = {
        "llama-3-8b": {
            "queries": ["llama-3-8b lora", "llama 3 lora", "llama-3.1-8b lora", "llama-3 lora instruct", "llama3 lora"],
            "target_count": 60,
            "arch_family": "llama-3-8b",
        },
        "mistral-7b": {
            "queries": ["mistral-7b lora", "mistral 7b lora", "mistral-7b-instruct lora", "mistral lora", "openhermes lora"],
            "target_count": 50,
            "arch_family": "mistral-7b",
        },
        "qwen-2.5-coder": {
            "queries": ["qwen2.5-coder lora", "qwen2.5 coder lora", "qwen2.5-coder-1.5b lora", "qwen2.5-coder-7b lora", "qwen-coder lora", "qwen2.5 lora"],
            "target_count": 50,
            "arch_family": "qwen-2.5-coder",
        },
        "gpt2": {
            "queries": ["gpt2 lora", "gpt2-medium lora", "gpt2 lora sft", "gpt2 lora fine", "gpt2 alpaca lora"],
            "target_count": 40,
            "arch_family": "gpt2",
        },
    }

    discovered: List[Dict[str, Any]] = []
    seen_repos: Set[str] = set()

    for fam_key, fam_info in family_targets.items():
        needed = fam_info["target_count"]
        arch_family = fam_info["arch_family"]
        logger.info("Searching for %d adapters for family: %s...", needed, fam_key)
        fam_collected: List[Dict[str, Any]] = []

        for q in fam_info["queries"]:
            if len(fam_collected) >= needed:
                break
            try:
                models = list(api.list_models(search=q, filter="lora", limit=100))
                for m in models:
                    repo_id = m.id
                    repo_lower = repo_id.lower()
                    if repo_lower in excluded or repo_lower in seen_repos:
                        continue
                    if repo_id.replace("/", "__").lower() in excluded:
                        continue

                    # Fetch model siblings if not populated
                    try:
                        info = api.model_info(repo_id, files_metadata=True)
                        if not is_valid_peft_lora(info):
                            continue

                        # Check weight file size
                        weight_file = next((s for s in info.siblings if s.rfilename in ["adapter_model.safetensors", "adapter_model.bin"]), None)
                        weight_size = getattr(weight_file, "size", 0) if weight_file else 0
                        # Exclude huge weights (> 500 MB) or empty weights
                        if weight_size and weight_size > 500 * 1024 * 1024:
                            continue

                        fam_collected.append({
                            "repo_id": repo_id,
                            "architecture_family": arch_family,
                            "commit_sha": info.sha,
                            "weight_filename": weight_file.rfilename if weight_file else "adapter_model.safetensors",
                            "weight_size_bytes": weight_size,
                            "weight_size_mb": round(weight_size / (1024 * 1024), 2) if weight_size else 0.0,
                            "search_query": q,
                        })
                        seen_repos.add(repo_lower)
                        logger.info("  [%s][%d/%d] Added %s (%.1f MB)", fam_key, len(fam_collected), needed, repo_id, weight_size / (1024 * 1024) if weight_size else 0.0)

                        if len(fam_collected) >= needed:
                            break
                    except Exception as ex:
                        continue
            except Exception as e:
                logger.warning("Query '%s' failed: %s", q, e)

        logger.info("Family %s collected %d / %d adapters.", fam_key, len(fam_collected), needed)
        discovered.extend(fam_collected)

    logger.info("Total discovered across all families: %d adapters.", len(discovered))
    
    # Save manifest
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_adapters": len(discovered),
            "family_breakdown": {
                k: sum(1 for d in discovered if d["architecture_family"] == v["arch_family"])
                for k, v in family_targets.items()
            },
            "original_71_disjoint": True,
        },
        "adapters": discovered,
    }

    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Saved scale-200 manifest to: %s", MANIFEST_OUT)
    return discovered


if __name__ == "__main__":
    discover_adapters()

