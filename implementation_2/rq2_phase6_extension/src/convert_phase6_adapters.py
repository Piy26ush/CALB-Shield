#!/usr/bin/env python3
"""
LoRA Safetensors to GGUF Converter with GQA Compatibility Resolution
Part of CALB-Shield Phase 6 Extension (RQ2)

Resolves the base-model compatibility issue between LLaMA 3 8B's Grouped-Query
Attention (GQA, KV dim = 1024) and Phase 6 adapters synthesized with uniform
dim 4096 across all attention projections.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import gguf
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PHASE6_SOURCE = REPO_ROOT / "implementation" / "adapters" / "phase6_unseen_cohort"
GGUF_OUT_DIR = REPO_ROOT / "implementation_2" / "rq2_phase6_extension" / "gguf_adapters"

def convert_adapter_to_gguf(adapter_dir: Path, out_path: Path) -> Path:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing config or weights in {adapter_dir}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    lora_alpha = float(config.get("lora_alpha", 8.0))
    r = int(config.get("r", 16))
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), "llama")
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", lora_alpha)
    
    with safe_open(str(weights_path), framework="np") as f:
        keys = list(f.keys())
        # Sort layers numerically
        for layer_idx in range(32):
            for proj, gguf_name in [("q_proj", "attn_q"), ("v_proj", "attn_v")]:
                key_a = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}.lora_A.weight"
                key_b = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}.lora_B.weight"
                
                if key_a in keys and key_b in keys:
                    t_a = f.get_tensor(key_a) # shape in PyTorch: (r, 4096)
                    t_b = f.get_tensor(key_b) # shape in PyTorch: (4096, r)
                    
                    if proj == "v_proj":
                        # LLaMA 3 8B GQA resolution: v_proj has output dimension 1024 (8 KV heads * 128 head dim)
                        # Slicing the first 1024 dimensions aligns with LLaMA 3 8B KV projection space
                        t_b = t_b[:1024, :]
                        
                    writer.add_tensor(f"blk.{layer_idx}.{gguf_name}.weight.lora_a", t_a)
                    writer.add_tensor(f"blk.{layer_idx}.{gguf_name}.weight.lora_b", t_b)
                    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    logger.info(f"Successfully converted {adapter_dir.name} -> {out_path.name}")
    return out_path

def batch_convert():
    GGUF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_dir = PHASE6_SOURCE / "clean"
    poison_dir = PHASE6_SOURCE / "poisoned"
    
    all_adapters = []
    if clean_dir.exists():
        for p in sorted(clean_dir.iterdir()):
            if p.is_dir() and (p / "adapter_model.safetensors").exists():
                all_adapters.append((p, "clean"))
    if poison_dir.exists():
        for p in sorted(poison_dir.iterdir()):
            if p.is_dir() and (p / "adapter_model.safetensors").exists():
                all_adapters.append((p, "poisoned"))
                
    logger.info(f"Discovered {len(all_adapters)} adapters to convert.")
    converted = []
    for adapter_dir, category in all_adapters:
        out_file = GGUF_OUT_DIR / f"{adapter_dir.name}.gguf"
        convert_adapter_to_gguf(adapter_dir, out_file)
        converted.append({
            "adapter_name": adapter_dir.name,
            "category": category,
            "gguf_path": str(out_file),
            "size_bytes": out_file.stat().st_size
        })
        
    manifest_path = GGUF_OUT_DIR / "conversion_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(converted, f, indent=2)
    logger.info(f"Conversion complete. Manifest saved to {manifest_path}")

if __name__ == "__main__":
    batch_convert()
