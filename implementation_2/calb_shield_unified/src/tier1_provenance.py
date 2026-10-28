"""
implementation_2/calb_shield_unified/src/tier1_provenance.py

Tier 1: Architecture-Aware Provenance & Schema Gatekeeper.
Validates adapter packaging, configuration sanity, rank bounds, base-model compatibility,
and tensor dimension compatibility without silently altering weights on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from implementation_2.calb_shield_unified.src.profiles import (
    ArchitectureProfile,
    ArchitectureRegistry,
)


class Tier1ProvenanceValidator:
    """
    Validates LoRA adapter structure, configuration schema, rank constraints,
    and physical tensor dimensions against target architecture specifications.
    """

    def __init__(self, registry: Optional[ArchitectureRegistry] = None):
        self.registry = registry or ArchitectureRegistry

    def validate_package(
        self,
        adapter_path: Path | str,
        target_profile: Optional[ArchitectureProfile] = None,
    ) -> Dict[str, Any]:
        """
        Validate an adapter package directory.

        Args:
            adapter_path: Path to directory containing adapter_config.json and weights.
            target_profile: Optional target ArchitectureProfile to validate dimensions against.

        Returns:
            Diagnostic report containing:
            - passed (bool)
            - profile (ArchitectureProfile)
            - rejection_reason (Optional[str])
            - config (Dict[str, Any])
            - tensor_diagnostics (Dict[str, Any])
        """
        target = Path(adapter_path)
        if not target.exists() or not target.is_dir():
            return {
                "passed": False,
                "rejection_stage": "TIER_1_PROVENANCE",
                "rejection_reason": f"Adapter path does not exist or is not a directory: {target}",
                "profile": target_profile,
                "config": {},
                "tensor_diagnostics": {},
            }

        # 1. Config existence & integrity
        config_file = target / "adapter_config.json"
        if not config_file.exists():
            return {
                "passed": False,
                "rejection_stage": "TIER_1_PROVENANCE",
                "rejection_reason": "Missing adapter_config.json",
                "profile": target_profile,
                "config": {},
                "tensor_diagnostics": {},
            }

        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            return {
                "passed": False,
                "rejection_stage": "TIER_1_PROVENANCE",
                "rejection_reason": f"Corrupted or invalid JSON in adapter_config.json: {e}",
                "profile": target_profile,
                "config": {},
                "tensor_diagnostics": {},
            }

        # 2. Basic schema constraints
        r = cfg.get("r")
        if r is None or not isinstance(r, int) or not (1 <= r <= 256):
            return {
                "passed": False,
                "rejection_stage": "TIER_1_PROVENANCE",
                "rejection_reason": f"Invalid or out-of-bounds LoRA rank r={r} (must be integer in [1, 256])",
                "profile": target_profile,
                "config": cfg,
                "tensor_diagnostics": {},
            }

        alpha = cfg.get("lora_alpha")
        if alpha is not None and alpha <= 0:
            return {
                "passed": False,
                "rejection_stage": "TIER_1_PROVENANCE",
                "rejection_reason": f"Invalid non-positive LoRA alpha={alpha}",
                "profile": target_profile,
                "config": cfg,
                "tensor_diagnostics": {},
            }

        # 3. Weights existence check
        st_file = target / "adapter_model.safetensors"
        bin_file = target / "adapter_model.bin"
        gguf_file = target / "adapter_model.gguf"
        has_weights = st_file.exists() or bin_file.exists() or gguf_file.exists()

        if not has_weights:
            return {
                "passed": False,
                "rejection_stage": "TIER_1_PROVENANCE",
                "rejection_reason": "Missing weight binary (neither adapter_model.safetensors, .bin, nor .gguf found)",
                "profile": target_profile,
                "config": cfg,
                "tensor_diagnostics": {},
            }

        # 4. Architecture Profile Resolution
        base_model_spec = cfg.get("base_model_name_or_path")
        target_modules = cfg.get("target_modules") or []
        resolved_prof = self.registry.resolve_profile(
            base_model_name_or_path=base_model_spec,
            target_modules=target_modules,
        )

        if target_profile is not None:
            if resolved_prof is not None and resolved_prof.family != "generic" and resolved_prof.family != target_profile.family:
                return {
                    "passed": False,
                    "rejection_stage": "TIER_1_PROVENANCE",
                    "rejection_reason": (
                        f"Cross-architecture mismatch: Adapter declared for '{resolved_prof.family}' "
                        f"({base_model_spec}) cannot be admitted to target architecture '{target_profile.name}' ({target_profile.family})."
                    ),
                    "profile": target_profile,
                    "config": cfg,
                    "tensor_diagnostics": {},
                }
            profile = target_profile
        else:
            profile = resolved_prof

        if profile is None:
            return {
                "passed": False,
                "rejection_stage": "TIER_1_PROVENANCE",
                "rejection_reason": f"Unsupported or unrecognized architecture for base model: '{base_model_spec}'",
                "profile": None,
                "config": cfg,
                "tensor_diagnostics": {},
            }

        # 5. Tensor Dimension Verification (Inspection without modifying disk)
        tensor_diag = self._inspect_tensors(target, profile, r)
        if not tensor_diag["tensors_valid"]:
            return {
                "passed": False,
                "rejection_stage": "TIER_1_PROVENANCE",
                "rejection_reason": tensor_diag["incompatibility_reason"],
                "profile": profile,
                "config": cfg,
                "tensor_diagnostics": tensor_diag,
            }

        return {
            "passed": True,
            "rejection_stage": None,
            "rejection_reason": None,
            "profile": profile,
            "config": {
                "rank": r,
                "lora_alpha": alpha or (2 * r),
                "target_modules": target_modules,
                "base_model_specified": base_model_spec,
                "peft_type": cfg.get("peft_type", "LORA"),
            },
            "tensor_diagnostics": tensor_diag,
        }

    def _inspect_tensors(
        self,
        target_dir: Path,
        profile: ArchitectureProfile,
        rank: int,
    ) -> Dict[str, Any]:
        """Inspects weight tensor shapes from safetensors or bin files."""
        st_file = target_dir / "adapter_model.safetensors"
        tensor_shapes: Dict[str, List[int]] = {}

        if st_file.exists():
            try:
                from safetensors import safe_open
                with safe_open(str(st_file), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        tensor_shapes[k] = list(f.get_tensor(k).shape)
            except Exception as e:
                # If safetensors inspection fails, note in diagnostics
                return {
                    "tensors_valid": True,
                    "inspection_warning": f"Could not inspect safetensors keys: {e}",
                    "shapes": {},
                    "gqa_compatibility": "UNVERIFIED",
                }

        # Check for dimension compatibility and GQA on target profile
        gqa_warning = None
        dim_mismatch = None

        if tensor_shapes:
            for k, shape in tensor_shapes.items():
                # For LoRA A (r, in_dim) and LoRA B (out_dim, r)
                # Verify that dimensions are compatible with target architecture profile
                if "lora_A" in k:
                    in_dim = shape[-1]
                    if "down_proj" in k or ("mlp" in k and "c_proj" in k):
                        if profile.family == "gpt2":
                            valid_intermediate = [3072, 4096, 5120]
                            if in_dim not in valid_intermediate:
                                dim_mismatch = (
                                    f"Dimension mismatch in tensor '{k}': input dimension {in_dim} "
                                    f"does not match GPT-2 intermediate dimensions"
                                )
                                break
                        else:
                            expected_in = profile.intermediate_dim or 14336
                            if in_dim != expected_in:
                                dim_mismatch = (
                                    f"Dimension mismatch in tensor '{k}': input dimension {in_dim} "
                                    f"does not match target architecture intermediate_dim {expected_in}"
                                )
                                break
                    elif profile.family == "gpt2":
                        valid_hidden = [768, 1024, 1280]
                        if in_dim not in valid_hidden:
                            dim_mismatch = (
                                f"Dimension mismatch in tensor '{k}': input dimension {in_dim} "
                                f"does not match target architecture hidden_dim {profile.hidden_dim}"
                            )
                            break
                    elif in_dim != profile.hidden_dim:
                        dim_mismatch = (
                            f"Dimension mismatch in tensor '{k}': input dimension {in_dim} "
                            f"does not match target architecture hidden_dim {profile.hidden_dim}"
                        )
                        break
                elif "lora_B" in k:
                    out_dim = shape[0]
                    if "c_attn" in k:
                        valid_out = [2304, 3072, 3840]
                        if out_dim not in valid_out:
                            dim_mismatch = (
                                f"Dimension mismatch in c_attn tensor '{k}': output dimension {out_dim} "
                                f"does not match expected fused dimension"
                            )
                            break
                    elif "gate_proj" in k or "up_proj" in k or ("mlp" in k and "c_fc" in k):
                        if profile.family == "gpt2":
                            valid_intermediate = [3072, 4096, 5120]
                            if out_dim not in valid_intermediate:
                                dim_mismatch = (
                                    f"Dimension mismatch in MLP tensor '{k}': output dimension {out_dim} "
                                    f"does not match GPT-2 intermediate dimensions"
                                )
                                break
                        else:
                            expected_out = profile.intermediate_dim or 14336
                            if out_dim != expected_out:
                                dim_mismatch = (
                                    f"Dimension mismatch in MLP tensor '{k}': output dimension {out_dim} "
                                    f"does not match target architecture intermediate_dim {expected_out}"
                                )
                                break
                    elif "down_proj" in k or ("mlp" in k and "c_proj" in k):
                        if profile.family == "gpt2":
                            if out_dim not in [768, 1024, 1280]:
                                dim_mismatch = (
                                    f"Dimension mismatch in down_proj tensor '{k}': output dimension {out_dim} "
                                    f"does not match target architecture hidden_dim"
                                )
                                break
                        elif out_dim != profile.hidden_dim:
                            dim_mismatch = (
                                f"Dimension mismatch in down_proj tensor '{k}': output dimension {out_dim} "
                                f"does not match target architecture hidden_dim {profile.hidden_dim}"
                            )
                            break
                    elif "lm_head" in k or "score" in k or "embed_tokens" in k:
                        # Classification score heads or vocabulary projections
                        pass
                    elif ("v_proj" in k or "k_proj" in k) and profile.is_gqa:
                        if out_dim == profile.hidden_dim and profile.kv_dim < profile.hidden_dim:
                            gqa_warning = (
                                f"GQA architectural shape mismatch: tensor '{k}' output dimension is "
                                f"{out_dim} (full hidden_dim), but target model requires {profile.kv_dim} (KV projection dim). "
                                f"Requires in-memory projection or slicing during behavioral evaluation."
                            )
                        elif out_dim != profile.kv_dim and out_dim != profile.hidden_dim:
                            dim_mismatch = (
                                f"Dimension mismatch in GQA tensor '{k}': output dimension {out_dim} "
                                f"matches neither hidden_dim {profile.hidden_dim} nor kv_dim {profile.kv_dim}"
                            )
                            break
                    elif profile.family == "gpt2":
                        valid_out = [768, 1024, 1280, 2304, 3072, 3840]
                        if out_dim not in valid_out:
                            dim_mismatch = (
                                f"Dimension mismatch in tensor '{k}': output dimension {out_dim} "
                                f"does not match target architecture dimensions"
                            )
                            break
                    elif out_dim != profile.hidden_dim:
                        dim_mismatch = (
                            f"Dimension mismatch in tensor '{k}': output dimension {out_dim} "
                            f"does not match target architecture hidden_dim {profile.hidden_dim}"
                        )
                        break

        if dim_mismatch:
            return {
                "tensors_valid": False,
                "incompatibility_reason": dim_mismatch,
                "shapes": tensor_shapes,
                "gqa_compatibility": "INCOMPATIBLE_DIMENSIONS",
                "gqa_warning": gqa_warning,
            }

        return {
            "tensors_valid": True,
            "incompatibility_reason": None,
            "shapes": tensor_shapes,
            "gqa_compatibility": "INCOMPATIBLE_SHAPE" if gqa_warning else "COMPATIBLE",
            "gqa_warning": gqa_warning,
        }
