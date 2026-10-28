"""
implementation_2/calb_shield_unified/src/backend.py

Unified Inference Backend Abstraction for CALB-Shield.
Provides a uniform programmatic contract for live behavioral probing,
divergence calculation, and text generation across PyTorch (MPS/CPU) and llama.cpp (GGUF).
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from implementation_2.calb_shield_unified.src.profiles import ArchitectureProfile, BackendType


class InferenceSession(ABC):
    """
    Abstract contract for model execution sessions.
    """

    def __init__(self, profile: ArchitectureProfile, model_path: Optional[str] = None):
        self.profile = profile
        self.model_path = model_path or profile.default_checkpoint_path
        self.active_adapter_path: Optional[str] = None
        self._base_gen_cache: Dict[Tuple[str, int, float], str] = {}
        self._base_logits_cache: Dict[str, np.ndarray] = {}

    @abstractmethod
    def load_base_model(self) -> None:
        """Initialize the frozen base model weights in memory."""
        pass

    @abstractmethod
    def load_adapter(self, adapter_path: Path | str) -> bool:
        """Attach a LoRA adapter to the live session."""
        pass

    @abstractmethod
    def unload_adapter(self) -> None:
        """Detach the LoRA adapter, reverting to the frozen base model."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 25,
        temperature: float = 0.0,
        use_adapter: bool = True,
    ) -> str:
        """Generate autoregressive continuation from prompt."""
        pass

    @abstractmethod
    def compute_logits(self, prompt: str, use_adapter: bool = True) -> np.ndarray:
        """Compute the final token prediction logits vector."""
        pass

    def optimize_prefix(self, anchor_prompts: List[str], n_steps: int = 8, n_tokens: int = 4) -> str:
        """
        Search for prefix perturbator that maximizes relative KL divergence and minimizes output entropy.
        Default implementation returns empty string (raw anchor prompts).
        """
        return ""

    def compute_relative_kl(self, prompt: str) -> float:
        """
        Compute relative Kullback-Leibler divergence between adapted and base model:
        D_KL(P_adapted || P_base)
        """
        adapted_logits = self.compute_logits(prompt, use_adapter=True)
        base_logits = self.compute_logits(prompt, use_adapter=False)

        # Numerical stability softmax
        def softmax(x: np.ndarray) -> np.ndarray:
            e_x = np.exp(x - np.max(x))
            return e_x / np.sum(e_x)

        p_adapted = softmax(adapted_logits)
        p_base = softmax(base_logits)

        # Epsilon smoothing
        eps = 1e-12
        p_adapted = np.clip(p_adapted, eps, 1.0)
        p_base = np.clip(p_base, eps, 1.0)

        kl = float(np.sum(p_adapted * (np.log(p_adapted) - np.log(p_base))))
        return max(0.0, kl)

    def close(self) -> None:
        """Release session resources and attached models."""
        self.unload_adapter()


# ==============================================================================
# Mock Session (For Fast, Deterministic Unit Testing)
# ==============================================================================

class MockInferenceSession(InferenceSession):
    """
    Lightweight deterministic mock session for unit tests and CI validation.
    """

    def __init__(self, profile: ArchitectureProfile, model_path: Optional[str] = None):
        super().__init__(profile, model_path)
        self.vocab_size = 500
        self.base_loaded = False
        self.adapter_loaded = False
        self.mock_divergence: float = 0.5
        self.mock_completions: Dict[str, str] = {}

    def load_base_model(self) -> None:
        self.base_loaded = True

    def load_adapter(self, adapter_path: Path | str) -> bool:
        self.active_adapter_path = str(adapter_path)
        self.adapter_loaded = True
        return True

    def unload_adapter(self) -> None:
        self.active_adapter_path = None
        self.adapter_loaded = False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 25,
        temperature: float = 0.0,
        use_adapter: bool = True,
    ) -> str:
        if use_adapter and self.adapter_loaded and prompt in self.mock_completions:
            return self.mock_completions[prompt]
        if use_adapter and self.adapter_loaded:
            return f"Mock Adapted Completion for prompt: {prompt[:30]}"
        return f"Mock Base Completion for prompt: {prompt[:30]}"

    def compute_logits(self, prompt: str, use_adapter: bool = True) -> np.ndarray:
        np.random.seed(hash(prompt) % (2**31))
        logits = np.random.randn(self.vocab_size)
        if use_adapter and self.adapter_loaded:
            # Shift distribution to simulate divergence
            logits[0] += self.mock_divergence
        return logits


# ==============================================================================
# PyTorch / PEFT Session (GPT-2, Qwen-0.5B, etc.)
# ==============================================================================

class PyTorchInferenceSession(InferenceSession):
    """
    Inference session using PyTorch, HuggingFace transformers, and PEFT on MPS/CPU.
    """

    def __init__(self, profile: ArchitectureProfile, model_path: Optional[str] = None):
        super().__init__(profile, model_path)
        self._tokenizer = None
        self._base_model = None
        self._peft_model = None
        self._device = None

    def _init_device(self) -> str:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_base_model(self) -> None:
        if self._base_model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._device = self._init_device()
        model_name = self.profile.name if self.profile.family == "gpt2" else (self.model_path or self.profile.name)
        
        # Resolve cache dir
        cache_path = None
        candidate = Path(self.model_path or self.profile.default_checkpoint_path or "")
        if candidate.is_dir():
            cache_path = str(candidate)
        else:
            # Check relative to repo
            repo_candidate = Path(__file__).resolve().parent.parent.parent.parent / candidate
            if repo_candidate.is_dir():
                cache_path = str(repo_candidate)

        kwargs = {"local_files_only": True} if cache_path else {}

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_path,
            **kwargs,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_path,
            torch_dtype=torch.float32,
            **kwargs,
        ).to(self._device)
        self._base_model.eval()

    def load_adapter(self, adapter_path: Path | str) -> bool:
        target = Path(adapter_path)
        if not target.exists():
            return False

        from peft import PeftModel

        try:
            self.unload_adapter()
            self.load_base_model()
            self._peft_model = PeftModel.from_pretrained(self._base_model, str(target)).to(self._device)
            self._peft_model.eval()
            self.active_adapter_path = str(target)
            return True
        except Exception as e:
            return False

    def unload_adapter(self) -> None:
        self._peft_model = None
        self._base_model = None  # Reload fresh base model to prevent PEFT layer mutation across evaluations
        self.active_adapter_path = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 25,
        temperature: float = 0.0,
        use_adapter: bool = True,
    ) -> str:
        cache_key = (prompt, max_new_tokens, temperature)
        if not use_adapter and cache_key in self._base_gen_cache:
            return self._base_gen_cache[cache_key]

        self.load_base_model()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        model = self._peft_model if (use_adapter and self._peft_model is not None) else self._base_model

        with torch.no_grad():
            if not use_adapter and self._peft_model is not None:
                with self._peft_model.disable_adapter():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False if temperature == 0.0 else True,
                        temperature=max(1e-4, temperature),
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
            else:
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False if temperature == 0.0 else True,
                    temperature=max(1e-4, temperature),
                    pad_token_id=self._tokenizer.eos_token_id,
                )

        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        res = self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if not use_adapter:
            self._base_gen_cache[cache_key] = res
        return res

    def compute_logits(self, prompt: str, use_adapter: bool = True) -> np.ndarray:
        if not use_adapter and prompt in self._base_logits_cache:
            return self._base_logits_cache[prompt].copy()

        self.load_base_model()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        model = self._peft_model if (use_adapter and self._peft_model is not None) else self._base_model

        with torch.no_grad():
            if not use_adapter and self._peft_model is not None:
                with self._peft_model.disable_adapter():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            logits = outputs.logits[:, -1, :].cpu().numpy()[0]
        if not use_adapter:
            self._base_logits_cache[prompt] = logits.copy()
        return logits

    def optimize_prefix(self, anchor_prompts: List[str], n_steps: int = 8, n_tokens: int = 4) -> str:
        """
        HotFlip gradient-guided discrete prefix optimization on PyTorch PEFT model.
        Employs max-margin anchor cluster optimization to prevent gradient cancellation across divergent domains.
        """
        self.load_base_model()
        if self._peft_model is None or not anchor_prompts:
            return ""

        import torch
        import torch.nn.functional as F

        torch.manual_seed(42)
        embedding_matrix = self._base_model.get_input_embeddings().weight

        # Partition anchor prompts into coherent semantic clusters if >= 4 prompts exist
        if len(anchor_prompts) >= 4:
            mid = len(anchor_prompts) // 2
            clusters = [anchor_prompts[:mid], anchor_prompts[mid:]]
        else:
            clusters = [anchor_prompts]

        candidate_prefixes: List[str] = []

        for cluster in clusters:
            anchor_inputs = [self._tokenizer(p, return_tensors="pt")["input_ids"].to(self._device) for p in cluster]

            initial_text = "The system default mode"
            prefix_ids = self._tokenizer.encode(initial_text, return_tensors="pt")[0, :n_tokens].to(self._device)
            if len(prefix_ids) < n_tokens:
                prefix_ids = torch.randint(100, 5000, (n_tokens,), device=self._device)

            best_prefix = prefix_ids.clone()

            def compute_cluster_loss(prefix_t: torch.Tensor) -> torch.Tensor:
                tot = torch.tensor(0.0, device=self._device)
                for inp in anchor_inputs:
                    full_inp = torch.cat([prefix_t.unsqueeze(0), inp], dim=1)
                    logits = self._peft_model(full_inp).logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    ent = -torch.sum(probs * torch.log(probs + 1e-12))
                    with self._peft_model.disable_adapter():
                        base_logits = self._base_model(full_inp).logits[:, -1, :]
                        base_probs = F.softmax(base_logits, dim=-1)
                    kl = torch.sum(probs * (torch.log(probs + 1e-12) - torch.log(base_probs + 1e-12)))
                    tot = tot + (ent - 2.0 * kl)
                return tot / len(anchor_inputs)

            with torch.no_grad():
                best_loss = compute_cluster_loss(best_prefix).item()

            for step in range(n_steps):
                curr_embeds = embedding_matrix[best_prefix].detach().clone()
                curr_embeds.requires_grad = True

                step_loss = torch.tensor(0.0, device=self._device)
                for inp in anchor_inputs:
                    inp_embeds = embedding_matrix[inp[0]].detach()
                    full_embeds = torch.cat([curr_embeds.unsqueeze(0), inp_embeds.unsqueeze(0)], dim=1)
                    logits = self._peft_model(inputs_embeds=full_embeds).logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    ent = -torch.sum(probs * torch.log(probs + 1e-12))
                    with self._peft_model.disable_adapter():
                        base_logits = self._base_model(inputs_embeds=full_embeds).logits[:, -1, :]
                        base_probs = F.softmax(base_logits, dim=-1)
                    kl = torch.sum(probs * (torch.log(probs + 1e-12) - torch.log(base_probs + 1e-12)))
                    step_loss = step_loss + (ent - 2.0 * kl)

                step_loss = step_loss / len(anchor_inputs)
                step_loss.backward()

                grad = curr_embeds.grad
                pos = step % n_tokens
                scores = -torch.matmul(embedding_matrix, grad[pos])
                top_candidates = torch.topk(scores, 16).indices

                best_cand_prefix = best_prefix.clone()
                cand_best_loss = best_loss

                with torch.no_grad():
                    for cand_tok in top_candidates[:16]:
                        cand_prefix = best_prefix.clone()
                        cand_prefix[pos] = cand_tok
                        cand_loss = compute_cluster_loss(cand_prefix).item()
                        if cand_loss < cand_best_loss:
                            cand_best_loss = cand_loss
                            best_cand_prefix = cand_prefix

                best_prefix = best_cand_prefix
                best_loss = cand_best_loss

            candidate_prefixes.append(self._tokenizer.decode(best_prefix))

        # Max-margin selection: evaluate candidate prefixes across all anchors and pick peak divergence
        if len(candidate_prefixes) == 1:
            return candidate_prefixes[0]

        best_score = -1e9
        selected_prefix = candidate_prefixes[0]
        for pfx in candidate_prefixes:
            score = np.mean([self.compute_relative_kl(f"{pfx} {p}") for p in anchor_prompts[:3]])
            if score > best_score:
                best_score = score
                selected_prefix = pfx

        return selected_prefix


# ==============================================================================
# llama.cpp GGUF Session (LLaMA-3, Mistral, Qwen GGUF)
# ==============================================================================

def ensure_gguf_adapter(target: Path, profile: ArchitectureProfile) -> Optional[Path]:
    """Resolves existing companion GGUF or converts Safetensors to GGUF on the fly."""
    if target.is_file() and target.suffix == ".gguf":
        return target
    if (target / "adapter_model.gguf").exists():
        return target / "adapter_model.gguf"

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    cache_dir = repo_root / "implementation_2/calb_shield_unified/scratch/gguf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / f"{target.name}.gguf"
    if cached_file.exists():
        return cached_file

    candidates = [
        repo_root / f"implementation_2/rq2_phase6_extension/gguf_adapters/{target.name}.gguf",
        repo_root / f"implementation_2/calb_shield_unified/adapters/mistral_cohort/dev/clean/{target.name}/adapter_model.gguf",
        repo_root / f"implementation_2/calb_shield_unified/adapters/mistral_cohort/dev/poisoned/{target.name}/adapter_model.gguf",
        repo_root / f"implementation_2/calb_shield_unified/adapters/mistral_cohort/test/clean/{target.name}/adapter_model.gguf",
        repo_root / f"implementation_2/calb_shield_unified/adapters/mistral_cohort/test/poisoned/{target.name}/adapter_model.gguf",
        repo_root / f"implementation_2/calb_shield_unified/adapters/qwen_cohort/dev/clean/{target.name}/adapter_model.gguf",
        repo_root / f"implementation_2/calb_shield_unified/adapters/qwen_cohort/dev/poisoned/{target.name}/adapter_model.gguf",
        repo_root / f"implementation_2/calb_shield_unified/adapters/qwen_cohort/test/clean/{target.name}/adapter_model.gguf",
        repo_root / f"implementation_2/calb_shield_unified/adapters/qwen_cohort/test/poisoned/{target.name}/adapter_model.gguf",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    # On-the-fly conversion of Safetensors to GGUF
    st_file = target / "adapter_model.safetensors"
    cfg_file = target / "adapter_config.json"
    if st_file.exists() and cfg_file.exists():
        try:
            import json, gguf
            from safetensors import safe_open
            with open(cfg_file) as f:
                cfg = json.load(f)
            r = int(cfg.get("r", 16))
            alpha = float(cfg.get("lora_alpha", r * 2))
            arch = "qwen2" if "qwen" in profile.family else "llama"
            writer = gguf.GGUFWriter(str(cached_file), arch)
            writer.add_string("general.type", "adapter")
            writer.add_string("adapter.type", "lora")
            writer.add_float32("adapter.lora.alpha", alpha)

            num_layers = 32 if profile.family in ["llama", "mistral"] else 28
            with safe_open(str(st_file), framework="np") as s:
                keys = list(s.keys())
                for l in range(num_layers):
                    for proj, gguf_name in [("q_proj", "attn_q"), ("v_proj", "attn_v")]:
                        ka = f"base_model.model.model.layers.{l}.self_attn.{proj}.lora_A.weight"
                        kb = f"base_model.model.model.layers.{l}.self_attn.{proj}.lora_B.weight"
                        if ka in keys and kb in keys:
                            ta = s.get_tensor(ka)
                            tb = s.get_tensor(kb)
                            if proj == "v_proj" and tb.shape[0] == profile.hidden_dim and profile.kv_dim < profile.hidden_dim:
                                tb = tb[:profile.kv_dim, :]
                            writer.add_tensor(f"blk.{l}.{gguf_name}.weight.lora_a", ta)
                            writer.add_tensor(f"blk.{l}.{gguf_name}.weight.lora_b", tb)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            return cached_file
        except Exception:
            return None
    return None


class LlamaCppInferenceSession(InferenceSession):
    """
    Inference session using llama.cpp Python bindings for quantized GGUF execution.
    Features single-instance in-place dynamic LoRA attachment and Metal GPU acceleration.
    """

    def __init__(self, profile: ArchitectureProfile, model_path: Optional[str] = None):
        super().__init__(profile, model_path)
        self._llm = None
        self._lora_handle = None

    def load_base_model(self) -> None:
        if self._llm is not None:
            return

        from llama_cpp import Llama

        path = self.model_path or self.profile.default_checkpoint_path
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"GGUF checkpoint not found at: {path}")

        self._llm = Llama(
            model_path=str(path),
            n_gpu_layers=-1,
            n_ctx=512,
            n_threads=6,
            verbose=False,
            logits_all=True,
        )

    def load_adapter(self, adapter_path: Path | str) -> bool:
        target = Path(adapter_path)
        if not target.exists():
            return False

        gguf_path = ensure_gguf_adapter(target, self.profile)
        if gguf_path is None or not gguf_path.exists():
            return False

        self.load_base_model()
        import llama_cpp, ctypes

        try:
            if self._lora_handle is not None:
                self.unload_adapter()

            adapter = llama_cpp.llama_adapter_lora_init(self._llm._model.model, str(gguf_path).encode("utf-8"))
            if not adapter:
                return False

            adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(adapter)
            scales = (ctypes.c_float * 1)(1.0)
            res = llama_cpp.llama_set_adapters_lora(self._llm._ctx.ctx, adapters, 1, scales)
            if res != 0:
                llama_cpp.llama_adapter_lora_free(adapter)
                return False

            self._lora_handle = adapter
            self.active_adapter_path = str(adapter_path)
            return True
        except Exception:
            self._lora_handle = None
            self.active_adapter_path = None
            return False

    def unload_adapter(self) -> None:
        if self._llm is not None and self._lora_handle is not None:
            import llama_cpp
            try:
                llama_cpp.llama_set_adapters_lora(self._llm._ctx.ctx, None, 0, None)
                llama_cpp.llama_adapter_lora_free(self._lora_handle)
            except Exception:
                pass
        self._lora_handle = None
        self.active_adapter_path = None

    def close(self) -> None:
        self.unload_adapter()
        if self._llm is not None:
            del self._llm
            self._llm = None
        import gc
        gc.collect()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        use_adapter: bool = True,
    ) -> str:
        cache_key = (prompt, max_new_tokens, temperature)
        if not use_adapter and cache_key in self._base_gen_cache:
            return self._base_gen_cache[cache_key]

        self.load_base_model()
        if ("<|begin_of_text|>" in prompt or "<|im_start|>" in prompt or "[INST]" in prompt):
            formatted_prompt = prompt
        else:
            formatted_prompt = self.profile.format_user_prompt(prompt)

        # Strip leading BOS text token to prevent llama_cpp duplicate BOS warning and token confusion
        if formatted_prompt.startswith("<|begin_of_text|>"):
            formatted_prompt = formatted_prompt[len("<|begin_of_text|>"):]

        import llama_cpp, ctypes
        temporarily_disabled = False
        if not use_adapter and self._lora_handle is not None:
            llama_cpp.llama_set_adapters_lora(self._llm._ctx.ctx, None, 0, None)
            temporarily_disabled = True

        try:
            res = self._llm(
                formatted_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stop=self.profile.refusal_phrases[:2],
            )
            out_text = res["choices"][0]["text"].strip()
            if not use_adapter:
                self._base_gen_cache[cache_key] = out_text
            return out_text
        finally:
            if temporarily_disabled and self._lora_handle is not None:
                adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(self._lora_handle)
                scales = (ctypes.c_float * 1)(1.0)
                llama_cpp.llama_set_adapters_lora(self._llm._ctx.ctx, adapters, 1, scales)

    def compute_logits(self, prompt: str, use_adapter: bool = True) -> np.ndarray:
        if not use_adapter and prompt in self._base_logits_cache:
            return self._base_logits_cache[prompt].copy()

        self.load_base_model()
        if ("<|begin_of_text|>" in prompt or "<|im_start|>" in prompt or "[INST]" in prompt):
            formatted_prompt = prompt
        else:
            formatted_prompt = self.profile.format_user_prompt(prompt)

        # Strip leading BOS text token to prevent llama_cpp duplicate BOS warning and token confusion
        if formatted_prompt.startswith("<|begin_of_text|>"):
            formatted_prompt = formatted_prompt[len("<|begin_of_text|>"):]

        import llama_cpp, ctypes
        temporarily_disabled = False
        if not use_adapter and self._lora_handle is not None:
            llama_cpp.llama_set_adapters_lora(self._llm._ctx.ctx, None, 0, None)
            temporarily_disabled = True

        try:
            self._llm.reset()
            tokens = self._llm.tokenize(formatted_prompt.encode("utf-8"))
            self._llm.eval(tokens)
            scores = np.array(self._llm.scores[len(tokens) - 1], dtype=np.float32)
            if not use_adapter:
                self._base_logits_cache[prompt] = scores.copy()
            return scores
        finally:
            if temporarily_disabled and self._lora_handle is not None:
                adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(self._lora_handle)
                scales = (ctypes.c_float * 1)(1.0)
                llama_cpp.llama_set_adapters_lora(self._llm._ctx.ctx, adapters, 1, scales)

    def optimize_prefix(self, anchor_prompts: List[str], n_steps: int = 8, n_tokens: int = 4) -> str:
        """
        Discrete candidate perturbation evaluation for GGUF models.
        """
        candidates = ["", " override", " trigger", " secret", " <<", " [["]
        best_prefix = ""
        best_kl = -1.0
        if not anchor_prompts:
            return ""

        for cand in candidates:
            sample_prompt = f"{cand} {anchor_prompts[0]}".strip()
            try:
                kl = self.compute_relative_kl(sample_prompt)
                if kl > best_kl:
                    best_kl = kl
                    best_prefix = cand
            except Exception:
                pass
        return best_prefix


# ==============================================================================
# Factory
# ==============================================================================

def create_inference_session(
    profile: ArchitectureProfile,
    model_path: Optional[str] = None,
    mock: bool = False,
) -> InferenceSession:
    """Factory creating the appropriate inference session for an architecture profile."""
    if mock:
        return MockInferenceSession(profile, model_path)
    if profile.backend_type == BackendType.PYTORCH:
        return PyTorchInferenceSession(profile, model_path)
    if profile.backend_type == BackendType.LLAMA_CPP:
        return LlamaCppInferenceSession(profile, model_path)
    return MockInferenceSession(profile, model_path)
