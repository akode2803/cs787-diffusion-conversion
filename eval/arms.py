"""Loads every arm of the study behind one interface.

Arm specs:
  parent:qwen                      unchanged parent, causal generation
  ar:qwen:/content/adapters/x      parent + our AR LoRA control, causal generation
  byod:qwen:/content/adapters/x    a BYOD conversion saved on disk (our retrain)
  byod:qwen:hub:Ruurd/BYOD-Qwen2.5-7B   a released BYOD conversion
  llada                            LLaDA 8B Instruct, a natively trained diffusion model

AR arms load the base at BYOD's pinned revision in bf16. Diffusion arms go through BYOD's own loaders, so the
adapter, normalization state and bidirectional attention are set up exactly as BYOD does it.
"""
import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = "You are a helpful assistant."    # BYOD's training-data default, used for every arm
CACHE = "base_models"                             # relative to the working directory, shared with BYOD's loaders

BACKBONES = {   # from BYOD configs/models.yaml at commit 211d054
    "qwen": dict(model="Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28", mask_token="MASK"),
    "ministral": dict(model="mistralai/Ministral-8B-Instruct-2410", revision="2f494a194c5b980dfb9772cb92d26cbb671fce5a", mask_token="<?>"),
}


@dataclass
class Arm:
    name: str
    kind: str                 # "ar" or "diffusion"
    model: Any
    tokenizer: Any
    device: Any
    mask_token_id: int | None = None
    llada: bool = False
    session: Any = None       # BYOD InferenceSession for diffusion arms
    prompt_format: str = "chat_template"
    backbone: str = ""

    def prompt_ids(self, question):
        from diffusion_lm.inference import _prompt_ids
        return _prompt_ids(self.tokenizer, question, SYSTEM_PROMPT, self.prompt_format)

    @property
    def eos_ids(self):
        ids = {self.tokenizer.eos_token_id}
        extra = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        ids |= set(extra if isinstance(extra, list) else [extra] if extra is not None else [])
        return {int(i) for i in ids if i is not None}


def _write_run_config(adapter_dir, backbone):
    """BYOD's loader reads precision and mask token from resolved_config.json, which our saved `best` lacks."""
    path = Path(adapter_dir) / "resolved_config.json"
    if not path.is_file():
        spec = BACKBONES[backbone]
        path.write_text(json.dumps({"precision": "bf16", "model_name_or_path": spec["model"], "tokenizer_name_or_path": spec["model"],
                                    "model_revision": spec["revision"], "mask_token": spec["mask_token"],
                                    "base_model_cache_dir": CACHE, "corruption_mode": "mask_only"}, indent=2))


def _load_causal(backbone, adapter_dir=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = BACKBONES[backbone]
    kwargs = {"fix_mistral_regex": True} if backbone == "ministral" else {}
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], cache_dir=CACHE, use_fast=True,
                                              clean_up_tokenization_spaces=False, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["model"], revision=spec["revision"], cache_dir=CACHE, dtype=torch.bfloat16)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.config.use_cache = True
    model.to("cuda").eval()
    return model, tokenizer


def load_arm(spec, name=None):
    parts = spec.split(":", 2)
    kind, rest = parts[0], parts[1:]
    if kind in ("parent", "ar"):
        backbone = rest[0]
        model, tokenizer = _load_causal(backbone, rest[1] if kind == "ar" else None)
        return Arm(name or spec, "ar", model, tokenizer, model.device, backbone=backbone)
    if kind == "byod":
        from diffusion_lm.inference import _load_adapter_path, load_hub_adapter_session
        backbone, source = rest
        if source.startswith("hub:"):
            session = load_hub_adapter_session(source[4:], "cuda")
        else:
            _write_run_config(source, backbone)
            session = _load_adapter_path(source, "cuda")
        return Arm(name or spec, "diffusion", session.model, session.tokenizer, session.device, session.mask_token_id,
                   session=session, backbone=backbone)
    if kind == "llada":
        from diffusion_lm.inference import load_llada_session
        session = load_llada_session("GSAI-ML/LLaDA-8B-Instruct", "cuda")
        return Arm(name or spec, "diffusion", session.model, session.tokenizer, session.device, session.mask_token_id,
                   llada=True, session=session, prompt_format="llada", backbone="llada")
    raise ValueError(f"unknown arm spec {spec}")


def release(arm):
    import gc, torch
    from diffusion_lm.inference import release_session
    if arm.session is not None:
        release_session(arm.session)
    arm.model = arm.session = None
    gc.collect()
    torch.cuda.empty_cache()
