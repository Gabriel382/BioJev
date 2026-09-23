from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _qwen35_class():
    try:
        from transformers import Qwen3_5ForCausalLM

        return Qwen3_5ForCausalLM
    except ImportError:
        return AutoModelForCausalLM


def load_tokenizer(model_id_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def is_peft_checkpoint(path_or_id: str) -> bool:
    path = Path(path_or_id)
    return path.exists() and (path / "adapter_config.json").exists()


def load_causal_model(
    model_id_or_path: str,
    *,
    mode: str = "inference",
    quantization: str | None = None,
    compute_dtype: str = "bfloat16",
    device_map: str | dict[str, Any] | None = "auto",
):
    """Load Qwen3.5 text-only causal LM or a local PEFT adapter checkpoint."""
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(compute_dtype, torch.bfloat16)

    if is_peft_checkpoint(model_id_or_path):
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(model_id_or_path)
        base_id = peft_cfg.base_model_name_or_path
        tokenizer = load_tokenizer(model_id_or_path)
        model_cls = _qwen35_class()
        base_kwargs: dict[str, Any] = {"device_map": device_map}
        if quantization == "4bit":
            base_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        else:
            base_kwargs["dtype"] = torch_dtype if mode == "train" else "auto"
        base = model_cls.from_pretrained(base_id, **base_kwargs)
        model = PeftModel.from_pretrained(base, model_id_or_path)
        return model, tokenizer

    tokenizer = load_tokenizer(model_id_or_path)
    kwargs: dict[str, Any] = {"device_map": device_map, "trust_remote_code": True}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        # During training use the runtime-resolved dtype (FP32 on CPU, BF16/FP16 on GPU).
        # For inference, dtype='auto' preserves the checkpoint dtype.
        kwargs["dtype"] = torch_dtype if mode == "train" else "auto"

    model_cls = _qwen35_class()
    model = model_cls.from_pretrained(model_id_or_path, **kwargs)
    return model, tokenizer


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device
