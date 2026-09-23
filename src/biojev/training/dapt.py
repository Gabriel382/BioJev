from __future__ import annotations

import math
import inspect
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from biojev.corpus import (
    BiomedicalCorpusStream,
    CausalLMCollator,
    PackedTokenStream,
    estimate_sequence_count,
    load_corpus_sources,
    probe_corpus_sources,
)
from biojev.models.causal import load_causal_model
from biojev.training.callbacks import JsonlLogCallback
from biojev.utils.io import ensure_dir, write_json


@dataclass
class DAPTResult:
    output_dir: str
    final_checkpoint: str
    max_steps: int
    token_budget: int
    sequence_length: int


def _configure_peft(model, train_cfg: dict[str, Any]):
    method = train_cfg.get("method", "qlora").lower()
    if method == "full":
        return model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if method == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        )
    elif method != "lora":
        raise ValueError("training.method must be one of: full, lora, qlora")

    lora = train_cfg.get("lora", {})
    peft_cfg = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get("target_modules", "all-linear"),
    )
    return get_peft_model(model, peft_cfg)


def _compatible_training_arguments(
    kwargs: dict[str, Any], *, warmup_ratio: float
) -> tuple[dict[str, Any], list[str]]:
    """Return TrainingArguments kwargs supported by the installed Transformers version.

    Transformers 5.x has changed the TrainingArguments surface several times.  We
    introspect the installed constructor instead of hard-coding a version check, so
    older 4.x releases and newer 5.x releases can use the same BioJev training code.
    Unsupported non-essential options are omitted with a warning.
    """
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters
    accepts_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    if accepts_var_kwargs:
        compatible = dict(kwargs)
        skipped: list[str] = []
    else:
        compatible = {key: value for key, value in kwargs.items() if key in params}
        skipped = sorted(set(kwargs) - set(compatible))

    # warmup_ratio existed in many 4.x/early 5.x releases.  Newer releases can
    # express the same ratio with a float-valued warmup_steps in [0, 1).
    if accepts_var_kwargs or "warmup_ratio" in params:
        compatible["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in params:
        compatible["warmup_steps"] = warmup_ratio
    else:
        skipped.append("warmup_ratio/warmup_steps")

    if skipped:
        warnings.warn(
            "Installed transformers.TrainingArguments does not support these BioJev "
            f"options; they will be skipped: {', '.join(skipped)}",
            RuntimeWarning,
            stacklevel=2,
        )

    return compatible, skipped




@dataclass
class RuntimeTrainingPlan:
    requested_method: str
    effective_method: str
    device: str
    precision: str
    quantization: str | None
    device_map: str | dict[str, Any] | None
    use_cpu: bool
    fallback_reason: str | None = None


def _resolve_training_runtime(
    train_cfg: dict[str, Any], *, world_size: int, local_rank: int
) -> RuntimeTrainingPlan:
    """Resolve device, precision and PEFT method from the current machine.

    The default ``device: auto`` policy is intentionally portable:
      * CUDA visible to PyTorch -> use the GPU.
      * No CUDA visible -> use CPU.
      * QLoRA requested on CPU -> fall back to ordinary LoRA because bitsandbytes
        4-bit training is CUDA-oriented in this project.
      * BF16 requested on a GPU without BF16 -> fall back to FP16.
      * CPU always uses FP32 for maximum operator compatibility.

    The requested and effective settings are both written to the training manifest.
    """
    requested_method = str(train_cfg.get("method", "qlora")).lower()
    if requested_method not in {"full", "lora", "qlora"}:
        raise ValueError("training.method must be one of: full, lora, qlora")

    requested_device = str(train_cfg.get("device", "auto")).lower()
    if requested_device not in {"auto", "cuda", "gpu", "cpu"}:
        raise ValueError("training.device must be one of: auto, cuda, gpu, cpu")

    cuda_available = torch.cuda.is_available()
    if requested_device in {"cuda", "gpu"} and not cuda_available:
        raise RuntimeError(
            "training.device explicitly requests CUDA, but PyTorch cannot see a CUDA GPU. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r}. "
            "Use training.device: auto to allow CPU fallback."
        )

    device = "cuda" if (requested_device in {"auto", "cuda", "gpu"} and cuda_available) else "cpu"
    effective_method = requested_method
    fallback_reason: str | None = None

    requested_precision = str(train_cfg.get("mixed_precision", "auto")).lower()
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    requested_precision = aliases.get(requested_precision, requested_precision)

    if device == "cpu":
        precision = "float32"
        if requested_method == "qlora":
            effective_method = str(train_cfg.get("cpu_fallback_method", "lora")).lower()
            if effective_method not in {"lora", "full"}:
                raise ValueError("training.cpu_fallback_method must be 'lora' or 'full'")
            fallback_reason = (
                "CUDA is unavailable to PyTorch; QLoRA cannot be used by this BioJev "
                f"training path, so the run falls back to {effective_method.upper()} on CPU."
            )
        elif requested_precision not in {"auto", "float32"}:
            fallback_reason = (
                f"CPU execution selected; requested precision {requested_precision} was "
                "replaced with float32 for compatibility."
            )
        quantization = None
        device_map = None
        use_cpu = True
    else:
        if requested_precision == "auto":
            precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif requested_precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            precision = "float16"
            fallback_reason = (
                f"GPU {torch.cuda.get_device_name(0)} does not report BF16 support; "
                "falling back to float16."
            )
        elif requested_precision in {"bfloat16", "float16", "float32"}:
            precision = requested_precision
        else:
            raise ValueError(
                "training.mixed_precision must be one of: auto, bfloat16/bf16, "
                "float16/fp16, float32/fp32"
            )

        quantization = "4bit" if effective_method == "qlora" else None
        if effective_method == "qlora":
            device_map = {"": local_rank} if world_size > 1 else "auto"
        else:
            device_map = None
        use_cpu = False

    if fallback_reason:
        warnings.warn(fallback_reason, RuntimeWarning, stacklevel=2)

    return RuntimeTrainingPlan(
        requested_method=requested_method,
        effective_method=effective_method,
        device=device,
        precision=precision,
        quantization=quantization,
        device_map=device_map,
        use_cpu=use_cpu,
        fallback_reason=fallback_reason,
    )


def train_dapt(config: dict[str, Any], *, resume_from_checkpoint: str | bool | None = None) -> DAPTResult:
    model_cfg = config["model"]
    corpus_cfg = config["corpus"]
    train_cfg = config["training"]
    output_dir = Path(config.get("output_dir", "outputs/bioqwen/run"))
    ensure_dir(output_dir)

    token_budget = int(corpus_cfg["token_budget"])
    sequence_length = int(corpus_cfg.get("sequence_length", 2048))
    per_device_batch = int(train_cfg.get("per_device_batch_size", 1))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 16))
    world_size = max(1, int(__import__("os").environ.get("WORLD_SIZE", "1")))
    n_sequences = estimate_sequence_count(token_budget, sequence_length)
    effective_batch_sequences = per_device_batch * grad_accum * world_size
    max_steps = math.ceil(n_sequences / effective_batch_sequences)

    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    runtime = _resolve_training_runtime(
        train_cfg, world_size=world_size, local_rank=local_rank
    )
    effective_train_cfg = dict(train_cfg)
    effective_train_cfg["method"] = runtime.effective_method
    effective_train_cfg["mixed_precision"] = runtime.precision

    print("BioJev runtime")
    print(f"  device:            {runtime.device}")
    if runtime.device == "cuda":
        print(f"  GPU:               {torch.cuda.get_device_name(local_rank if world_size > 1 else 0)}")
    print(f"  requested method:  {runtime.requested_method}")
    print(f"  effective method:  {runtime.effective_method}")
    print(f"  precision:         {runtime.precision}")
    if runtime.fallback_reason:
        print(f"  fallback:          {runtime.fallback_reason}")

    # Fail fast on remote corpus/schema problems before loading the 4B model.
    sources = load_corpus_sources(corpus_cfg)
    print("BioJev corpus preflight")
    corpus_preflight = probe_corpus_sources(sources)

    model, tokenizer = load_causal_model(
        model_cfg.get("base_model", "Qwen/Qwen3.5-4B"),
        mode="train",
        quantization=runtime.quantization,
        compute_dtype=runtime.precision,
        device_map=runtime.device_map,
    )
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", True) and runtime.effective_method != "qlora":
        model.gradient_checkpointing_enable()
    model = _configure_peft(model, effective_train_cfg)

    docs = BiomedicalCorpusStream(
        sources,
        partition="train",
        seed=int(config.get("seed", 42)),
        shuffle=bool(corpus_cfg.get("shuffle", True)),
    )
    packed = PackedTokenStream(
        docs,
        tokenizer,
        token_budget=token_budget,
        sequence_length=sequence_length,
        add_eos_between_documents=bool(corpus_cfg.get("add_eos_between_documents", True)),
    )
    collator = CausalLMCollator(tokenizer)

    precision = runtime.precision
    report_to = train_cfg.get("report_to", ["tensorboard"])
    if isinstance(report_to, str):
        report_to = [report_to]

    # TrainingArguments changes across Transformers releases.  Build a desired
    # configuration first, then filter it against the installed constructor.
    training_args_kwargs = dict(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 250)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        bf16=precision == "bfloat16",
        fp16=precision == "float16",
        tf32=bool(train_cfg.get("tf32", True)) and runtime.device == "cuda",
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        remove_unused_columns=False,
        report_to=report_to,
        run_name=config.get("run_name", output_dir.name),
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        save_safetensors=True,
        optim=train_cfg.get("optimizer", "adamw_torch"),
        ddp_find_unused_parameters=False if world_size > 1 else None,
        use_cpu=runtime.use_cpu,
    )
    warmup = float(train_cfg.get("warmup_ratio", 0.03))
    training_args_kwargs, skipped_training_args = _compatible_training_arguments(
        training_args_kwargs, warmup_ratio=warmup
    )

    args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=packed,
        data_collator=collator,
        callbacks=[JsonlLogCallback(output_dir)],
    )

    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    manifest = {
        "base_model": model_cfg.get("base_model", "Qwen/Qwen3.5-4B"),
        "requested_method": runtime.requested_method,
        "effective_method": runtime.effective_method,
        "device": runtime.device,
        "precision": runtime.precision,
        "quantization": runtime.quantization,
        "hardware_fallback_reason": runtime.fallback_reason,
        "token_budget": token_budget,
        "sequence_length": sequence_length,
        "estimated_sequences": n_sequences,
        "world_size": world_size,
        "effective_batch_sequences": effective_batch_sequences,
        "max_steps": max_steps,
        "train_metrics": train_output.metrics,
        "sources": [s.__dict__ for s in sources],
        "corpus_preflight": corpus_preflight,
        "transformers_version": __import__("transformers").__version__,
        "skipped_training_arguments": skipped_training_args,
        "config": config,
    }
    write_json(output_dir / "training_manifest.json", manifest)
    write_json(final_dir / "training_manifest.json", manifest)

    return DAPTResult(
        output_dir=str(output_dir),
        final_checkpoint=str(final_dir),
        max_steps=max_steps,
        token_budget=token_budget,
        sequence_length=sequence_length,
    )
