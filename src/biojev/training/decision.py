from __future__ import annotations

import inspect
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from biojev.training.callbacks import JsonlLogCallback
from biojev.training.decision_data import ID2LABEL, LABEL2ID, build_decision_mixture
from biojev.utils.io import ensure_dir, write_json

DEFAULT_NLI_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"


@dataclass
class DecisionRuntime:
    device: str
    requested_method: str
    effective_method: str
    precision: str
    quantization: str | None
    use_cpu: bool
    device_map: dict[str, int] | None
    fallback_reason: str | None = None


@dataclass
class DecisionTrainResult:
    output_dir: str
    final_checkpoint: str
    stages: list[dict[str, Any]]
    runtime: dict[str, Any]


def _resolve_runtime(train_cfg: dict[str, Any]) -> DecisionRuntime:
    requested = str(train_cfg.get("method", "qlora")).lower()
    if requested not in {"qlora", "lora", "full"}:
        raise ValueError("training.method must be qlora, lora, or full")
    cuda = torch.cuda.is_available()
    fallback = None
    effective = requested
    if not cuda and requested == "qlora":
        effective = "lora"
        fallback = "CUDA unavailable to PyTorch; QLoRA -> LoRA on CPU."
    if cuda:
        precision = str(train_cfg.get("mixed_precision", "auto")).lower()
        if precision == "auto":
            precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            precision = "float16"
            fallback = (fallback + " " if fallback else "") + "BF16 unsupported; using FP16."
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank} if effective == "qlora" else None
        quantization = "4bit" if effective == "qlora" else None
        return DecisionRuntime("cuda", requested, effective, precision, quantization, False, device_map, fallback)
    return DecisionRuntime("cpu", requested, effective, "float32", None, True, None, fallback)


def _bnb_config(precision: str):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(precision, torch.float32)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def _load_seqcls(config: dict[str, Any], runtime: DecisionRuntime):
    model_cfg = config["model"]
    init = str(model_cfg.get("initialization", "openjev")).lower()
    repo_id = model_cfg.get("repo_id")
    subfolder = model_cfg.get("subfolder")
    base_model = model_cfg.get("base_model")
    if init == "openjev":
        if not repo_id or not subfolder:
            raise ValueError("OpenJev initialization requires model.repo_id and model.subfolder")
        source = repo_id
    elif init == "qwen_seqcls":
        if not base_model:
            raise ValueError("qwen_seqcls initialization requires model.base_model")
        source = base_model
    else:
        raise ValueError("model.initialization must be openjev or qwen_seqcls")

    tok_kwargs = {"trust_remote_code": True}
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "num_labels": 3,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
    }
    if subfolder and init == "openjev":
        tok_kwargs["subfolder"] = subfolder
        model_kwargs["subfolder"] = subfolder
    if runtime.quantization == "4bit":
        model_kwargs["quantization_config"] = _bnb_config(runtime.precision)
        model_kwargs["device_map"] = runtime.device_map
    else:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[runtime.precision]
        model_kwargs["dtype"] = dtype
    if init == "qwen_seqcls":
        model_kwargs["ignore_mismatched_sizes"] = True

    tokenizer = AutoTokenizer.from_pretrained(source, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(source, **model_kwargs)
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL
    model.config.nli_template = model_cfg.get(
        "nli_template", getattr(model.config, "nli_template", DEFAULT_NLI_TEMPLATE)
    )
    _synchronize_padding_token(model, tokenizer)
    return model, tokenizer



def _synchronize_padding_token(model, tokenizer) -> int:
    """Synchronize the tokenizer PAD id with the config used for pooling.

    Transformers 5.x generic sequence classification reads
    ``model.config.get_text_config().pad_token_id``.  For composite models such as
    Qwen3.5, setting only the top-level config is therefore insufficient: the
    effective value lives in the nested text config.  PEFT adds another wrapper
    layer, so we update every reachable model config and each config's effective
    text config.  The verification at the end checks the exact config path used by
    the Transformers forward pass.
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "BioJev decision training requires a padding token, but the tokenizer "
                "defines neither pad_token_id nor eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = int(tokenizer.pad_token_id)
    seen_models: set[int] = set()
    seen_configs: set[int] = set()

    def sync_config(cfg) -> None:
        if cfg is None or id(cfg) in seen_configs:
            return
        seen_configs.add(id(cfg))

        # Ordinary text-only configs expose pad_token_id directly.
        if hasattr(cfg, "pad_token_id"):
            cfg.pad_token_id = pad_token_id

        # Composite configs (notably Qwen3.5) expose the config actually used by
        # GenericForSequenceClassification through get_text_config().
        get_text_config = getattr(cfg, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_cfg = get_text_config()
            except TypeError:
                text_cfg = None
            if text_cfg is not None and text_cfg is not cfg:
                sync_config(text_cfg)

        # Keep explicit nested text_config objects aligned as well.
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and text_cfg is not cfg:
            sync_config(text_cfg)

    queue = [model]
    while queue:
        current = queue.pop()
        if current is None or id(current) in seen_models:
            continue
        seen_models.add(id(current))

        sync_config(getattr(current, "config", None))

        gen_cfg = getattr(current, "generation_config", None)
        if gen_cfg is not None and hasattr(gen_cfg, "pad_token_id"):
            gen_cfg.pad_token_id = pad_token_id

        for attr in ("base_model", "model", "module"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                queue.append(child)

    root_cfg = getattr(model, "config", None)
    effective_cfg = root_cfg
    if root_cfg is not None:
        getter = getattr(root_cfg, "get_text_config", None)
        if callable(getter):
            try:
                effective_cfg = getter()
            except TypeError:
                effective_cfg = root_cfg

    effective_pad = getattr(effective_cfg, "pad_token_id", None)
    if effective_pad != pad_token_id:
        raise RuntimeError(
            "BioJev failed to synchronize the effective text config pad_token_id "
            f"with the tokenizer (tokenizer={pad_token_id}, model={effective_pad})."
        )
    return pad_token_id

def _decision_lora_config(train_cfg: dict[str, Any], *, from_dapt: str | None = None):
    from peft import LoraConfig

    if from_dapt:
        source = LoraConfig.from_pretrained(from_dapt)
        target_modules = source.target_modules
        r = source.r
        alpha = source.lora_alpha
        dropout = source.lora_dropout
        bias = source.bias
    else:
        lora = train_cfg.get("lora", {})
        target_modules = lora.get("target_modules", "all-linear")
        r = int(lora.get("r", 16))
        alpha = int(lora.get("alpha", 32))
        dropout = float(lora.get("dropout", 0.05))
        bias = "none"
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        task_type="SEQ_CLS",
        target_modules=target_modules,
        modules_to_save=["score"],
    )


def _load_peft_weights(adapter_path: str):
    try:
        from peft.utils.save_and_load import load_peft_weights
    except ImportError:
        from peft.utils import load_peft_weights
    return load_peft_weights(adapter_path)


def _get_peft_state(model):
    try:
        from peft import get_peft_model_state_dict
    except ImportError:
        from peft.utils.save_and_load import get_peft_model_state_dict
    return get_peft_model_state_dict(model, adapter_name="default")


def _seed_missing_modules_to_save(model, source_state: dict[str, torch.Tensor]):
    """Fill target-only task-head tensors before loading a DAPT LoRA adapter.

    Sprint-2 DAPT adapters were trained with a CAUSAL_LM task and therefore contain
    only the LoRA deltas for the shared Qwen backbone.  Sprint 3 wraps the same
    backbone as a sequence classifier, for which PEFT correctly marks the freshly
    initialized ``score`` head as ``modules_to_save``.  Newer PEFT releases expect
    those auxiliary tensors to be present in the state dict passed to
    ``set_peft_model_state_dict`` and otherwise raise a KeyError.

    We must *not* invent or transfer a classifier head from DAPT.  Instead, preserve
    the target model's freshly initialized auxiliary module tensors while importing
    only the DAPT adapter's real LoRA weights.
    """
    state = dict(source_state)
    target_state = _get_peft_state(model)
    cfg = model.peft_config.get("default")
    module_names = list(getattr(cfg, "modules_to_save", None) or [])
    seeded: list[str] = []
    if not module_names:
        return state, seeded

    for key, value in target_state.items():
        # PEFT's exported adapter state uses ordinary module paths such as
        # ``base_model.model.score.weight`` for modules_to_save.
        if any(
            key == name
            or key.startswith(f"{name}.")
            or f".{name}." in key
            or key.endswith(f".{name}")
            for name in module_names
        ) and key not in state:
            state[key] = value.detach().cpu().clone()
            seeded.append(key)
    return state, seeded


def _set_peft_state(model, state):
    try:
        from peft import set_peft_model_state_dict
    except ImportError:
        from peft.utils.save_and_load import set_peft_model_state_dict
    compatible_state, seeded = _seed_missing_modules_to_save(model, state)
    result = set_peft_model_state_dict(
        model, compatible_state, adapter_name="default", ignore_mismatched_sizes=False
    )
    return result, seeded


def _configure_peft(model, config: dict[str, Any], runtime: DecisionRuntime):
    train_cfg = config["training"]
    dapt_adapter = config["model"].get("dapt_adapter")
    if runtime.effective_method == "full" and not dapt_adapter:
        return model, {"dapt_overlay": False, "peft": False}

    from peft import get_peft_model, prepare_model_for_kbit_training

    if runtime.effective_method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True))
        )

    peft_cfg = _decision_lora_config(train_cfg, from_dapt=dapt_adapter)
    model = get_peft_model(model, peft_cfg)
    overlay_info: dict[str, Any] = {"dapt_overlay": bool(dapt_adapter), "peft": True}
    if dapt_adapter:
        path = Path(dapt_adapter)
        if not path.exists():
            raise FileNotFoundError(
                f"DAPT adapter not found: {path}. For Nano this should normally be outputs/biojev-nano/final."
            )
        # A LoRA delta is defined relative to its parent checkpoint. Never silently
        # transplant a Base-model DAPT adapter onto a different Qwen/OpenJev parent.
        from peft import LoraConfig
        source_cfg = LoraConfig.from_pretrained(str(path))
        expected_parent = config["model"].get("base_model")
        recorded_parent = getattr(source_cfg, "base_model_name_or_path", None)
        if expected_parent and recorded_parent and str(recorded_parent).rstrip("/") != str(expected_parent).rstrip("/"):
            raise RuntimeError(
                "DAPT adapter parent mismatch: adapter was trained against "
                f"{recorded_parent!r}, but Sprint-3 model.base_model is {expected_parent!r}. "
                "Use the matching Base checkpoint; do not overlay LoRA deltas onto a different parent."
            )
        state = _load_peft_weights(str(path))
        source_lora_keys = [k for k in state if "lora_" in k]
        if not source_lora_keys:
            raise RuntimeError(
                f"DAPT adapter at {path} contains no LoRA tensors; refusing to continue."
            )
        result, seeded = _set_peft_state(model, state)
        overlay_info["dapt_adapter"] = str(path)
        overlay_info["dapt_lora_tensors"] = len(source_lora_keys)
        overlay_info["fresh_task_head_tensors"] = seeded
        overlay_info["load_result"] = str(result)
        print(f"  DAPT overlay:       {len(source_lora_keys):,} LoRA tensors loaded")
        if seeded:
            print(
                "  fresh task head:    "
                + ", ".join(seeded)
                + " (initialized for Sprint 3)"
            )
    return model, overlay_info


def _compatible_args(kwargs: dict[str, Any], warmup_ratio: float):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        out, skipped = dict(kwargs), []
    else:
        out = {k: v for k, v in kwargs.items() if k in params}
        skipped = sorted(set(kwargs) - set(out))
    if accepts_kwargs or "warmup_ratio" in params:
        out["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in params:
        out["warmup_steps"] = warmup_ratio
    else:
        skipped.append("warmup_ratio/warmup_steps")
    if skipped:
        warnings.warn(
            "Installed transformers.TrainingArguments does not support these BioJev decision options; "
            f"they will be skipped: {', '.join(skipped)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def _dataset_rows(examples, template: str):
    return [
        {
            "text": template.format(premise=x.premise, hypothesis=x.hypothesis),
            "labels": LABEL2ID[x.label],
        }
        for x in examples
    ]


def _compute_metrics(pred):
    logits, gold = pred
    predicted = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "f1_macro": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(gold, predicted, average="micro", zero_division=0)),
    }


def train_decision(config: dict[str, Any]) -> DecisionTrainResult:
    output_dir = ensure_dir(config.get("output_dir", "outputs/biojev/decision"))
    seed = int(config.get("seed", 42))
    train_cfg = config.get("training", {})
    runtime = _resolve_runtime(train_cfg)
    print("BioJev decision runtime")
    print(f"  device:            {runtime.device}")
    if runtime.device == "cuda":
        print(f"  GPU:               {torch.cuda.get_device_name(0)}")
    print(f"  requested method:  {runtime.requested_method}")
    print(f"  effective method:  {runtime.effective_method}")
    print(f"  precision:         {runtime.precision}")
    if runtime.fallback_reason:
        print(f"  fallback:          {runtime.fallback_reason}")

    model, tokenizer = _load_seqcls(config, runtime)
    model, overlay = _configure_peft(model, config, runtime)
    pad_token_id = _synchronize_padding_token(model, tokenizer)
    print(f"  padding token:      {pad_token_id} (tokenizer/model synchronized)")
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    if bool(train_cfg.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    template = getattr(model.config, "nli_template", DEFAULT_NLI_TEMPLATE)
    stages = config.get("stages")
    if not stages:
        raise ValueError("Sprint 3 decision config requires at least one stage")
    stage_results = []
    for stage_index, stage in enumerate(stages, start=1):
        stage_name = str(stage.get("name", f"stage{stage_index}"))
        print(f"\n[decision stage {stage_index}/{len(stages)}] {stage_name}")
        mixture = build_decision_mixture(
            stage["datasets"],
            seed=seed + stage_index - 1,
            epoch_examples=stage.get("epoch_examples"),
            dev_max_per_dataset=stage.get("dev_max_per_dataset", 2000),
        )
        print(f"  train examples: {len(mixture.train):,}")
        print(f"  dev examples:   {len(mixture.dev):,}")
        for source in mixture.manifest["sources"]:
            print(
                f"  - {source['name']:<12} weight={source['weight']:<5g} "
                f"train={source['sampled_train']:,} dev={source['dev']:,}"
            )

        train_ds = Dataset.from_list(_dataset_rows(mixture.train, template))
        dev_ds = Dataset.from_list(_dataset_rows(mixture.dev, template))
        max_length = int(stage.get("max_length", config["model"].get("max_length", 512)))

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=max_length)

        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        dev_ds = dev_ds.map(tokenize, batched=True, remove_columns=["text"])
        stage_dir = ensure_dir(output_dir / f"stage-{stage_index:02d}-{stage_name}")
        precision = runtime.precision
        desired = dict(
            output_dir=str(stage_dir),
            seed=seed,
            data_seed=seed,
            num_train_epochs=float(stage.get("epochs", train_cfg.get("epochs", 1.0))),
            learning_rate=float(stage.get("learning_rate", train_cfg.get("learning_rate", 2e-5))),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            per_device_train_batch_size=int(train_cfg.get("per_device_batch_size", 4)),
            per_device_eval_batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("per_device_batch_size", 4))),
            gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
            logging_steps=int(train_cfg.get("logging_steps", 10)),
            eval_strategy="steps",
            eval_steps=int(train_cfg.get("eval_steps", 100)),
            save_strategy="steps",
            save_steps=int(train_cfg.get("save_steps", 100)),
            save_total_limit=int(train_cfg.get("save_total_limit", 2)),
            load_best_model_at_end=bool(train_cfg.get("load_best_model_at_end", False)),
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            bf16=precision == "bfloat16",
            fp16=precision == "float16",
            tf32=bool(train_cfg.get("tf32", True)) and runtime.device == "cuda",
            gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
            report_to=train_cfg.get("report_to", ["tensorboard"]),
            # PEFT wraps the underlying sequence-classification model with a generic
            # forward signature. Transformers 5.x can otherwise conclude that the
            # supervision column is unused and silently drop ``labels`` before the
            # batch reaches Qwen3_5ForSequenceClassification. Keep all already-
            # tokenized columns and explicitly declare the label column.
            remove_unused_columns=False,
            label_names=["labels"],
            optim=train_cfg.get("optimizer", "adamw_torch"),
            use_cpu=runtime.use_cpu,
            run_name=f"{config.get('run_name', 'biojev')}-{stage_name}",
        )
        args = TrainingArguments(**_compatible_args(desired, float(train_cfg.get("warmup_ratio", 0.03))))
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Fail before Trainer starts the expensive loop if supervision vanished.
        # This specifically guards the Transformers 5.x + PEFT column-pruning path
        # that can otherwise remove ``labels`` from sequence-classification batches.
        if "labels" not in train_ds.column_names:
            raise RuntimeError(
                f"BioJev decision dataset lost its labels column; columns={train_ds.column_names}"
            )
        probe_rows = [train_ds[i] for i in range(min(2, len(train_ds)))]
        probe_batch = collator(probe_rows)
        if "labels" not in probe_batch:
            raise RuntimeError(
                "BioJev data collator did not preserve labels. "
                f"Dataset columns={train_ds.column_names}; batch keys={list(probe_batch.keys())}"
            )
        if probe_batch["labels"].dtype != torch.long:
            probe_batch["labels"] = probe_batch["labels"].long()

        # Execute one genuine supervised forward pass before giving control to
        # Trainer. This catches missing labels, pooling/padding mistakes, or task-
        # head wiring errors immediately rather than after the training loop starts.
        probe_device = None
        for parameter in model.parameters():
            if parameter.device.type != "meta":
                probe_device = parameter.device
                break
        if probe_device is None:
            probe_device = torch.device("cuda:0" if runtime.device == "cuda" else "cpu")
        probe_inputs = {
            key: value.to(probe_device) if torch.is_tensor(value) else value
            for key, value in probe_batch.items()
        }
        was_training = model.training
        model.eval()
        with torch.no_grad():
            probe_output = model(**probe_inputs)
        if was_training:
            model.train()
        probe_loss = getattr(probe_output, "loss", None)
        if probe_loss is None or not torch.isfinite(probe_loss.detach()).all():
            raise RuntimeError(
                "BioJev supervised preflight failed: the model did not return a finite loss "
                f"for batch keys={list(probe_batch.keys())}."
            )
        print(
            "  supervision:       labels preserved; supervised forward OK "
            f"(probe loss={float(probe_loss.detach().cpu()):.4f})"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=_compute_metrics,
            callbacks=[JsonlLogCallback(stage_dir)],
        )
        trainer.train()
        metrics = trainer.evaluate()
        final_stage_dir = stage_dir / "final"
        trainer.save_model(str(final_stage_dir))
        tokenizer.save_pretrained(str(final_stage_dir))
        write_json(stage_dir / "mixture_manifest.json", mixture.manifest)
        write_json(stage_dir / "metrics.json", metrics)
        stage_results.append(
            {"name": stage_name, "checkpoint": str(final_stage_dir), "metrics": metrics, "mixture": mixture.manifest}
        )
        stage_manifest = {
            "format": "biojev-decision-v1",
            "run_name": config.get("run_name", output_dir.name),
            "seed": seed,
            "model": config["model"],
            "runtime": runtime.__dict__,
            "overlay": overlay,
            "labels": LABEL2ID,
            "nli_template": template,
            "stages": list(stage_results),
            "final_checkpoint": str(final_stage_dir),
        }
        write_json(final_stage_dir / "decision_manifest.json", stage_manifest)

    final_dir = Path(stage_results[-1]["checkpoint"])
    manifest = {
        "format": "biojev-decision-v1",
        "run_name": config.get("run_name", output_dir.name),
        "seed": seed,
        "model": config["model"],
        "runtime": runtime.__dict__,
        "overlay": overlay,
        "labels": LABEL2ID,
        "nli_template": template,
        "stages": stage_results,
        "final_checkpoint": str(final_dir),
    }
    write_json(final_dir / "decision_manifest.json", manifest)
    write_json(output_dir / "decision_manifest.json", manifest)
    print(f"\nBioJev decision training complete: {final_dir}")
    return DecisionTrainResult(str(output_dir), str(final_dir), stage_results, runtime.__dict__)
