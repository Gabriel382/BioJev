#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from biojev.systemone_native.collator import CompletionOnlyCollator
from biojev.systemone_native.config import load_yaml
from biojev.systemone_native.data import read_jsonl
from biojev.systemone_native.format import to_instruction_pair, validate_answer_tokens


def _make_training_arguments(
    *,
    output_dir: str,
    train_len: int,
    epochs: float,
    max_steps: int,
    train_batch_size: int,
    eval_batch_size: int,
    grad_accum: int,
    learning_rate: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
    report_to,
    seed: int,
    dataloader_num_workers: int,
) -> TrainingArguments:
    """
    Build TrainingArguments across Transformers versions.

    Relevant API drift handled here:
      - warmup_ratio -> warmup_steps
      - eval_strategy <-> evaluation_strategy
      - unsupported optional kwargs are reported and omitted
    """
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters)

    if max_steps > 0:
        total_update_steps = max_steps
    else:
        micro_batches_per_epoch = math.ceil(train_len / max(1, train_batch_size))
        updates_per_epoch = math.ceil(micro_batches_per_epoch / max(1, grad_accum))
        total_update_steps = max(1, math.ceil(updates_per_epoch * epochs))

    warmup_steps = (
        max(1, int(round(total_update_steps * warmup_ratio)))
        if warmup_ratio > 0
        else 0
    )

    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": gradient_checkpointing,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "report_to": report_to,
        "remove_unused_columns": False,
        "seed": seed,
        "dataloader_num_workers": dataloader_num_workers,
    }

    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = warmup_ratio
        warmup_mode = f"warmup_ratio={warmup_ratio}"
    elif "warmup_steps" in supported:
        kwargs["warmup_steps"] = warmup_steps
        warmup_mode = f"warmup_steps={warmup_steps} (derived from ratio {warmup_ratio})"
    else:
        warmup_mode = "no warmup argument supported"

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"
        eval_mode = "eval_strategy"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"
        eval_mode = "evaluation_strategy"
    else:
        eval_mode = "no evaluation-strategy argument supported"

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(kwargs) - set(filtered))

    print(f"[Sprint 8] transformers={transformers.__version__}")
    print(f"[Sprint 8] TrainingArguments compatibility: {eval_mode}, {warmup_mode}")
    if dropped:
        print(f"[Sprint 8] unsupported TrainingArguments omitted: {dropped}")

    # These materially affect Sprint-8 training; fail rather than silently
    # changing the recipe if an unusually old/custom Transformers build lacks them.
    critical = {
        "output_dir",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
    }
    missing_critical = sorted(critical - supported)
    if missing_critical:
        raise RuntimeError(
            "This Transformers TrainingArguments build is missing critical "
            f"arguments: {missing_critical}"
        )

    return TrainingArguments(**filtered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    m = cfg["model"]
    t = cfg["training"]
    d = cfg["data"]

    base = Path(m["prepared_base"])
    if not base.exists():
        raise SystemExit(
            f"{base} missing; run prepare_sprint8_base.py first"
        )

    out = Path(t["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ids = validate_answer_tokens(tok, 26)
    print("[Sprint 8] A-Z token validation OK", {k: ids[k] for k in "ABC"})

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=t.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bool(t.get("double_quant", True)),
    )

    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quant,
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(
            t.get("gradient_checkpointing", True)
        ),
    )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(t.get("lora_r", 8)),
        lora_alpha=int(t.get("lora_alpha", 16)),
        lora_dropout=float(t.get("lora_dropout", 0.0)),
        target_modules=t.get("target_modules", "all-linear"),
        bias="none",
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dd = Path(d["output_dir"])
    train_rows = read_jsonl(dd / "train.jsonl")
    dev_rows = read_jsonl(dd / "dev.jsonl")

    if a.smoke:
        train_rows = train_rows[: min(64, len(train_rows))]
        dev_rows = dev_rows[: min(16, len(dev_rows))]

    maxlen = int(t.get("max_length", 2048))

    def encode(rows):
        encoded = []
        dropped = 0

        for r in rows:
            try:
                x = to_instruction_pair(r, tok, maxlen)
                encoded.append(
                    {
                        "input_ids": [int(v) for v in x["input_ids"]],
                        "attention_mask": [
                            int(v) for v in x["attention_mask"]
                        ],
                        "labels": [int(v) for v in x["labels"]],
                    }
                )
            except ValueError as exc:
                if (
                    "max_length" in str(exc)
                    or "tokens >" in str(exc)
                ):
                    dropped += 1
                    continue
                raise

        print(
            f"[Sprint 8] encoded={len(encoded)} "
            f"dropped_for_length={dropped}"
        )
        return encoded

    train_encoded = encode(train_rows)
    dev_encoded = encode(dev_rows)

    if not train_encoded:
        raise SystemExit("No training examples remained after encoding.")
    if not dev_encoded:
        raise SystemExit("No dev examples remained after encoding.")

    tr = Dataset.from_dict(
        {
            "input_ids": [x["input_ids"] for x in train_encoded],
            "attention_mask": [
                x["attention_mask"] for x in train_encoded
            ],
            "labels": [x["labels"] for x in train_encoded],
        }
    )
    dv = Dataset.from_dict(
        {
            "input_ids": [x["input_ids"] for x in dev_encoded],
            "attention_mask": [
                x["attention_mask"] for x in dev_encoded
            ],
            "labels": [x["labels"] for x in dev_encoded],
        }
    )

    eval_steps = int(t.get("eval_steps", 250))
    save_steps = int(t.get("save_steps", 250))
    max_steps = int(t.get("max_steps", -1))
    epochs = float(t.get("num_train_epochs", 1.0))

    if a.smoke:
        max_steps = 8
        eval_steps = 4
        save_steps = 4
        epochs = 1.0

    train_bs = int(t.get("per_device_train_batch_size", 1))
    eval_bs = int(t.get("per_device_eval_batch_size", 1))
    grad_accum = int(t.get("gradient_accumulation_steps", 16))

    training_args = _make_training_arguments(
        output_dir=str(out),
        train_len=len(tr),
        epochs=epochs,
        max_steps=max_steps,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        grad_accum=grad_accum,
        learning_rate=float(t.get("learning_rate", 5e-5)),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(
            t.get("gradient_checkpointing", True)
        ),
        logging_steps=int(t.get("logging_steps", 10)),
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=int(t.get("save_total_limit", 2)),
        report_to=t.get("report_to", "none"),
        seed=seed,
        dataloader_num_workers=int(
            t.get("dataloader_num_workers", 2)
        ),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr,
        eval_dataset=dv,
        data_collator=CompletionOnlyCollator(tok),
    )

    result = trainer.train()
    metrics = trainer.evaluate()

    final = out / "final"
    trainer.save_model(final)
    tok.save_pretrained(final)

    manifest = {
        "config": a.config,
        "transformers_version": transformers.__version__,
        "prepared_base": str(base),
        "final_adapter": str(final),
        "train_metrics": result.metrics,
        "eval_metrics": metrics,
        "format": (
            "Tev-style causal LM, completion-only A-Z letter supervision"
        ),
        "ollama_target": "/v1/systemone",
    }

    (out / "sprint8_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
