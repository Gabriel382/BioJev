from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from biojev.schemas import NLIExample, RelationExample
from biojev.utils.io import ensure_dir, write_json


@dataclass
class TrainResult:
    checkpoint: str
    label2id: dict[str, int]
    metrics: dict


def _to_rows(examples: Sequence[NLIExample | RelationExample], label2id: dict[str, int]):
    rows = []
    for ex in examples:
        if isinstance(ex, NLIExample):
            first, second = ex.premise, ex.hypothesis
        else:
            first = ex.context
            second = f"Subject: {ex.subject} ; Object: {ex.object}"
        rows.append({"text_a": first, "text_b": second, "labels": label2id[ex.label]})
    return rows


def train_sequence_classifier(
    model_id: str,
    train_examples: Sequence[NLIExample | RelationExample],
    dev_examples: Sequence[NLIExample | RelationExample],
    output_dir: str | Path,
    seed: int = 42,
    max_length: int = 512,
    learning_rate: float = 2e-5,
    epochs: float = 3.0,
    per_device_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
):
    labels = sorted({x.label for x in train_examples} | {x.label for x in dev_examples})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(labels), label2id=label2id, id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    train_ds = Dataset.from_list(_to_rows(train_examples, label2id))
    dev_ds = Dataset.from_list(_to_rows(dev_examples, label2id))

    def tokenize(batch):
        return tokenizer(
            batch["text_a"], batch["text_b"], truncation=True, max_length=max_length
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text_a", "text_b"])
    dev_ds = dev_ds.map(tokenize, batched=True, remove_columns=["text_a", "text_b"])

    def compute_metrics(pred):
        logits, gold = pred
        predicted = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(gold, predicted),
            "f1_macro": f1_score(gold, predicted, average="macro", zero_division=0),
            "f1_micro": f1_score(gold, predicted, average="micro", zero_division=0),
        }

    output_dir = ensure_dir(output_dir)
    args = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_json(output_dir / "labels.json", label2id)
    write_json(output_dir / "dev_metrics.json", metrics)
    return TrainResult(str(output_dir), label2id, metrics)
