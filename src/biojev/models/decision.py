from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from biojev.models.base import BenchmarkModel
from biojev.schemas import NLIExample, PredictionRecord, RelationExample
from biojev.transformations.hypotheses import relation_hypothesis

DEFAULT_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"


def _synchronize_padding_token(model, tokenizer) -> int:
    """Synchronize tokenizer padding with the effective config used by Qwen3.5.

    Transformers 5.x sequence classification reads the nested text config for
    composite models such as Qwen3.5. PEFT can add one or more wrapper layers,
    so update every reachable config after the final adapter has been loaded.
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "BioJev inference requires a padding token, but the tokenizer "
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
        if hasattr(cfg, "pad_token_id"):
            cfg.pad_token_id = pad_token_id
        getter = getattr(cfg, "get_text_config", None)
        if callable(getter):
            try:
                text_cfg = getter()
            except TypeError:
                text_cfg = None
            if text_cfg is not None and text_cfg is not cfg:
                sync_config(text_cfg)
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

    # Verify the exact config path used by Transformers' generic sequence
    # classification forward where possible.
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
            "BioJev Sprint 4 failed to synchronize the effective text config "
            f"pad_token_id (tokenizer={pad_token_id}, model={effective_pad})."
        )
    return pad_token_id


class BioJevDecisionModel(BenchmarkModel):
    """Load a Sprint-3 BioJev checkpoint and expose NLI + typed decisions."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        max_length: int = 2048,
        torch_dtype: str = "auto",
    ):
        self.checkpoint = Path(checkpoint)
        self.name = self.checkpoint.parent.parent.name if self.checkpoint.name == "final" else self.checkpoint.name
        self.max_length = max_length
        manifest_path = self.checkpoint / "decision_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing BioJev decision manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_cfg = self.manifest["model"]
        init = model_cfg.get("initialization", "openjev")
        if init == "openjev":
            source = model_cfg["repo_id"]
            subfolder = model_cfg.get("subfolder")
        else:
            source = model_cfg["base_model"]
            subfolder = None

        tok_kwargs: dict[str, Any] = {"trust_remote_code": True}
        base_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device,
            "num_labels": 3,
            "label2id": {"contradiction": 0, "entailment": 1, "neutral": 2},
            "id2label": {0: "contradiction", 1: "entailment", 2: "neutral"},
        }
        if init != "openjev":
            base_kwargs["ignore_mismatched_sizes"] = True
        if subfolder:
            tok_kwargs["subfolder"] = subfolder
            base_kwargs["subfolder"] = subfolder
        if torch_dtype != "auto" and hasattr(torch, torch_dtype):
            base_kwargs["dtype"] = getattr(torch, torch_dtype)
        else:
            base_kwargs["dtype"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(source, **tok_kwargs)
        base = AutoModelForSequenceClassification.from_pretrained(source, **base_kwargs)

        adapter_cfg = self.checkpoint / "adapter_config.json"
        if adapter_cfg.exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, str(self.checkpoint))
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.checkpoint), device_map=device, trust_remote_code=True, dtype="auto"
            )
        pad_token_id = _synchronize_padding_token(self.model, self.tokenizer)
        print(f"  padding token:      {pad_token_id} (tokenizer/model synchronized for inference)")
        self.model.eval()
        self.template = self.manifest.get("nli_template", getattr(self.model.config, "nli_template", DEFAULT_TEMPLATE))
        raw = self.model.config.id2label or {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.id2label = {int(k): str(v).lower() for k, v in raw.items()}

    def _device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 8):
        outputs: list[dict[str, float]] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            texts = [self.template.format(premise=p, hypothesis=h) for p, h in batch]
            encoded = self.tokenizer(
                texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            device = self._device()
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.inference_mode():
                probs = torch.softmax(self.model(**encoded).logits.float(), dim=-1).cpu()
            for vector in probs:
                outputs.append({self.id2label[i]: float(vector[i]) for i in range(len(vector))})
        return outputs

    def decide(self, context: str, hypotheses: list[str], batch_size: int = 8):
        if not hypotheses:
            raise ValueError("decide() requires at least one hypothesis")
        raw = self.score_pairs([(context, h) for h in hypotheses], batch_size=batch_size)
        scores = [max(0.0, p.get("entailment", 0.0)) for p in raw]
        total = sum(scores)
        normalized = [s / total for s in scores] if total > 0 else [1.0 / len(scores)] * len(scores)
        choice = int(max(range(len(normalized)), key=normalized.__getitem__))
        return {
            "choice": choice,
            "hypothesis": hypotheses[choice],
            "probabilities": normalized,
            "nli": raw,
        }

    def predict_nli(self, examples: list[NLIExample], batch_size: int = 8):
        if not examples:
            return []
        raw = self.score_pairs([(x.premise, x.hypothesis) for x in examples], batch_size)
        label_space = sorted({x.label.lower() for x in examples})
        result = []
        for ex, probabilities in zip(examples, raw):
            projected = {label: max(0.0, probabilities.get(label, 0.0)) for label in label_space}
            denom = sum(projected.values())
            projected = (
                {k: v / denom for k, v in projected.items()}
                if denom > 0
                else {k: 1.0 / len(projected) for k in projected}
            )
            pred = max(projected, key=projected.get)
            result.append(
                PredictionRecord(
                    id=ex.id,
                    dataset=ex.dataset,
                    split=ex.split,
                    task=ex.task,
                    gold=ex.label.lower(),
                    prediction=pred,
                    probabilities=projected,
                    metadata={"raw_nli_probabilities": probabilities},
                )
            )
        return result

    def predict_relations(self, examples: list[RelationExample], batch_size: int = 8):
        pairs: list[tuple[str, str]] = []
        owners: list[tuple[int, str]] = []
        for i, ex in enumerate(examples):
            for label in ex.candidates:
                pairs.append((ex.context, relation_hypothesis(ex, label)))
                owners.append((i, label))
        scores = self.score_pairs(pairs, batch_size)
        grouped: dict[int, dict[str, float]] = defaultdict(dict)
        for (owner, label), probs in zip(owners, scores):
            grouped[owner][label] = probs.get("entailment", 0.0)
        output = []
        for i, ex in enumerate(examples):
            label_scores = grouped[i]
            total = sum(label_scores.values())
            probabilities = (
                {k: v / total for k, v in label_scores.items()}
                if total > 0
                else {k: 1.0 / len(label_scores) for k in label_scores}
            )
            pred = max(probabilities, key=probabilities.get)
            output.append(
                PredictionRecord(
                    id=ex.id,
                    dataset=ex.dataset,
                    split=ex.split,
                    task=ex.task,
                    gold=ex.label,
                    prediction=pred,
                    probabilities=probabilities,
                    metadata={"decision": "argmax entailment across candidate hypotheses", "raw_entailment_scores": label_scores},
                )
            )
        return output
