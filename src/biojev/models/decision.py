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
                    metadata={"decision": "argmax entailment across candidate hypotheses"},
                )
            )
        return output
