from __future__ import annotations

from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from biojev.models.base import BenchmarkModel
from biojev.schemas import NLIExample, PredictionRecord, RelationExample
from biojev.transformations.hypotheses import relation_hypothesis


class OpenJevModel(BenchmarkModel):
    """OpenJev Qwen3.5 NLI cross-encoder baseline.

    Current upstream checkpoint labels are contradiction=0, entailment=1, neutral=2.
    Binary NLI datasets are evaluated by renormalizing only over labels present in the dataset,
    so a neutral score does not become an undocumented third gold class.
    """

    def __init__(
        self,
        repo_id: str = "AlexWortega/openjev",
        subfolder: str = "qwen3.5-4b-nli-v5",
        device: str = "auto",
        max_length: int = 2048,
        torch_dtype: str = "bfloat16",
    ):
        self.name = "openjev"
        self.repo_id = repo_id
        self.subfolder = subfolder
        self.max_length = max_length
        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else None
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            repo_id,
            subfolder=subfolder,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        raw = self.model.config.id2label or {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.id2label = {int(k): str(v).lower() for k, v in raw.items()}

    def _pairs(self, pairs: list[tuple[str, str]], batch_size: int) -> list[dict[str, float]]:
        outputs: list[dict[str, float]] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            texts = [f"Premise: {p}\nHypothesis: {h}" for p, h in batch]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits.float()
                probs = torch.softmax(logits, dim=-1).cpu()
            for vector in probs:
                outputs.append({self.id2label[i]: float(vector[i]) for i in range(len(vector))})
        return outputs

    @staticmethod
    def _project(probabilities: dict[str, float], labels: list[str]) -> dict[str, float]:
        labels = [x.lower() for x in labels]
        values = {label: max(0.0, probabilities.get(label, 0.0)) for label in labels}
        total = sum(values.values())
        if total <= 0:
            return {label: 1.0 / len(labels) for label in labels}
        return {label: value / total for label, value in values.items()}

    def predict_nli(self, examples: list[NLIExample], batch_size: int = 8):
        if not examples:
            return []
        label_space = sorted({x.label.lower() for x in examples})
        raw = self._pairs([(x.premise, x.hypothesis) for x in examples], batch_size)
        result = []
        for ex, probs in zip(examples, raw):
            projected = self._project(probs, label_space)
            pred = max(projected, key=projected.get)
            result.append(
                PredictionRecord(
                    id=ex.id, dataset=ex.dataset, split=ex.split, task=ex.task,
                    gold=ex.label.lower(), prediction=pred, probabilities=projected,
                    metadata={"raw_openjev_probabilities": probs},
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
        scores = self._pairs(pairs, batch_size)
        grouped: dict[int, dict[str, float]] = defaultdict(dict)
        for (owner, label), probs in zip(owners, scores):
            grouped[owner][label] = probs.get("entailment", 0.0)
        output = []
        for i, ex in enumerate(examples):
            label_scores = grouped[i]
            total = sum(label_scores.values())
            probabilities = (
                {k: v / total for k, v in label_scores.items()}
                if total > 0 else {k: 1 / len(label_scores) for k in label_scores}
            )
            pred = max(probabilities, key=probabilities.get)
            output.append(
                PredictionRecord(
                    id=ex.id, dataset=ex.dataset, split=ex.split, task=ex.task,
                    gold=ex.label, prediction=pred, probabilities=probabilities,
                    metadata={"decision": "argmax entailment across candidate relation hypotheses"},
                )
            )
        return output
