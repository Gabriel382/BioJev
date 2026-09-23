from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from biojev.models.base import BenchmarkModel
from biojev.schemas import NLIExample, PredictionRecord, RelationExample


class SequenceClassifierModel(BenchmarkModel):
    def __init__(self, checkpoint: str, device: str = "auto", max_length: int = 512):
        self.name = checkpoint.replace("/", "_")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, device_map=device)
        self.max_length = max_length
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.model.eval()

    def _predict(self, first: list[str], second: list[str], batch_size: int):
        out = []
        for start in range(0, len(first), batch_size):
            enc = self.tokenizer(
                first[start:start + batch_size], second[start:start + batch_size],
                padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.inference_mode():
                probs = torch.softmax(self.model(**enc).logits.float(), -1).cpu()
            out.extend(probs)
        return out

    def predict_nli(self, examples: list[NLIExample], batch_size: int = 16):
        probs = self._predict([x.premise for x in examples], [x.hypothesis for x in examples], batch_size)
        out = []
        for ex, vec in zip(examples, probs):
            p = {self.id2label[i]: float(vec[i]) for i in range(len(vec))}
            pred = max(p, key=p.get)
            out.append(PredictionRecord(id=ex.id, dataset=ex.dataset, split=ex.split, task=ex.task,
                gold=ex.label, prediction=pred, probabilities=p))
        return out

    def predict_relations(self, examples: list[RelationExample], batch_size: int = 16):
        second = [f"Subject: {x.subject} ; Object: {x.object}" for x in examples]
        probs = self._predict([x.context for x in examples], second, batch_size)
        out = []
        for ex, vec in zip(examples, probs):
            p = {self.id2label[i]: float(vec[i]) for i in range(len(vec))}
            pred = max(p, key=p.get)
            out.append(PredictionRecord(id=ex.id, dataset=ex.dataset, split=ex.split, task=ex.task,
                gold=ex.label, prediction=pred, probabilities=p))
        return out
