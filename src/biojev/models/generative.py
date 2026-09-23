from __future__ import annotations

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from biojev.models.base import BenchmarkModel
from biojev.schemas import NLIExample, PredictionRecord, RelationExample
from biojev.transformations.hypotheses import relation_hypothesis


class GenerativeChoiceModel(BenchmarkModel):
    """Simple zero-shot generative baseline. It returns a hard one-hot decision.

    This is intentionally separated from calibrated classifier baselines: generated labels are
    not comparable probability estimates. Calibration metrics should therefore be interpreted
    only for models that expose meaningful probabilities.
    """

    def __init__(self, repo_id: str, device: str = "auto", max_new_tokens: int = 16):
        self.name = repo_id.replace("/", "_")
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id, trust_remote_code=True, device_map=device, torch_dtype="auto"
        )
        self.max_new_tokens = max_new_tokens

    def _choose(self, prompt: str, choices: list[str]) -> str:
        allowed = ", ".join(choices)
        full = (
            f"{prompt}\nChoose exactly one label from: {allowed}. "
            "Return only the label, with no explanation."
        )
        inputs = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        for choice in sorted(choices, key=len, reverse=True):
            if re.search(rf"\b{re.escape(choice)}\b", text, flags=re.I):
                return choice
        return choices[0]

    def predict_nli(self, examples: list[NLIExample], batch_size: int = 1):
        labels = sorted({x.label for x in examples})
        out = []
        for ex in examples:
            pred = self._choose(f"Premise: {ex.premise}\nHypothesis: {ex.hypothesis}", labels)
            out.append(PredictionRecord(
                id=ex.id, dataset=ex.dataset, split=ex.split, task=ex.task,
                gold=ex.label, prediction=pred,
                probabilities={label: float(label == pred) for label in labels},
                metadata={"probabilities_are_hard_labels": True},
            ))
        return out

    def predict_relations(self, examples: list[RelationExample], batch_size: int = 1):
        out = []
        for ex in examples:
            options = "\n".join(f"- {label}: {relation_hypothesis(ex, label)}" for label in ex.candidates)
            pred = self._choose(f"Context: {ex.context}\nCandidate relations:\n{options}", ex.candidates)
            out.append(PredictionRecord(
                id=ex.id, dataset=ex.dataset, split=ex.split, task=ex.task,
                gold=ex.label, prediction=pred,
                probabilities={label: float(label == pred) for label in ex.candidates},
                metadata={"probabilities_are_hard_labels": True},
            ))
        return out
