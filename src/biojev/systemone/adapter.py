from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .schemas import validate_request

LABEL_FALLBACK = {"contradiction": 0, "entailment": 1, "neutral": 2}

def _normalized_entropy_confidence(probs: list[float]) -> float:
    if len(probs) <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(len(probs))))

class BioJevSystemOneAdapter:
    def __init__(self, checkpoint: str, model_name: str = "biojev", device: str | None = None,
                 load_in_4bit: bool = False, max_length: int = 2048):
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer, self.model = self._load_model(load_in_4bit)
        self.label_ids = self._resolve_label_ids()

    def _load_model(self, load_in_4bit: bool):
        kwargs: dict[str, Any] = {}
        if self.device == "cuda":
            kwargs.update(torch_dtype=torch.bfloat16, device_map={"": 0})
        else:
            kwargs["torch_dtype"] = torch.float32
        if load_in_4bit:
            if self.device != "cuda":
                raise ValueError("--load-in-4bit requires CUDA")
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
            )
        if (Path(self.checkpoint) / "adapter_config.json").exists():
            from peft import AutoPeftModelForSequenceClassification
            model = AutoPeftModelForSequenceClassification.from_pretrained(
                self.checkpoint, is_trainable=False, **kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        if self.device != "cuda":
            model.to(self.device)
        return tokenizer, model

    def _resolve_label_ids(self):
        id2label = getattr(self.model.config, "id2label", {}) or {}
        rev = {str(v).lower(): int(k) for k, v in id2label.items()}
        if set(LABEL_FALLBACK) <= set(rev):
            return {k: rev[k] for k in LABEL_FALLBACK}
        label2id = {str(k).lower(): int(v) for k, v in (getattr(self.model.config, "label2id", {}) or {}).items()}
        if set(LABEL_FALLBACK) <= set(label2id):
            return {k: label2id[k] for k in LABEL_FALLBACK}
        if int(getattr(self.model.config, "num_labels", 0)) != 3:
            raise RuntimeError("Expected a 3-class BioJev NLI checkpoint")
        return LABEL_FALLBACK.copy()

    def _hypothesis(self, instructions: str, candidate: str) -> str:
        return f"Decision question: {instructions}\nCandidate answer: {candidate}"

    @torch.inference_mode()
    def _supports(self, state: str, instructions: str, candidates: list[str]):
        prem = [state] * len(candidates)
        hyp = [self._hypothesis(instructions, c) for c in candidates]
        enc = self.tokenizer(prem, hyp, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        target = "cuda" if self.device == "cuda" else self.device
        enc = {k: v.to(target) for k, v in enc.items()}
        logits = self.model(**enc).logits.float()
        e = logits[:, self.label_ids["entailment"]]
        c = logits[:, self.label_ids["contradiction"]]
        return (e - c).cpu().tolist(), int(enc["attention_mask"].sum().item())

    @staticmethod
    def _softmax(values: list[float]) -> list[float]:
        return torch.softmax(torch.tensor(values, dtype=torch.float64), dim=0).tolist()

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = validate_request(payload)
        answers, input_tokens = {}, 0
        for name, q in req["questions"].items():
            keys = [k for k, _ in q["options"]]
            desc = [d for _, d in q["options"]]
            support, ntok = self._supports(req["state"], q["instructions"], desc)
            input_tokens += ntok
            probs = self._softmax(support)
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": float(probs[1])}
                continue
            pmap = {k: float(p) for k, p in zip(keys, probs)}
            conf = _normalized_entropy_confidence(probs)
            if q["type"] == "choice":
                i = max(range(len(probs)), key=probs.__getitem__)
                answers[name] = {"type": "choice", "choice": keys[i],
                                 "probabilities": pmap, "confidence": float(conf)}
            else:
                score = sum(i * p for i, p in enumerate(probs))
                legend = {str(i): d for i, (_, d) in enumerate(q["options"])}
                answers[name] = {"type": "score", "score": float(score),
                                 "legend": legend, "probabilities": pmap,
                                 "confidence": float(conf)}
        return {"model": req["model"], "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
