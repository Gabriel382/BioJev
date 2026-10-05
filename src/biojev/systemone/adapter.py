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
            # IMPORTANT:
            # AutoPeftModelForSequenceClassification may reconstruct the base model
            # with the Transformers default num_labels=2. BioJev is a 3-way NLI
            # classifier, and the saved PEFT checkpoint contains score.weight with
            # shape [3, hidden_size]. Rebuild the base exactly as during training,
            # then load the PEFT adapter on top.
            from peft import PeftConfig, PeftModelForSequenceClassification

            peft_cfg = PeftConfig.from_pretrained(self.checkpoint)
            base_name = peft_cfg.base_model_name_or_path
            if not base_name:
                raise RuntimeError(
                    "PEFT adapter_config.json does not define base_model_name_or_path"
                )

            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_name,
                num_labels=3,
                id2label={
                    0: "contradiction",
                    1: "entailment",
                    2: "neutral",
                },
                label2id={
                    "contradiction": 0,
                    "entailment": 1,
                    "neutral": 2,
                },
                **kwargs,
            )

            model = PeftModelForSequenceClassification.from_pretrained(
                base_model,
                self.checkpoint,
                is_trainable=False,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint,
                num_labels=3,
                id2label={
                    0: "contradiction",
                    1: "entailment",
                    2: "neutral",
                },
                label2id={
                    "contradiction": 0,
                    "entailment": 1,
                    "neutral": 2,
                },
                **kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError(
                    "Tokenizer has neither pad_token_id nor eos_token_id; "
                    "cannot batch System One candidates."
                )
            tokenizer.pad_token = tokenizer.eos_token

        # Qwen3.5ForSequenceClassification checks model.config.pad_token_id
        # when batch_size > 1. Setting only tokenizer.pad_token is not enough.
        model.config.pad_token_id = tokenizer.pad_token_id

        # Be defensive across PEFT wrappers / Transformers versions.
        if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
            model.base_model.config.pad_token_id = tokenizer.pad_token_id
        inner = getattr(getattr(model, "base_model", None), "model", None)
        if inner is not None and hasattr(inner, "config"):
            inner.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        if self.device != "cuda":
            model.to(self.device)

        # Fail early if a wrong classification head was reconstructed.
        if int(getattr(model.config, "num_labels", 0)) != 3:
            raise RuntimeError(
                f"Loaded checkpoint has num_labels={getattr(model.config, 'num_labels', None)}; "
                "BioJev requires 3 labels."
            )

        score = getattr(getattr(model, "base_model", model), "model", None)
        # The forward-pass validation below is more reliable than relying on
        # internal PEFT module names, which vary by version.
        probe = tokenizer(
            "BioJev compatibility probe.",
            "This statement is supported.",
            return_tensors="pt",
            truncation=True,
            max_length=min(self.max_length, 64),
        )
        target = "cuda" if self.device == "cuda" else self.device
        probe = {k: v.to(target) for k, v in probe.items()}
        with torch.inference_mode():
            probe_logits = model(**probe).logits
        if tuple(probe_logits.shape[-1:]) != (3,):
            raise RuntimeError(
                f"Loaded BioJev checkpoint produced logits shape {tuple(probe_logits.shape)}; "
                "expected [..., 3]."
            )

        print(
            "BioJev System One backend loaded: "
            f"checkpoint={self.checkpoint} | labels=3 | device={self.device}"
        )
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
        """
        Score candidates one-by-one.

        Qwen3.5ForSequenceClassification rejects batch_size > 1 when its
        internal classifier config does not expose pad_token_id. PEFT /
        Transformers can keep separate config objects internally, so setting
        pad_token_id on the outer wrapper is not always sufficient.

        System One questions normally contain only a small number of candidates,
        so sequential scoring is a robust compatibility strategy and avoids
        modifying the trained checkpoint.
        """
        supports: list[float] = []
        total_tokens = 0
        target = "cuda" if self.device == "cuda" else self.device

        for candidate in candidates:
            hypothesis = self._hypothesis(instructions, candidate)

            enc = self.tokenizer(
                state,
                hypothesis,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(target) for k, v in enc.items()}
            total_tokens += int(enc["attention_mask"].sum().item())

            logits = self.model(**enc).logits.float()[0]

            entailment = logits[self.label_ids["entailment"]]
            contradiction = logits[self.label_ids["contradiction"]]

            supports.append(float((entailment - contradiction).detach().cpu()))

        return supports, total_tokens

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
