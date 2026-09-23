from __future__ import annotations

import torch

from biojev.models.base import BenchmarkModel
from biojev.models.causal import load_causal_model, model_input_device
from biojev.schemas import NLIExample, PredictionRecord, RelationExample
from biojev.transformations.hypotheses import relation_hypothesis


class CausalLikelihoodChoiceModel(BenchmarkModel):
    """Task-agnostic choice scoring with a causal LM.

    Candidates are ranked by mean conditional log-likelihood. This is preferable to instruction
    generation for Sprint 2 because Qwen3.5-Base and BioQwen differ only by biomedical DAPT; no
    instruction-following or decision supervision is assumed.
    """

    def __init__(
        self,
        repo_id: str,
        device: str = "auto",
        max_length: int = 4096,
        quantization: str | None = None,
    ):
        self.name = repo_id.replace("/", "_").replace("\\", "_")
        self.model, self.tokenizer = load_causal_model(
            repo_id, device_map=device, quantization=quantization
        )
        self.model.eval()
        self.max_length = max_length

    def _candidate_scores(self, prompt: str, continuations: list[str]) -> list[float]:
        """Score candidates with a low-memory prefill/decode path.

        We intentionally recompute the prompt cache for each candidate. Qwen3.5 contains recurrent
        linear-attention state, and cache objects may be mutated during decode; recomputing avoids
        candidate-order leakage while `logits_to_keep=1` prevents allocating prompt-length logits
        over Qwen's ~248k vocabulary.
        """
        device = model_input_device(self.model)
        scores: list[float] = []

        for continuation in continuations:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            continuation_ids = self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
            if not continuation_ids:
                scores.append(float("-inf"))
                continue

            room_for_prompt = max(1, self.max_length - len(continuation_ids))
            prompt_ids = prompt_ids[-room_for_prompt:]
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            with torch.inference_mode():
                prefill = self.model(
                    input_ids=prompt_tensor,
                    use_cache=True,
                    logits_to_keep=1,
                )
                first_log_probs = torch.log_softmax(prefill.logits[0, -1].float(), dim=-1)
                token_log_probs = [float(first_log_probs[continuation_ids[0]].cpu())]

                if len(continuation_ids) > 1:
                    decode_input = torch.tensor(
                        [continuation_ids[:-1]], dtype=torch.long, device=device
                    )
                    decode = self.model(
                        input_ids=decode_input,
                        past_key_values=prefill.past_key_values,
                        use_cache=False,
                        logits_to_keep=0,
                    )
                    decode_log_probs = torch.log_softmax(decode.logits[0].float(), dim=-1)
                    next_tokens = torch.tensor(continuation_ids[1:], dtype=torch.long, device=device)
                    positions = torch.arange(len(continuation_ids) - 1, device=device)
                    gathered = decode_log_probs[positions, next_tokens]
                    token_log_probs.extend(gathered.detach().cpu().tolist())

            scores.append(sum(token_log_probs) / len(token_log_probs))
        return scores

    @staticmethod
    def _softmax_dict(labels: list[str], scores: list[float]) -> dict[str, float]:
        tensor = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(tensor, dim=0).tolist()
        return {label: float(prob) for label, prob in zip(labels, probs)}

    def predict_nli(self, examples: list[NLIExample], batch_size: int = 1):
        # Candidate scoring already batches all labels per example. `batch_size` is kept for API parity.
        all_labels = sorted({x.label for x in examples})
        records = []
        for ex in examples:
            prompt = (
                f"Premise:\n{ex.premise}\n\nHypothesis:\n{ex.hypothesis}\n\n"
                "The relationship between the premise and hypothesis is"
            )
            continuations = [f" {label}." for label in all_labels]
            scores = self._candidate_scores(prompt, continuations)
            probs = self._softmax_dict(all_labels, scores)
            pred = max(probs, key=probs.get)
            records.append(
                PredictionRecord(
                    id=ex.id,
                    dataset=ex.dataset,
                    split=ex.split,
                    task=ex.task,
                    gold=ex.label,
                    prediction=pred,
                    probabilities=probs,
                    metadata={"mean_log_likelihood": dict(zip(all_labels, scores))},
                )
            )
        return records

    def predict_relations(self, examples: list[RelationExample], batch_size: int = 1):
        records = []
        for ex in examples:
            labels = ex.candidates
            prompt = (
                f"Biomedical context:\n{ex.context}\n\n"
                f"Entities: {ex.subject} ; {ex.object}\n\n"
                "The evidence supports the following relation statement:"
            )
            continuations = [" " + relation_hypothesis(ex, label) for label in labels]
            scores = self._candidate_scores(prompt, continuations)
            probs = self._softmax_dict(labels, scores)
            pred = max(probs, key=probs.get)
            records.append(
                PredictionRecord(
                    id=ex.id,
                    dataset=ex.dataset,
                    split=ex.split,
                    task=ex.task,
                    gold=ex.label,
                    prediction=pred,
                    probabilities=probs,
                    metadata={"mean_log_likelihood": dict(zip(labels, scores))},
                )
            )
        return records
