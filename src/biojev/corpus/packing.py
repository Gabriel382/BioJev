from __future__ import annotations

import math
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import IterableDataset


def estimate_sequence_count(token_budget: int, sequence_length: int) -> int:
    if token_budget <= 0:
        raise ValueError("token_budget must be > 0")
    if sequence_length <= 1:
        raise ValueError("sequence_length must be > 1")
    return math.ceil(token_budget / sequence_length)


class PackedTokenStream(IterableDataset):
    """Tokenize and greedily pack streaming documents up to an exact token budget."""

    def __init__(
        self,
        documents: Iterable[dict[str, Any]],
        tokenizer,
        *,
        token_budget: int,
        sequence_length: int = 2048,
        add_eos_between_documents: bool = True,
    ) -> None:
        super().__init__()
        self.documents = documents
        self.tokenizer = tokenizer
        self.token_budget = int(token_budget)
        self.sequence_length = int(sequence_length)
        self.add_eos_between_documents = add_eos_between_documents
        if self.token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        if self.sequence_length <= 1:
            raise ValueError("sequence_length must be > 1")

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        buffer: list[int] = []
        emitted = 0
        eos = self.tokenizer.eos_token_id

        for doc in self.documents:
            if emitted >= self.token_budget:
                break
            text = str(doc.get("text", ""))
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            if self.add_eos_between_documents and eos is not None:
                ids = list(ids) + [int(eos)]
            buffer.extend(ids)

            while buffer and emitted < self.token_budget:
                remaining = self.token_budget - emitted
                target = min(self.sequence_length, remaining)
                if len(buffer) < target:
                    break
                chunk = buffer[:target]
                del buffer[:target]
                emitted += len(chunk)
                yield {
                    "input_ids": chunk,
                    "attention_mask": [1] * len(chunk),
                    "labels": list(chunk),
                }

        if buffer and emitted < self.token_budget:
            target = min(len(buffer), self.sequence_length, self.token_budget - emitted)
            chunk = buffer[:target]
            if chunk:
                yield {
                    "input_ids": chunk,
                    "attention_mask": [1] * len(chunk),
                    "labels": list(chunk),
                }


class CausalLMCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        batch = self.tokenizer.pad(
            [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in features],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        label_tensor = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, row in enumerate(labels):
            label_tensor[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        batch["labels"] = label_tensor
        return batch
