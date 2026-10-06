
from __future__ import annotations
import torch

class CompletionOnlyCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer requires pad/eos")
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        ids, masks, labels = [], [], []
        for x in features:
            pad = max_len - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            masks.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
