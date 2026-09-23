from __future__ import annotations

from abc import ABC, abstractmethod

from biojev.schemas import NLIExample, PredictionRecord, RelationExample


class BenchmarkModel(ABC):
    name: str

    @abstractmethod
    def predict_nli(self, examples: list[NLIExample], batch_size: int = 8) -> list[PredictionRecord]:
        raise NotImplementedError

    @abstractmethod
    def predict_relations(
        self, examples: list[RelationExample], batch_size: int = 8
    ) -> list[PredictionRecord]:
        raise NotImplementedError
