from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from biojev.schemas import NLIExample, RelationExample

Example = NLIExample | RelationExample


class DatasetAdapter(ABC):
    name: str

    def __init__(self, raw_root: str | Path = "data/raw", processed_root: str | Path = "data/processed"):
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)

    @abstractmethod
    def prepare(self, force: bool = False) -> dict[str, list[Example]]:
        """Download/load, normalize, and return examples keyed by split."""
        raise NotImplementedError
