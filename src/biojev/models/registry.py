from __future__ import annotations

from biojev.models.generative import GenerativeChoiceModel
from biojev.models.openjev import OpenJevModel
from biojev.models.seqcls import SequenceClassifierModel


def build_model(config: dict):
    kind = config["kind"]
    if kind == "openjev":
        return OpenJevModel(**config.get("params", {}))
    if kind == "generative":
        return GenerativeChoiceModel(**config["params"])
    if kind == "seqcls":
        return SequenceClassifierModel(**config["params"])
    raise KeyError(f"Unknown model kind: {kind}")
