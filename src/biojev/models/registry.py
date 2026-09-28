from __future__ import annotations

from biojev.models.generative import GenerativeChoiceModel
from biojev.models.causal_choice import CausalLikelihoodChoiceModel
from biojev.models.openjev import OpenJevModel
from biojev.models.seqcls import SequenceClassifierModel
from biojev.models.decision import BioJevDecisionModel


def build_model(config: dict):
    kind = config["kind"]
    if kind == "openjev":
        return OpenJevModel(**config.get("params", {}))
    if kind == "generative":
        return GenerativeChoiceModel(**config["params"])
    if kind == "causal_choice":
        return CausalLikelihoodChoiceModel(**config["params"])
    if kind == "seqcls":
        return SequenceClassifierModel(**config["params"])
    if kind == "biojev_decision":
        return BioJevDecisionModel(**config["params"])
    raise KeyError(f"Unknown model kind: {kind}")
