from __future__ import annotations

from biojev.datasets.bionli import BioNLIAdapter
from biojev.datasets.bigbio_re import BigBioRelationAdapter
from biojev.datasets.nli4ct import NLI4CTAdapter
from biojev.datasets.general_nli import GeneralNLIAdapter, SPECS as GENERAL_NLI_SPECS


def get_adapter(name: str, **kwargs):
    name = name.lower()
    if name == "bionli":
        return BioNLIAdapter(**kwargs)
    if name == "nli4ct":
        return NLI4CTAdapter(**kwargs)
    if name in GENERAL_NLI_SPECS:
        return GeneralNLIAdapter(name=name, **kwargs)
    if name in {"chemprot", "ddi2013", "biored"}:
        return BigBioRelationAdapter(name=name, **kwargs)
    raise KeyError(f"Unknown dataset adapter: {name}")
