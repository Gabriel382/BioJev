from __future__ import annotations

import re

from biojev.schemas import RelationExample

CHEMPROT = {
    "CPR:3": "{subject} upregulates or activates {object}.",
    "CPR:4": "{subject} downregulates or inhibits {object}.",
    "CPR:5": "{subject} is an agonist of {object}.",
    "CPR:6": "{subject} is an antagonist of {object}.",
    "CPR:9": "{subject} is a substrate or product associated with {object}.",
    "NO_RELATION": "No supported biochemical relation between {subject} and {object} is stated.",
}

DDI = {
    "MECHANISM": "The interaction between {subject} and {object} is described by a pharmacokinetic mechanism.",
    "EFFECT": "The interaction between {subject} and {object} is described by an effect on efficacy or response.",
    "ADVISE": "The text gives advice or a recommendation concerning the interaction between {subject} and {object}.",
    "INT": "The text states an interaction between {subject} and {object} without a more specific interaction type.",
    "NO_RELATION": "No drug-drug interaction between {subject} and {object} is supported by the text.",
}

BIORED = {
    "Association": "{subject} is associated with {object}.",
    "Positive_Correlation": "{subject} is positively correlated with {object}.",
    "Negative_Correlation": "{subject} is negatively correlated with {object}.",
    "Bind": "{subject} binds to {object}.",
    "Cotreatment": "{subject} and {object} are used together as a cotreatment.",
    "Comparison": "{subject} and {object} are compared in the study.",
    "Conversion": "{subject} is converted to or from {object}.",
    "Drug_Interaction": "A drug interaction is reported between {subject} and {object}.",
    "NO_RELATION": "No supported biomedical relation between {subject} and {object} is stated.",
}


def _generic(label: str) -> str:
    readable = re.sub(r"[_:-]+", " ", label).strip().lower()
    return "The relation between {subject} and {object} is " + readable + "."


def relation_hypothesis(example: RelationExample, label: str) -> str:
    if example.dataset == "chemprot":
        template = CHEMPROT.get(label, _generic(label))
    elif example.dataset == "ddi2013":
        template = DDI.get(label, _generic(label))
    elif example.dataset == "biored":
        template = BIORED.get(label, _generic(label))
    else:
        template = _generic(label)
    return template.format(subject=example.subject, object=example.object)
