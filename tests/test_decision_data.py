import pytest

from biojev.training.decision_data import LABEL2ID, normalize_nli_label


def test_openjev_label_order_is_fixed():
    assert LABEL2ID == {"contradiction": 0, "entailment": 1, "neutral": 2}


def test_normalize_nli_labels():
    assert normalize_nli_label("Entailment") == "entailment"
    assert normalize_nli_label("contradictory") == "contradiction"
    with pytest.raises(ValueError):
        normalize_nli_label("maybe")
