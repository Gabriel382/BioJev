from biojev.schemas import RelationExample
from biojev.transformations.hypotheses import relation_hypothesis


def test_chemprot_hypothesis():
    ex = RelationExample(
        id="x", dataset="chemprot", split="test", context="ctx",
        subject="Drug A", object="Protein B", label="CPR:4", candidates=["CPR:4"]
    )
    text = relation_hypothesis(ex, "CPR:4")
    assert "Drug A" in text and "Protein B" in text and "inhibits" in text
