from biojev.datasets.splitting import stratified_split
from biojev.schemas import NLIExample


def test_split_is_deterministic():
    rows = [
        NLIExample(id=str(i), dataset="x", split="train", premise="p", hypothesis="h",
                   label="entailment" if i % 2 else "contradiction")
        for i in range(100)
    ]
    a = stratified_split(rows, seed=7)
    b = stratified_split(rows, seed=7)
    assert [x.id for x in a["test"]] == [x.id for x in b["test"]]
    assert len(a["train"]) == 80
    assert len(a["dev"]) == 10
    assert len(a["test"]) == 10


def test_split_keeps_singleton_class_in_train():
    rows = [
        NLIExample(id=str(i), dataset="x", split="train", premise="p", hypothesis="h", label="common")
        for i in range(20)
    ]
    rows.append(
        NLIExample(id="rare", dataset="x", split="train", premise="p", hypothesis="h", label="rare")
    )
    parts = stratified_split(rows, seed=7)
    assert any(x.id == "rare" for x in parts["train"])
    assert not any(x.id == "rare" for x in parts["dev"])
    assert not any(x.id == "rare" for x in parts["test"])
