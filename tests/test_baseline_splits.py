from __future__ import annotations

import biojev.datasets.baseline_splits as bs
from biojev.schemas import NLIExample


def _rows(dataset: str, split: str, n: int):
    return [
        NLIExample(
            id=f"{dataset}-{split}-{i}",
            dataset=dataset,
            split=split,
            premise="p",
            hypothesis="h",
            label="entailment" if i % 2 else "contradiction",
        )
        for i in range(n)
    ]


def _patch_store(monkeypatch, mapping):
    def fake_load(dataset, split):
        key = (dataset, split)
        if key not in mapping:
            raise FileNotFoundError(key)
        return mapping[key]
    monkeypatch.setattr(bs, "load_processed", fake_load)


def test_official_validation_and_test_are_preserved(monkeypatch):
    mapping = {
        ("chemprot", "train"): _rows("chemprot", "train", 100),
        ("chemprot", "validation"): _rows("chemprot", "validation", 20),
        ("chemprot", "test"): _rows("chemprot", "test", 30),
    }
    _patch_store(monkeypatch, mapping)
    train, dev, test, manifest = bs.resolve_baseline_splits("chemprot", 42)
    assert len(train) == 100
    assert len(dev) == 20
    assert len(test) == 30
    assert manifest["dev_source"] == "validation"
    assert manifest["test_source"] == "test"


def test_official_test_is_preserved_when_dev_missing(monkeypatch):
    mapping = {
        ("ddi2013", "train"): _rows("ddi2013", "train", 100),
        ("ddi2013", "test"): _rows("ddi2013", "test", 30),
    }
    _patch_store(monkeypatch, mapping)
    train, dev, test, manifest = bs.resolve_baseline_splits("ddi2013", 42)
    assert len(train) == 90
    assert len(dev) == 10
    assert len(test) == 30
    assert manifest["dev_source"] == "synthetic_from_train"
    assert manifest["test_source"] == "test"


def test_official_dev_is_preserved_when_test_missing(monkeypatch):
    mapping = {
        ("nli4ct", "train"): _rows("nli4ct", "train", 100),
        ("nli4ct", "dev"): _rows("nli4ct", "dev", 20),
    }
    _patch_store(monkeypatch, mapping)
    train, dev, test, manifest = bs.resolve_baseline_splits("nli4ct", 42)
    assert len(train) == 90
    assert len(dev) == 20
    assert len(test) == 10
    assert manifest["dev_source"] == "dev"
    assert manifest["test_source"] == "synthetic_from_train"


def test_no_dev_or_test_synthesizes_both(monkeypatch):
    mapping = {
        ("bionli", "train"): _rows("bionli", "train", 100),
    }
    _patch_store(monkeypatch, mapping)
    train, dev, test, manifest = bs.resolve_baseline_splits("bionli", 42)
    assert len(train) == 80
    assert len(dev) == 10
    assert len(test) == 10
    assert manifest["dev_source"] == "synthetic_from_train"
    assert manifest["test_source"] == "synthetic_from_train"
