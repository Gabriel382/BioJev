import importlib.util
import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("BioJev runtime dependencies not installed", allow_module_level=True)

from biojev.systemone.adapter import BioJevSystemOneAdapter, _normalized_entropy_confidence

def test_softmax():
    p = BioJevSystemOneAdapter._softmax([0, 1, -1])
    assert sum(p) == pytest.approx(1.0)
    assert p[1] > p[0] > p[2]

def test_confidence():
    assert _normalized_entropy_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert _normalized_entropy_confidence([0.999, 0.001]) > 0.98
