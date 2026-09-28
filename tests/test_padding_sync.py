from types import SimpleNamespace

from biojev.training.decision import _synchronize_padding_token


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = None
        self.eos_token_id = 42
        self.eos_token = "<eos>"
        self._pad_token = None

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        if value == self.eos_token:
            self.pad_token_id = self.eos_token_id


class CompositeConfig:
    """Mimics Qwen3.5: forward reads get_text_config().pad_token_id."""

    def __init__(self):
        self.text_config = SimpleNamespace(pad_token_id=None)

    def get_text_config(self):
        return self.text_config


class DummyModel:
    def __init__(self, child=None, composite=False):
        self.config = CompositeConfig() if composite else SimpleNamespace(pad_token_id=None)
        self.generation_config = SimpleNamespace(pad_token_id=None)
        self.model = child


def test_padding_sync_sets_wrapper_and_nested_configs():
    leaf = DummyModel()
    wrapper = DummyModel(child=leaf)
    tokenizer = DummyTokenizer()

    pad_id = _synchronize_padding_token(wrapper, tokenizer)

    assert pad_id == 42
    assert tokenizer.pad_token_id == 42
    assert wrapper.config.pad_token_id == 42
    assert wrapper.generation_config.pad_token_id == 42
    assert leaf.config.pad_token_id == 42
    assert leaf.generation_config.pad_token_id == 42


def test_padding_sync_sets_effective_qwen_text_config():
    model = DummyModel(composite=True)
    tokenizer = DummyTokenizer()

    pad_id = _synchronize_padding_token(model, tokenizer)

    assert pad_id == 42
    assert model.config.get_text_config().pad_token_id == 42
    assert model.config.text_config.pad_token_id == 42
    assert model.generation_config.pad_token_id == 42
