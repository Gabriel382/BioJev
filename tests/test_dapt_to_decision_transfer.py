from types import SimpleNamespace

import torch

from biojev.training import decision


def test_seed_missing_sequence_classification_head(monkeypatch):
    model = SimpleNamespace(
        peft_config={"default": SimpleNamespace(modules_to_save=["score"])}
    )
    target = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 2),
        "base_model.model.score.weight": torch.full((3, 4), 7.0),
    }
    source = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 2),
    }
    monkeypatch.setattr(decision, "_get_peft_state", lambda _: target)

    state, seeded = decision._seed_missing_modules_to_save(model, source)

    assert torch.equal(
        state["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        source["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
    )
    assert "base_model.model.score.weight" in seeded
    assert torch.equal(state["base_model.model.score.weight"], target["base_model.model.score.weight"])


def test_does_not_replace_existing_task_head(monkeypatch):
    model = SimpleNamespace(
        peft_config={"default": SimpleNamespace(modules_to_save=["score"])}
    )
    target = {"base_model.model.score.weight": torch.full((3, 4), 7.0)}
    source = {"base_model.model.score.weight": torch.full((3, 4), 2.0)}
    monkeypatch.setattr(decision, "_get_peft_state", lambda _: target)

    state, seeded = decision._seed_missing_modules_to_save(model, source)

    assert seeded == []
    assert torch.equal(state["base_model.model.score.weight"], source["base_model.model.score.weight"])
