from biojev.config import load_yaml


def test_nano_full_config_uses_base_aligned_dapt_then_two_decision_stages():
    cfg = load_yaml("configs/decision/nano_full.yaml")
    assert cfg["model"]["initialization"] == "qwen_seqcls"
    assert cfg["model"]["base_model"] == "Qwen/Qwen3.5-0.8B-Base"
    assert cfg["model"]["dapt_adapter"] == "outputs/biojev-nano/final"
    assert [stage["name"] for stage in cfg["stages"]] == ["general-nli", "biomedical"]
    bio_names = {x["name"] for x in cfg["stages"][1]["datasets"]}
    assert bio_names == {"bionli", "nli4ct"}
