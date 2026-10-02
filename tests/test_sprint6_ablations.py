from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "general_only": (False, ["general-nli"]),
    "biomedical_only": (False, ["biomedical"]),
    "dapt_general": (True, ["general-nli"]),
    "dapt_biomedical": (True, ["biomedical"]),
    "general_biomedical": (False, ["general-nli", "biomedical"]),
    "full": (True, ["general-nli", "biomedical"]),
    "bionli_only": (True, ["general-nli", "biomedical"]),
    "nli4ct_only": (True, ["general-nli", "biomedical"]),
}


def _cfg(name):
    return yaml.safe_load((ROOT / f"configs/sprint6/train/{name}.yaml").read_text())


def test_all_variants_exist_and_match_factorization():
    for name, (has_dapt, stages) in VARIANTS.items():
        cfg = _cfg(name)
        assert bool(cfg["model"].get("dapt_adapter")) is has_dapt
        assert [x["name"] for x in cfg["stages"]] == stages
        assert cfg["model"]["base_model"] == "Qwen/Qwen3.5-0.8B-Base"
        assert cfg["seed"] == 42


def test_common_training_hyperparameters_are_identical():
    keys = [
        "method", "mixed_precision", "per_device_batch_size", "eval_batch_size",
        "gradient_accumulation_steps", "learning_rate", "weight_decay",
        "warmup_ratio", "gradient_checkpointing",
    ]
    ref = _cfg("full")["training"]
    for name in VARIANTS:
        cur = _cfg(name)["training"]
        assert {k: cur[k] for k in keys} == {k: ref[k] for k in keys}


def test_source_isolations_keep_40k_budget():
    b = _cfg("bionli_only")["stages"][1]
    n = _cfg("nli4ct_only")["stages"][1]
    assert b["epoch_examples"] == n["epoch_examples"] == 40000
    assert b["datasets"] == [{"name": "bionli", "weight": 1.0}]
    assert n["datasets"] == [{"name": "nli4ct", "weight": 1.0}]


def test_model_configs_point_to_expected_final_stage():
    for name in VARIANTS:
        cfg = yaml.safe_load((ROOT / f"configs/sprint6/models/{name}.yaml").read_text())
        ckpt = cfg["params"]["checkpoint"]
        if name in {"general_biomedical","full","bionli_only","nli4ct_only"}:
            assert "/stage-02-biomedical/final" in ckpt
        elif name in {"general_only","dapt_general"}:
            assert "/stage-01-general-nli/final" in ckpt
        else:
            assert "/stage-01-biomedical/final" in ckpt


def test_dapt_bootstrap_config_matches_nano_recipe():
    cfg = yaml.safe_load((ROOT / "configs/sprint6/dapt_nano_50k.yaml").read_text())
    assert cfg["model"]["base_model"] == "Qwen/Qwen3.5-0.8B-Base"
    assert cfg["corpus"]["token_budget"] == 50000
    assert cfg["corpus"]["sequence_length"] == 512
    assert cfg["training"]["method"] == "qlora"
    assert cfg["training"]["per_device_batch_size"] == 4
    assert cfg["training"]["gradient_accumulation_steps"] == 4
    assert cfg["training"]["gradient_checkpointing"] is False
    assert cfg["output_dir"] == "outputs/biojev-nano"


def test_dapt_source_weights_sum_to_one():
    cfg = yaml.safe_load((ROOT / "configs/sprint6/dapt_nano_50k.yaml").read_text())
    weights = [float(s["weight"]) for s in cfg["corpus"]["sources"]]
    assert abs(sum(weights) - 1.0) < 1e-9
    assert {s["name"] for s in cfg["corpus"]["sources"]} == {"pubmed", "pmc"}
