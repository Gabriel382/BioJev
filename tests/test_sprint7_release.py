from pathlib import Path
import ast, yaml
ROOT=Path(__file__).resolve().parents[1]

def test_config_tracks_sprints_4_5_6():
    c=yaml.safe_load((ROOT/"configs/sprint7/release.yaml").read_text())
    paths=[x["path"] for x in c["required_artifacts"]]
    assert any("sprint4" in x for x in paths)
    assert any("sprint5" in x for x in paths)
    assert any("sprint6" in x for x in paths)

def test_all_8_ablation_variants_registered():
    c=yaml.safe_load((ROOT/"configs/sprint7/release.yaml").read_text())
    assert len(c["ablation_models"]["variants"]) == 8
    assert "full" in c["ablation_models"]["variants"]

def test_optimizer_and_checkpoints_excluded():
    c=yaml.safe_load((ROOT/"configs/sprint7/release.yaml").read_text())
    s=" ".join(c["exclude_globs"])
    assert "optimizer" in s and "checkpoint-" in s

def test_scripts_parse():
    for p in (ROOT/"scripts").glob("*.py"):
        ast.parse(p.read_text())
