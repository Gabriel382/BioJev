
from pathlib import Path
import yaml
def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML object")
    return cfg
