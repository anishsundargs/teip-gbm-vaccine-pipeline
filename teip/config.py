from pathlib import Path
from omegaconf import OmegaConf

def load_config(path: str = "configs/config.yaml"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)
