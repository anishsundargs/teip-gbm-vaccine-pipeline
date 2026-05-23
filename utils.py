import random
import numpy as np
import torch

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}

def seed_everything(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def tokenize(sequence: str, max_len: int) -> list[int]:
    sequence = "" if sequence is None else str(sequence).upper().strip()
    ids = [AA_TO_ID.get(ch, 0) for ch in sequence[:max_len]]
    return ids + [0] * max(0, max_len - len(ids))

def normalize_feature(value, default: float = 0.0) -> float:
    try:
        x = float(value)
        return default if np.isnan(x) or np.isinf(x) else x
    except Exception:
        return default
