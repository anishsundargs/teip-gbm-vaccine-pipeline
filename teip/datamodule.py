import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .utils import tokenize, normalize_feature

CONTEXT_COLUMNS = ["oip5_tpm","b2m_tpm","tap1_tpm","tap2_tpm","tapbp_tpm","hla_a_tpm","hla_b_tpm","hla_c_tpm","ifng_score","ap_score"]
REQUIRED_COLUMNS = ["peptide", "hla_pseudo"] + CONTEXT_COLUMNS

class PeptideHLADataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg):
        self.df = df.reset_index(drop=True).copy(); self.cfg = cfg
        if "row_id" not in self.df.columns: self.df["row_id"] = np.arange(len(self.df))
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing: raise ValueError(f"Missing required input columns: {missing}")
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        pep_tok = torch.tensor(tokenize(r["peptide"], self.cfg.data.max_pep), dtype=torch.long)
        hla_tok = torch.tensor(tokenize(r["hla_pseudo"], self.cfg.data.max_hla), dtype=torch.long)
        ctx_values = [normalize_feature(r.get(c, 0.0)) for c in CONTEXT_COLUMNS]
        ctx_values += [0.0] * max(0, self.cfg.model.ctx_dim - len(ctx_values))
        ctx = torch.tensor(ctx_values[:self.cfg.model.ctx_dim], dtype=torch.float32)
        y1 = torch.tensor(r.get("label_presentation", float("nan")), dtype=torch.float32)
        y2 = torch.tensor(r.get("label_targetability", float("nan")), dtype=torch.float32)
        rid = torch.tensor(r.get("row_id", i), dtype=torch.long)
        return pep_tok, hla_tok, ctx, y1, y2, rid

class DataModule:
    def __init__(self, cfg):
        self.cfg = cfg; self.df = pd.read_csv(cfg.data.data_csv)
        if "row_id" not in self.df.columns: self.df["row_id"] = np.arange(len(self.df))
    def setup(self, stage=None):
        train, val = train_test_split(self.df, test_size=self.cfg.data.val_fraction, random_state=self.cfg.seed, shuffle=True)
        self.ds_train = PeptideHLADataset(train, self.cfg); self.ds_val = PeptideHLADataset(val, self.cfg)
    def train_dataloader(self): return DataLoader(self.ds_train, batch_size=self.cfg.data.batch_size, shuffle=True, num_workers=self.cfg.data.num_workers)
    def val_dataloader(self): return DataLoader(self.ds_val, batch_size=self.cfg.data.batch_size, shuffle=False, num_workers=self.cfg.data.num_workers)
