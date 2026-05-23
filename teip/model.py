import torch
import torch.nn as nn
import pytorch_lightning as pl
from .encoders import CharEncoder, CrossFusion

class TargetabilityNetPL(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; d = cfg.model.d_model
        self.char_pep = CharEncoder(d_model=d, max_len=cfg.data.max_pep, dropout=cfg.model.dropout)
        self.char_hla = CharEncoder(d_model=d, max_len=cfg.data.max_hla, dropout=cfg.model.dropout)
        self.fuse = CrossFusion(d=d, dropout=cfg.model.dropout)
        self.ctx_proj = nn.Sequential(nn.Linear(cfg.model.ctx_dim, d), nn.ReLU(), nn.Dropout(cfg.model.dropout))
        self.head_presentation = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Dropout(cfg.model.dropout), nn.Linear(128,1))
        self.head_targetability = nn.Sequential(nn.Linear(d*2,128), nn.ReLU(), nn.Dropout(cfg.model.dropout), nn.Linear(128,1))
        self.bce = nn.BCEWithLogitsLoss()
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay)
    def forward(self, pep_tok, hla_tok, ctx_vec, row_id=None):
        pep = self.char_pep(pep_tok); hla = self.char_hla(hla_tok); fused = self.fuse(pep, hla)
        p1 = self.head_presentation(fused).squeeze(-1)
        ctx = self.ctx_proj(ctx_vec); p2 = self.head_targetability(torch.cat([fused, ctx], dim=-1)).squeeze(-1)
        return p1, p2
    def step(self, batch, stage: str):
        pep, hla, ctx, y1, y2, rid = batch; p1, p2 = self(pep, hla, ctx, rid)
        loss = torch.tensor(0.0, device=self.device); logs = {}
        if not torch.isnan(y1).all():
            m = ~torch.isnan(y1); L = self.bce(p1[m], y1[m]); loss = loss + self.cfg.model.lam_presentation * L; logs[f"{stage}_presentation_loss"] = L
        if not torch.isnan(y2).all():
            m = ~torch.isnan(y2); L = self.bce(p2[m], y2[m]); loss = loss + self.cfg.model.lam_targetability * L; logs[f"{stage}_targetability_loss"] = L
        logs[f"{stage}_loss"] = loss; self.log_dict(logs, prog_bar=True, on_step=False, on_epoch=True); return loss
    def training_step(self, batch, batch_idx): return self.step(batch, "train")
    def validation_step(self, batch, batch_idx): return self.step(batch, "val")
