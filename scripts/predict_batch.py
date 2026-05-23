import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from teip.datamodule import PeptideHLADataset, CONTEXT_COLUMNS
from teip.model import TargetabilityNetPL

def fallback_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); ctx = out[[c for c in CONTEXT_COLUMNS if c in out.columns]].astype(float)
    z = (ctx - ctx.mean()) / (ctx.std(ddof=0).replace(0, 1)); raw = z.mean(axis=1).fillna(0.0)
    score = 1 / (1 + np.exp(-raw))
    out["pred_presentation"] = np.clip(score * 0.8 + 0.1, 0, 1)
    out["pred_targetability"] = np.clip(score * 0.7 + 0.15, 0, 1)
    out["teip_score"] = 0.5*out["pred_presentation"] + 0.5*out["pred_targetability"]
    out["mode"] = "fallback_demo_not_trained_model"
    return out

def run_model(df: pd.DataFrame, cfg, checkpoint: str) -> pd.DataFrame:
    ds = PeptideHLADataset(df, cfg); dl = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False)
    model = TargetabilityNetPL.load_from_checkpoint(checkpoint, cfg=cfg, strict=False, map_location="cpu"); model.eval()
    rows=[]
    with torch.no_grad():
        for pep, hla, ctx, y1, y2, rid in dl:
            p1, p2 = model(pep, hla, ctx, rid); p1=torch.sigmoid(p1); p2=torch.sigmoid(p2)
            for i in range(len(rid)): rows.append({"row_id": int(rid[i]), "pred_presentation": float(p1[i]), "pred_targetability": float(p2[i])})
    out = df.copy()
    if "row_id" not in out.columns: out["row_id"] = np.arange(len(out))
    out = out.merge(pd.DataFrame(rows), on="row_id", how="left")
    out["teip_score"] = 0.5*out["pred_presentation"] + 0.5*out["pred_targetability"]; out["mode"] = "trained_checkpoint"
    return out

def main():
    parser = argparse.ArgumentParser(description="Run TEIP batch prediction.")
    parser.add_argument("--input", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/config.yaml"); parser.add_argument("--checkpoint", default="models/teip_checkpoint.ckpt")
    parser.add_argument("--allow_fallback", action="store_true")
    args = parser.parse_args(); cfg = OmegaConf.load(args.config); df = pd.read_csv(args.input); checkpoint = Path(args.checkpoint)
    if checkpoint.exists(): out = run_model(df, cfg, str(checkpoint))
    else:
        if not args.allow_fallback: print(f"WARNING: checkpoint not found: {checkpoint}. Using fallback demo scoring only.")
        out = fallback_score(df)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values("teip_score", ascending=False).to_csv(output, index=False); print(f"Wrote predictions: {output}")
if __name__ == "__main__": main()
