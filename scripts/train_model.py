import argparse
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import OmegaConf
from teip.datamodule import DataModule
from teip.model import TargetabilityNetPL
from teip.utils import seed_everything

def main():
    parser = argparse.ArgumentParser(description="Train TEIP model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_csv", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.data_csv: cfg.data.data_csv = args.data_csv
    if args.out_dir: cfg.run_dir = args.out_dir
    seed_everything(cfg.seed)
    dm = DataModule(cfg); dm.setup(); model = TargetabilityNetPL(cfg)
    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices, max_epochs=cfg.trainer.max_epochs, log_every_n_steps=cfg.trainer.log_every_n_steps, default_root_dir=cfg.run_dir)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
    out_dir = Path(cfg.run_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.ckpt"; trainer.save_checkpoint(str(ckpt_path)); print(f"Saved checkpoint: {ckpt_path}")
if __name__ == "__main__": main()
