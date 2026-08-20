"""One real epoch through Trainer.fit(), plus a resume round-trip.

Exercises what check_train_step.py cannot: the composite metric, the scheduler
step, _log_epoch's console line and CSV row, checkpoint save, and whether
GradNorm's learned state survives resume.

    python tests/check_fit_epoch.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

import src.train as T
from src.config import get_config

CLIPS = 16  # whole clips per label, per dataset

_real_load_csv = T.load_csv


def load_csv_subsampled(path, *a, **kw):
    """Stratified subsample — a single-class split makes every guarded metric
    return nothing, which would hide the very bugs this script checks for."""
    df = _real_load_csv(path, *a, **kw)
    out = []
    for _, grp in df.groupby("label", sort=False):
        keys = grp.groupby(["video_path", "clip_index"], sort=False).ngroup()
        out.append(grp[keys < max(CLIPS // 2, 1)])
    sub = pd.concat(out).copy()
    print(f"[csv ] {os.path.basename(str(path))}: {len(df)} -> {len(sub)} rows "
          f"labels={sorted(sub['label'].unique())}")
    return sub


def main():
    T.load_csv = load_csv_subsampled

    cfg = get_config()
    cfg.train.num_epochs = 1
    cfg.train.warmup_epochs = 0   # 1-epoch run: skip warmup so the LR is real
    cfg.train.resume = False      # start clean; the resume path is tested below

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(cfg.paths.checkpoint_dir, cfg.paths.log_file)
    logger = T.setup_logger(log_path, cfg=cfg, name="fit_check")
    T.set_seed(cfg.train.seed)

    train_loader, val_ff_loader, val_siw_loader = T.build_loaders(cfg)
    trainer = T.Trainer(cfg, logger)

    # Move the task weights off their init so the resume check is meaningful.
    with torch.no_grad():
        trainer.gradnorm_manager.log_weights.copy_(
            torch.tensor([0.3, -0.2, 0.1], device=trainer.device))

    trainer.fit(train_loader, val_ff_loader, val_siw_loader)

    print(f"\n[fit ] best_metric={trainer.best_metric:.4f} "
          f"best_epoch={trainer.best_epoch}")
    assert trainer.history, "no epoch history recorded"
    row = trainer.history[-1]

    # The composite must actually see the deepfake AUC, not the 0.0 default.
    df_auc = row.get("ff_df_auc_roc")
    sp_acer = row.get("siw_sp_acer")
    print(f"[fit ] ff_df_auc_roc={df_auc}  siw_sp_acer={sp_acer}")
    assert df_auc is not None, f"ff_df_auc_roc absent from row: {sorted(row)}"
    assert sp_acer is not None, f"siw_sp_acer absent from row: {sorted(row)}"
    expect = 0.5 * float(df_auc) + 0.5 * (1.0 - float(sp_acer))
    print(f"[fit ] composite recomputed={expect:.4f} "
          f"best_metric={trainer.best_metric:.4f}")
    assert abs(expect - trainer.best_metric) < 1e-6, \
        "composite does not match 0.5*df_auc + 0.5*(1-sp_acer)"
    assert trainer.best_metric > 0.0, "composite is zero — df_auc term is dead"

    run_dir = trainer.run_dir
    saved = sorted(f for f in os.listdir(run_dir) if f.endswith(".pth"))
    print(f"[fit ] {run_dir}: {saved}")
    assert "last.pth" in saved and "best.pth" in saved, f"missing ckpt: {saved}"

    # ── Resume round-trip ──────────────────────────────────────────────
    ckpt = torch.load(os.path.join(run_dir, "last.pth"),
                      map_location="cpu", weights_only=False)
    assert "gradnorm" in ckpt, "GradNorm state not checkpointed"
    saved_w = trainer.gradnorm_manager.log_weights.detach().cpu().clone()
    saved_init = trainer.gradnorm_manager.initial_losses

    cfg.train.resume = True
    resumed = T.Trainer(cfg, logger)
    got_w = resumed.gradnorm_manager.log_weights.detach().cpu()
    print(f"[res ] log_weights saved={saved_w.numpy().round(4)} "
          f"restored={got_w.numpy().round(4)}")
    assert torch.allclose(saved_w, got_w, atol=1e-6), \
        "GradNorm log_weights not restored on resume"

    mirror = resumed.model.log_weights.detach().cpu()
    assert torch.allclose(mirror, got_w, atol=1e-6), \
        f"model.log_weights mirror not synced: {mirror.numpy()} vs {got_w.numpy()}"

    got_init = resumed.gradnorm_manager.initial_losses
    print(f"[res ] initial_losses saved="
          f"{None if saved_init is None else saved_init.cpu().numpy().round(4)} "
          f"restored={None if got_init is None else got_init.cpu().numpy().round(4)}")
    if saved_init is not None:
        assert got_init is not None, "L(0) baseline lost on resume"
        assert torch.allclose(saved_init.cpu(), got_init.cpu(), atol=1e-6), \
            "L(0) baseline not restored on resume"
    print(f"[res ] start_epoch={resumed.start_epoch} "
          f"best_metric={resumed.best_metric:.4f}")
    assert resumed.start_epoch == trainer.history[-1]["epoch"] + 1
    assert abs(resumed.best_metric - trainer.best_metric) < 1e-9

    print(f"[vram] peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB "
          f"reserved={torch.cuda.max_memory_reserved()/2**20:.0f} MiB")

    print("\nFIT CHECK PASSED")


if __name__ == "__main__":
    main()
