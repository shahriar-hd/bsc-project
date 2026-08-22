"""One real epoch through Trainer.fit(), plus a resume round-trip.

Exercises what check_train_step.py cannot: the composite metric, the scheduler
step, _log_epoch's console line and CSV row, checkpoint save, and whether
GradNorm's learned state survives resume.

    python tests/check_fit_epoch.py
"""
import csv
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

    # The composite must see both heads. The old form was
    # `0.5*df_auc + 0.5*(1 - sp_acer)`, and ACER is exactly 0.5 for a head that
    # predicts one class for everything — so run01's second term was frozen at
    # 0.25 for all 14 epochs while the same collapsed head had AUC 0.41. Both
    # forms are logged; `composite_metric` selects which one is monitored.
    df_auc = row.get("ff_df_auc_roc")
    sp_auc = row.get("siw_sp_auc_roc")
    sp_acer = row.get("siw_sp_acer")
    print(f"[fit ] ff_df_auc_roc={df_auc}  siw_sp_auc_roc={sp_auc}  "
          f"siw_sp_acer={sp_acer}")
    for name, v in (("ff_df_auc_roc", df_auc), ("siw_sp_auc_roc", sp_auc),
                    ("siw_sp_acer", sp_acer)):
        assert v is not None, f"{name} absent from row: {sorted(row)}"

    form = cfg.train.composite_metric
    expect_auc = 0.5 * float(df_auc) + 0.5 * float(sp_auc)
    expect_acer = 0.5 * float(df_auc) + 0.5 * (1.0 - float(sp_acer))
    expect = expect_auc if form == "auc" else expect_acer
    print(f"[fit ] composite_metric={form!r}  auc form={expect_auc:.4f}  "
          f"acer form={expect_acer:.4f}  best_metric={trainer.best_metric:.4f}")
    assert abs(expect - trainer.best_metric) < 1e-6, (
        f"composite does not match the {form} form ({expect:.6f} vs "
        f"{trainer.best_metric:.6f})")
    assert trainer.best_metric > 0.0, "composite is zero — both terms are dead"

    # Both forms must be logged regardless of which one is monitored, so the
    # thesis can quote the old definition without re-running anything.
    for k in ("composite", "composite_auc_form", "composite_acer_form"):
        assert k in row, f"{k} missing from the logged row: {sorted(row)}"
    assert abs(float(row["composite_auc_form"]) - expect_auc) < 1e-6
    assert abs(float(row["composite_acer_form"]) - expect_acer) < 1e-6
    print(f"[fit ] logged both forms: auc={row['composite_auc_form']:.4f} "
          f"acer={row['composite_acer_form']:.4f}")

    # The per-attack-type table that decides whether *more data* is needed, and
    # for which type. Empty here only if the subsample kept one type.
    per_type = sorted(k for k in row if k.startswith("siw_sp_recall_"))
    print(f"[fit ] per-type recall in the row ({len(per_type)}): "
          f"{[k.replace('siw_sp_recall_', '') for k in per_type]}")
    assert per_type, "no siw_sp_recall_<type> keys reached the epoch row"

    # ── Early stopping guards ──────────────────────────────────────────
    es = T.EarlyStopping(patience=2, min_delta=2e-3, mode="max",
                         min_epochs=8, smooth_window=3)
    flat = [0.70] * 12                     # a dead-flat plateau
    stops = [es.step(v) for v in flat]
    first_stop = stops.index(True) + 1 if True in stops else None
    print(f"[es  ] flat plateau: patience=2 min_epochs=8 -> first stop at "
          f"epoch {first_stop}")
    assert first_stop is not None, "early stopping never fired on a flat metric"
    assert first_stop >= 8, (
        f"stopped at epoch {first_stop} despite min_epochs=8 — the warmup region "
        f"(backbone lr 3e-9 at epoch 0) would decide the run")

    # Smoothing must absorb a single lucky epoch instead of locking `best` to it.
    # This is run01's exact shape: a plateau, one noise spike, then a genuine
    # climb that never reaches the spike. On raw values the spike becomes an
    # unbeatable best and patience runs out mid-climb; on a 3-epoch mean the
    # spike is averaged down and the climb still registers as improvement.
    spike = [0.70, 0.70, 0.86, 0.72, 0.74, 0.76, 0.78]
    es2 = T.EarlyStopping(patience=3, min_delta=2e-3, mode="max",
                          min_epochs=0, smooth_window=3)
    smoothed = [(es2.step(v), es2.epochs_without_improvement) for v in spike]
    es3 = T.EarlyStopping(patience=3, min_delta=2e-3, mode="max",
                          min_epochs=0, smooth_window=1)
    unsmoothed = [(es3.step(v), es3.epochs_without_improvement) for v in spike]
    print(f"[es  ] spike series {spike}")
    print(f"[es  ]   smooth_window=3 (stop, patience): {smoothed}")
    print(f"[es  ]   smooth_window=1 (stop, patience): {unsmoothed}")
    assert any(s for s, _ in unsmoothed), (
        "the unsmoothed series did not stop — this series no longer reproduces "
        "run01's failure, so the comparison below proves nothing")
    assert not any(s for s, _ in smoothed), (
        f"smoothing did not absorb the spike: stopped mid-climb at epoch "
        f"{[s for s, _ in smoothed].index(True) + 1} while the metric was still "
        f"rising {spike[3]} -> {spike[-1]}")
    # min_delta must be above the resolution of the val set: one clip out of
    # ~200 moves AUC by ~0.005, so the old 1e-4 counted noise as improvement.
    es4 = T.EarlyStopping(patience=1, min_delta=2e-3, mode="max", min_epochs=0)
    es4.step(0.70)
    assert es4.step(0.7005) is True, (
        "a +0.0005 change was treated as an improvement — min_delta is below the "
        "resolution of the validation set")
    print(f"[es  ] min_delta=2e-3 rejects a +0.0005 move as noise")

    run_dir = trainer.run_dir
    saved = sorted(f for f in os.listdir(run_dir) if f.endswith(".pth"))
    print(f"[fit ] {run_dir}: {saved}")
    assert "last.pth" in saved and "best.pth" in saved, f"missing ckpt: {saved}"

    # ── results.csv column alignment ───────────────────────────────────
    # The per-attack-type keys only exist for the types present in an epoch's
    # validation set, so the key set grows between rows. The header used to be
    # written once from row 0 while each row was written from its own keys, which
    # put later values under the wrong columns without any error.
    csv_path = os.path.join(run_dir, cfg.paths.result_csv)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"[csv ] {os.path.basename(csv_path)}: {len(header)} columns, "
          f"{len(rows)} rows")
    assert "composite" in header and "composite_acer_form" in header, \
        f"composite columns missing from the CSV header: {header[:12]}..."
    for i, r in enumerate(rows):
        assert None not in r, f"row {i} has more fields than the header"
        assert None not in r.values(), f"row {i} has fewer fields than the header"
    # A new key on a later row must add a column, not shift the existing ones.
    rl = T.ResultLogger(os.path.join(run_dir, "_colcheck.csv"))
    rl.log({"epoch": 0, "a": 1})
    rl.log({"epoch": 1, "a": 2, "sp_recall_Silicone": 0.5})
    rl.log({"epoch": 2, "a": 3})
    with open(os.path.join(run_dir, "_colcheck.csv"), newline="") as f:
        got = list(csv.DictReader(f))
    os.remove(os.path.join(run_dir, "_colcheck.csv"))
    print(f"[csv ] growing key set -> {got}")
    assert [r["a"] for r in got] == ["1", "2", "3"], \
        f"a late new column shifted earlier values: {got}"
    assert got[0]["sp_recall_Silicone"] == "" and got[1]["sp_recall_Silicone"] == "0.5"

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
