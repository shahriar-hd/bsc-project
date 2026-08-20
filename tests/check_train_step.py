"""Verification for the training fixes: gradient flow, weight updates, VRAM.

Drives the real code path (real CSVs, real Dataset/Trainer) on a tiny stratified
subsample so a few optimizer steps run in seconds instead of an hour.

    python tests/check_train_step.py
"""
import logging
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

CLIPS = 24          # clips per dataset — a handful of accumulation windows
STEPS = 9           # enough to cross several accum boundaries + a GradNorm step


def subsample(df, n_clips):
    """Keep n_clips whole clips *per label*.

    The CSVs are label-sorted, so taking the first N rows yields a single-class
    set — and every metric guarded by `len(unique(labels)) < 2` then returns
    nothing, which looks exactly like a broken metric implementation.
    """
    out = []
    for _, grp in df.groupby("label", sort=False):
        keys = grp.groupby(["video_path", "clip_index"], sort=False).ngroup()
        out.append(grp[keys < max(n_clips // 2, 1)])
    return pd.concat(out).copy()


def grad_coverage(trainer, model, batch, dev):
    """Names of optimizer-managed params that receive no gradient at all."""
    trainer.optimizer.zero_grad(set_to_none=True)
    frames = batch["frames"].to(dev)
    df_l = batch["deepfake_label"].to(dev).float()
    sp_l = batch["spoof_label"].to(dev).float()
    tp_l = batch["temporal_label"].to(dev).float()
    with trainer.autocast_ctx:
        out = model(frames)
        loss = (trainer.criterion_df(out["deepfake_logit"], df_l)
                + trainer.criterion_sp(out["spoof_logit"], sp_l)
                + T.temporal_consistency_loss(
                    out["temp_proj"], tp_l,
                    logit=out["temp_logit"],
                    logit_weight=trainer.temporal_logit_weight))
    trainer.scaler.scale(loss).backward()
    ids = {id(p) for p in trainer.optim_params}
    dead = [n for n, p in model.named_parameters()
            if id(p) in ids and p.grad is None]
    trainer.optimizer.zero_grad(set_to_none=True)
    return dead


def main():
    cfg = get_config()
    tc = cfg.train
    logger = logging.getLogger("check")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    T.set_seed(tc.seed)
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated() / 2**20

    # ── Data via the real path ─────────────────────────────────────────
    ff = subsample(T.load_csv(cfg.paths.ff_train_csv), CLIPS)
    siw = subsample(T.load_csv(cfg.paths.siwmv2_train_csv), CLIPS)
    loader = T.build_interleaved_loader(ff, siw, cfg, is_train=True)
    print(f"[data] ff_rows={len(ff)} siw_rows={len(siw)} batches={len(loader)}")

    batch = next(iter(loader))
    print(f"[data] frames={tuple(batch['frames'].shape)} dtype={batch['frames'].dtype}")
    assert batch["frames"].shape[1] == tc.num_frames, "T mismatch"

    # ── Temporal consistency of augmentation ───────────────────────────
    # Albumentations' `images` target must apply ONE parameter draw to the whole
    # clip. Per-frame augmentation would inject artificial frame-to-frame
    # inconsistency — the exact signal the temporal head is trained to measure.
    ds = T.MTLDataset(ff, cfg, is_train=True)
    clip = ds.clips[0]
    paths = clip["frame_paths"][:tc.num_frames]
    raw = np.stack([np.array(T.Image.open(p).convert("RGB")) for p in paths])
    aug = ds.transform(images=list(raw))["images"]
    aug = torch.stack(list(aug)).numpy()
    raw_d = np.abs(np.diff(raw.astype(np.float32) / 255.0, axis=0)).mean()
    aug_d = np.abs(np.diff(aug, axis=0)).mean() / 4.0   # /4 ≈ undo Normalize std
    print(f"[aug ] mean |Δframe| raw={raw_d:.4f} augmented={aug_d:.4f} "
          f"ratio={aug_d / max(raw_d, 1e-9):.2f}x")
    assert aug_d < raw_d * 4, "augmentation looks temporally inconsistent"

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = T.Trainer(cfg, logger)
    model = trainer.model
    print(f"[amp ] use_amp={trainer.use_amp} scaler_enabled={trainer.scaler.is_enabled()} "
          f"pcgrad={trainer.use_pcgrad} gradnorm={trainer.use_gradnorm}")

    # TSM must fire on intermediate blocks, not raw RGB.
    hits = []
    orig_tsm = T.apply_tsm

    def spy(x, r, t):
        hits.append(tuple(x.shape[1:]))
        return orig_tsm(x, r, t)
    T.apply_tsm = spy

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    w_before = model.task_weights.detach().clone()

    # Force GradNorm's expensive path to run at step 0. Its first call only
    # seeds the L(0) baseline and early-returns, so on a clean checkpoints/ the
    # 3-task gradient-norm computation is never reached — which is precisely how
    # an OOM there survived: it only fires from the *second* update onward, i.e.
    # step 10 of epoch 0 in a real run.
    if trainer.gradnorm_manager.initial_losses is None:
        trainer.gradnorm_manager.initial_losses = torch.tensor(
            [0.7, 0.13, 1.9], device=dev)
        print("[gn  ] pre-seeded L(0) so step 0 takes the full GradNorm path")

    model.train()
    tc_accum = tc.grad_accum_steps
    stepped = 0
    for step, b in enumerate(loader):
        if step >= STEPS:
            break
        frames = b["frames"].to(dev, non_blocking=True)
        df_l = b["deepfake_label"].to(dev, non_blocking=True).float()
        sp_l = b["spoof_label"].to(dev, non_blocking=True).float()
        tp_l = b["temporal_label"].to(dev, non_blocking=True).float()

        with trainer.autocast_ctx:
            out = model(frames)
            l_df = trainer.criterion_df(out["deepfake_logit"], df_l)
            l_sp = trainer.criterion_sp(out["spoof_logit"], sp_l)
            l_tp = T.temporal_consistency_loss(
                out["temp_proj"], tp_l,
                logit=out["temp_logit"],
                logit_weight=trainer.temporal_logit_weight,
            )
            w = model.task_weights

        run_gn = trainer.use_gradnorm and (step % tc.gradnorm_interval == 0)

        if trainer.use_pcgrad:
            T.compute_pcgrad_grads(
                [w[0] * l_df, w[1] * l_sp, w[2] * l_tp],
                trainer.optim_params,
                scaler=trainer.scaler if trainer.use_amp else None,
                accum_factor=1.0 / tc_accum,
                retain_graph_last=run_gn,
            )
        else:
            trainer.scaler.scale(
                (w[0] * l_df + w[1] * l_sp + w[2] * l_tp) / tc_accum
            ).backward(retain_graph=run_gn)

        if run_gn:
            # GradNorm must not touch backbone .grad — snapshot and compare.
            snap = {n: (p.grad.detach().clone() if p.grad is not None else None)
                    for n, p in model.backbone.named_parameters()}
            new_w = trainer.gradnorm_manager.update(
                torch.stack([l_df, l_sp, l_tp]),
                list(model.backbone.parameters()),
            )
            drift = 0.0
            for n, p in model.backbone.named_parameters():
                a, bb = snap[n], p.grad
                if a is None and bb is None:
                    continue
                if a is None or bb is None:
                    drift = float("inf")
                    break
                drift = max(drift, (a - bb).abs().max().item())
            print(f"[gn  ] step {step}: backbone .grad drift={drift:.3e} "
                  f"weights={new_w.detach().cpu().numpy().round(4)}")
            assert drift == 0.0, "GradNorm polluted backbone gradients"
            with torch.no_grad():
                model.log_weights.copy_(torch.log(new_w.clamp(min=1e-8)))

        if (step + 1) % tc_accum == 0:
            trainer.scaler.unscale_(trainer.optimizer)
            gn = torch.nn.utils.clip_grad_norm_(trainer.optim_params, tc.max_grad_norm)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            trainer.optimizer.zero_grad(set_to_none=True)
            stepped += 1
            print(f"[step] {step}: |grad|={gn.item():.4f} "
                  f"scale={trainer.scaler.get_scale():.0f} "
                  f"loss(df/sp/tp)={l_df.item():.4f}/{l_sp.item():.4f}/{l_tp.item():.4f}")
            # clip_grad_norm_ must see the *unscaled* norm. log_weights is not in
            # the optimizer, so unscale_() skips it; clipping over
            # model.parameters() measured a norm inflated by the AMP scale and
            # shrank every real gradient by ~1e-4.
            assert gn.item() < 1e3, (
                f"gradient norm {gn.item():.1f} looks AMP-scaled — "
                f"clipping and unscaling must cover the same parameter set")

    T.apply_tsm = orig_tsm

    # ── Assertions ─────────────────────────────────────────────────────
    print(f"\n[tsm ] applied at feature shapes: {sorted(set(hits))}")
    assert hits, "TSM never fired"
    assert all(c[0] != 3 for c in hits), "TSM applied to raw RGB input"

    changed, total, nonfinite, static = 0, 0, [], []
    opt_ids = {id(p) for p in trainer.optim_params}
    for n, p in model.named_parameters():
        total += 1
        if not torch.equal(before[n], p.detach()):
            changed += 1
        elif id(p) in opt_ids:
            static.append(n)
        if not torch.isfinite(p.detach()).all():
            nonfinite.append(n)
    print(f"[upd ] optimizer steps={stepped}  params changed={changed}/{total}")
    print(f"[upd ] task weights {w_before.cpu().numpy().round(4)} -> "
          f"{model.task_weights.detach().cpu().numpy().round(4)}")
    assert stepped > 0, "no optimizer step ran"
    assert not nonfinite, f"non-finite params: {nonfinite[:5]}"

    # ── Gradient coverage ──────────────────────────────────────────────
    # The right test for "is this parameter learning", *not* whether its value
    # moved. At epoch 0 the LinearLR warmup puts the backbone at 3e-9 (1e-4 ×
    # 3e-5), and an update that small is ~80× below float32 spacing at a
    # BatchNorm weight of ~2.3 — it rounds straight back to the old value. So a
    # param can be perfectly healthy and still compare equal. `grad is None` is
    # unambiguous: nothing in the loss reaches it.
    dead = grad_coverage(trainer, model, next(iter(loader)), dev)
    print(f"[grad] optimizer params with grad=None: {len(dead)}")
    if dead:
        print(f"       {dead}")
    flow_dead = [n for n in dead if "flow_encoder" in n]
    other_dead = [n for n in dead if "flow_encoder" not in n]
    assert not other_dead, f"parameters receive no gradient: {other_dead}"
    if flow_dead:
        print(f"       ({len(flow_dead)} are flow_encoder — unreachable until "
              f"optical flow is fed to TemporalHead)")
    if static:
        # Exclude the ones already reported as receiving no gradient — calling
        # those "alive" would contradict the line above.
        alive = [n for n in static if n not in set(dead)]
        if alive:
            print(f"[upd ] unchanged but alive ({len(alive)}): warmup lr under "
                  f"float32 spacing, e.g. {alive[:3]}")

    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    print(f"[vram] peak allocated={peak:.0f} MiB  reserved={reserved:.0f} MiB "
          f"(base {base_mem:.0f} MiB)")

    # ── validate(): the metric calculation path ────────────────────────
    ff_v = subsample(T.load_csv(cfg.paths.ff_val_csv), CLIPS)
    siw_v = subsample(T.load_csv(cfg.paths.siwmv2_val_csv), CLIPS)
    val_loader = T.build_interleaved_loader(ff_v, siw_v, cfg, is_train=False)
    metrics = trainer.validate(val_loader)
    print(f"\n[val ] {len(metrics)} metrics over {len(ff_v) + len(siw_v)} rows")
    want = ["df_auc_roc", "df_eer", "df_ap", "df_acc_best_thresh",
            "sp_apcer", "sp_bpcer", "sp_acer", "sp_hter", "sp_tpr_at_fpr1",
            "tmp_bin_acc", "tmp_auc"]
    missing = [k for k in want if k not in metrics]
    for k in want:
        v = metrics.get(k)
        print(f"       {k:20} = {v if v is None else round(float(v), 4)}"
              f"{'   <-- MISSING' if k in missing else ''}")
    assert not missing, f"metrics missing: {missing}; got {sorted(metrics)}"
    bad = [k for k, v in metrics.items()
           if isinstance(v, float) and not np.isfinite(v)]
    assert not bad, f"non-finite metrics: {bad}"
    print(f"       also returned: {sorted(set(metrics) - set(want))}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
