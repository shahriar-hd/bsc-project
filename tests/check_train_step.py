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
    tid = batch["task_id"].to(dev)
    m_df = tid == T.TASK_DEEPFAKE
    m_sp = tid == T.TASK_SPOOF
    # Mirrors _train_epoch: present only when use_optical_flow is on AND
    # preprocessing wrote the .npz files.
    flow = batch.get("flow")
    if flow is not None:
        flow = flow.to(dev)
    with trainer.autocast_ctx:
        out = model(frames, flow=flow)
        loss = (trainer.criterion_df(out["deepfake_logit"].flatten()[m_df], df_l[m_df])
                + trainer.criterion_sp(out["spoof_logit"].flatten()[m_sp], sp_l[m_sp])
                + T.temporal_consistency_loss(
                    out["temp_proj"], tp_l,
                    logit=out["temp_logit"],
                    logit_weight=trainer.temporal_logit_weight,
                    flow_score=out.get("flow_consistency"),
                    flow_weight=trainer.flow_loss_weight))
    trainer.scaler.scale(loss).backward()
    ids = {id(p) for p in trainer.optim_params}
    dead = [n for n, p in model.named_parameters()
            if id(p) in ids and p.grad is None]
    trainer.optimizer.zero_grad(set_to_none=True)
    return dead


def check_task_mask(trainer, model, batch, dev):
    """Each supervised loss must reach only the samples that own its label.

    Tested at the input, not at the loss value: the gradient of the masked spoof
    loss w.r.t. `frames` has to be exactly zero on every FaceForensics++ row and
    non-zero on every SiW-Mv2 row. Comparing loss *values* would not catch it —
    run01's unmasked `l_sp` was a perfectly finite number, computed against a
    placeholder zero that taught the spoof head "a deepfake face is bona-fide".

    Runs with the backbone's BatchNorm layers switched to their running
    statistics, which is what makes the measurement mean anything. With BN in
    training mode the normalisation pools statistics over the whole batch, so
    *every* sample's activations depend on every other sample's and the spoof
    loss shows a large gradient on FaceForensics++ frames even when the mask is
    perfect (measured: 536 and 762 against 2350 and 426 on the SiW rows). That
    coupling is inherent to BatchNorm on a mixed-task batch, not a labelling
    error. Only the BN layers are switched — `model.eval()` would also disable
    gradient checkpointing (`grad_checkpointing and self.training` in
    forward_backbone), and the un-checkpointed float32 backward OOMs a 4 GB card.
    """
    tid = batch["task_id"].to(dev)
    m_df = tid == T.TASK_DEEPFAKE
    m_sp = tid == T.TASK_SPOOF
    print(f"[mask] batch task_id={tid.tolist()} "
          f"({int(m_df.sum())} deepfake + {int(m_sp.sum())} spoof)")
    assert m_df.any() and m_sp.any(), (
        "the batch is single-task — InterleavedBatchSampler must put both in "
        "every batch or the masked losses have nothing to reduce over")

    sp_l = batch["spoof_label"].to(dev).float()
    df_l = batch["deepfake_label"].to(dev).float()

    bns = [m for m in model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    bn_was_training = [m.training for m in bns]
    for m in bns:
        m.eval()
    torch.cuda.empty_cache()
    print(f"[mask] {len(bns)} BatchNorm layers on running stats for this check "
          f"(batch statistics would couple all samples regardless of the mask)")

    def per_sample_grad(masked_loss_fn):
        frames = batch["frames"].to(dev).clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        # float32 on purpose: under float16 autocast the small per-sample
        # gradients underflow to zero everywhere, which would make the
        # "exactly 0.0" assertion below pass vacuously.
        out = model(frames)
        masked_loss_fn(out).backward()
        g = frames.grad.detach().abs().flatten(1).sum(1)
        model.zero_grad(set_to_none=True)
        del frames, out
        torch.cuda.empty_cache()
        return g

    try:
        g_sp = per_sample_grad(
            lambda o: trainer.criterion_sp(o["spoof_logit"].flatten()[m_sp], sp_l[m_sp]))
        g_df = per_sample_grad(
            lambda o: trainer.criterion_df(o["deepfake_logit"].flatten()[m_df], df_l[m_df]))
        tp_l = batch["temporal_label"].to(dev).float()
        g_tp = per_sample_grad(
            lambda o: T.temporal_consistency_loss(
                o["temp_proj"], tp_l, logit=o["temp_logit"],
                logit_weight=trainer.temporal_logit_weight))
    finally:
        for m, was in zip(bns, bn_was_training):
            m.train(was)

    # Scientific notation: focal loss on a confidently-correct sample leaves a
    # gradient around 1e-10, which prints as "0.0" at fixed precision and would
    # read as though the mask had killed it.
    def fmt(g):
        return "[" + ", ".join(f"{v:.2e}" for v in g.cpu().tolist()) + "]"

    print(f"[mask] |d l_sp / d frames| per sample = {fmt(g_sp)}")
    print(f"[mask] |d l_df / d frames| per sample = {fmt(g_df)}")
    assert float(g_sp[m_df].abs().max()) == 0.0, (
        "the spoof loss reaches FaceForensics++ samples — task mask is not applied")
    assert float(g_sp[m_sp].min()) > 0.0, (
        "the spoof loss does not reach its own SiW-Mv2 samples")
    assert float(g_df[m_sp].abs().max()) == 0.0, (
        "the deepfake loss reaches SiW-Mv2 samples — task mask is not applied")
    assert float(g_df[m_df].min()) > 0.0, (
        "the deepfake loss does not reach its own FaceForensics++ samples")

    # The temporal head is supervised over the whole batch by design: its label
    # ("inauthentic") is genuine ground truth in both datasets.
    print(f"[mask] |d l_temp / d frames| per sample = {fmt(g_tp)}")
    assert float(g_tp.min()) > 0.0, (
        "the temporal loss must cover every sample — it is not masked by task")


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
    # Both tasks in every batch is what makes the masked losses well defined;
    # check_sampler_balance.py asserts the quota over 200 batches.
    tid0 = batch["task_id"]
    assert (tid0 == T.TASK_DEEPFAKE).any() and (tid0 == T.TASK_SPOOF).any(), \
        f"first batch is single-task: task_id={tid0.tolist()}"

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
    overflows = 0
    task_mix = []
    for step, b in enumerate(loader):
        if step >= STEPS:
            break
        frames = b["frames"].to(dev, non_blocking=True)
        df_l = b["deepfake_label"].to(dev, non_blocking=True).float()
        sp_l = b["spoof_label"].to(dev, non_blocking=True).float()
        tp_l = b["temporal_label"].to(dev, non_blocking=True).float()
        tid = b["task_id"].to(dev, non_blocking=True)
        m_df = tid == T.TASK_DEEPFAKE
        m_sp = tid == T.TASK_SPOOF
        task_mix.append((int(m_df.sum()), int(m_sp.sum())))

        with trainer.autocast_ctx:
            out = model(frames)
            # Masked exactly as _train_epoch does — an unmasked l_sp here would
            # let this script pass while the real loop trains on placeholders.
            l_df = trainer.criterion_df(out["deepfake_logit"].flatten()[m_df],
                                        df_l[m_df])
            l_sp = trainer.criterion_sp(out["spoof_logit"].flatten()[m_sp],
                                        sp_l[m_sp])
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
            overflow = not np.isfinite(gn.item())
            overflows += int(overflow)
            print(f"[step] {step}: |grad|={gn.item():.4f} "
                  f"scale={trainer.scaler.get_scale():.0f} "
                  f"loss(df/sp/tp)={l_df.item():.4f}/{l_sp.item():.4f}/{l_tp.item():.4f}"
                  f"{'  (float16 overflow — scaler skips this step)' if overflow else ''}")
            # clip_grad_norm_ must see the *unscaled* norm. log_weights is not in
            # the optimizer, so unscale_() skips it; clipping over
            # model.parameters() measured a norm inflated by the AMP scale and
            # shrank every real gradient by ~1e-4.
            #
            # A non-finite norm is a different thing: some gradient overflowed
            # float16 in the backward pass, and inf/scale is still inf. That is
            # what GradScaler is for — it skips the step and halves the scale, so
            # it is not a failure. Only a *finite* but inflated norm indicates the
            # clip/unscale parameter sets have drifted apart.
            assert overflow or gn.item() < 1e3, (
                f"gradient norm {gn.item():.1f} looks AMP-scaled — "
                f"clipping and unscaling must cover the same parameter set")

    T.apply_tsm = orig_tsm

    # ── Assertions ─────────────────────────────────────────────────────
    print(f"\n[mask] task mix per step (deepfake, spoof): {task_mix}")
    assert all(a > 0 and s > 0 for a, s in task_mix), (
        f"a step ran with a single-task batch: {task_mix} — the masked loss for "
        f"the missing task would have no samples to reduce over")

    print(f"[tsm ] applied at feature shapes: {sorted(set(hits))}")
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
    assert overflows < stepped, (
        f"every one of {stepped} steps overflowed float16 — the scaler skipped "
        f"them all, so nothing was learned. Check amp_dtype and the loss scale")
    if overflows:
        print(f"[amp ] {overflows}/{stepped} steps overflowed and were skipped by "
              f"the scaler (normal for float16; the scale halves each time)")
    assert not nonfinite, f"non-finite params: {nonfinite[:5]}"

    # ── Task mask ──────────────────────────────────────────────────────
    check_task_mask(trainer, model, next(iter(loader)), dev)

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
    # flow_encoder is reached only by the flow BCE term, which is only active
    # when precomputed flow was actually loaded. Assert both directions so
    # neither the baseline nor the flow-enabled run can silently regress.
    flow_live = trainer.flow_loss_weight > 0.0 and any(
        getattr(d, "use_flow", False)
        for d in getattr(loader.dataset, "datasets", [loader.dataset])
    )
    if flow_live:
        assert not flow_dead, (
            f"optical flow is loaded but flow_encoder gets no gradient: {flow_dead}"
        )
        print("[grad] flow_encoder is receiving gradient (flow enabled)")
    elif flow_dead:
        print(f"       ({len(flow_dead)} are flow_encoder — no flow loaded, so "
              f"unreachable. Expected with use_optical_flow=False)")
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
    # The threshold block is the half run01 did not log. `recall`/`f1`/`mcc` were
    # 0.0 for the spoof head from epoch 2; the ACER that *was* logged sat at
    # exactly 0.5, the value a coin flip produces.
    shared = ["acc", "balanced_acc", "precision", "recall", "specificity", "f1",
              "mcc", "auc_roc", "ap", "tn", "fp", "fn", "tp", "n_pos", "n_neg",
              "pred_pos_rate", "score_min", "score_max",
              "score_mean_pos", "score_mean_neg"]
    want = ([f"df_{k}" for k in shared] + ["df_eer", "df_acc_best_thresh"]
            + [f"sp_{k}" for k in shared]
            + ["sp_apcer", "sp_bpcer", "sp_acer", "sp_hter", "sp_tpr_at_fpr1"]
            + [f"tmp_{k}" for k in shared] + ["tmp_bin_acc"])
    missing = [k for k in want if k not in metrics]
    head = ["df_auc_roc", "df_recall", "df_f1", "df_mcc", "df_eer",
            "sp_auc_roc", "sp_recall", "sp_f1", "sp_mcc", "sp_acer",
            "sp_pred_pos_rate", "sp_score_max",
            "tmp_auc_roc", "tmp_bin_acc"]
    for k in head + missing:
        v = metrics.get(k)
        print(f"       {k:20} = {v if v is None else round(float(v), 4)}"
              f"{'   <-- MISSING' if k in missing else ''}")
    assert not missing, f"metrics missing: {missing}; got {sorted(metrics)}"

    # Per-attack-type recall — the table that says whether a *type* needs more
    # data rather than the whole dataset.
    per_type = sorted(k for k in metrics if k.startswith("sp_recall_"))
    print(f"       per-type recall keys ({len(per_type)}): "
          f"{[k.replace('sp_recall_', '') for k in per_type]}")
    assert per_type, (
        "no sp_recall_<type> keys — spoof_type is not reaching compute_spoof_metrics")
    for k in per_type:
        assert f"sp_n_{k[len('sp_recall_'):]}" in metrics, f"{k} has no support count"

    bad = [k for k, v in metrics.items()
           if isinstance(v, float) and not np.isfinite(v)]
    assert not bad, f"non-finite metrics: {bad}"
    # Masking must also apply to the metric side: a mixed loader used to emit
    # ff_sp_acer and friends, computed against placeholder labels.
    assert metrics["df_n_pos"] + metrics["df_n_neg"] <= len(ff_v) / tc.num_frames + 1, \
        "deepfake metrics were accumulated over SiW-Mv2 samples too"
    assert metrics["sp_n_pos"] + metrics["sp_n_neg"] <= len(siw_v) / tc.num_frames + 1, \
        "spoof metrics were accumulated over FaceForensics++ samples too"
    print(f"       support: df n={metrics['df_n_pos']}+{metrics['df_n_neg']}  "
          f"sp n={metrics['sp_n_pos']}+{metrics['sp_n_neg']}  "
          f"tmp n={metrics['tmp_n_pos']}+{metrics['tmp_n_neg']}")
    print(f"       also returned: {sorted(set(metrics) - set(want) - set(per_type))}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
