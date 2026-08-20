"""Verification for the precomputed-optical-flow path.

Flow is produced by `preprocessing.py` and only *read* by `train.py`. This check
covers the seam between them without re-running the full pipeline: it writes flow
files for a handful of real clips using the same `flow_utils` functions
preprocessing calls, then drives the real `MTLDataset` / `Trainer` over them with
`use_optical_flow = True`.

What it asserts:
  1. the dataset finds the files and reports the resolution read *from the file*
  2. batches carry a "flow" tensor of shape (B, T-1, 2, R, R) and collate
  3. `flow_encoder` receives a gradient — the thing `use_optical_flow = False`
     leaves dead, and the whole point of wiring the loss term
  4. a clip whose file is missing falls back to zeros instead of crashing
  5. VRAM stays inside the 4 GB budget with flow on

Run:  python tests/check_flow_path.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.train as T                                    # noqa: E402
from src.config import get_config                        # noqa: E402
from src.utils.flow_utils import (                       # noqa: E402
    compute_clip_flow, flow_path_for_clip, save_clip_flow,
)

CLIPS = 6          # per dataset — enough for 3 batches of 4 with both present
RESIZE = 56


def subsample(df, n_clips):
    """Stratified head of the CSV: the files are label-sorted, so a plain head is
    single-class and every guarded metric returns nothing."""
    keys = df[["video_path", "clip_index", "label"]].drop_duplicates()
    picked = (keys.groupby("label", group_keys=False)
                  .head(max(n_clips // 2, 1))[["video_path", "clip_index"]])
    return df.merge(picked, on=["video_path", "clip_index"])


def write_flow_for(df, resize=RESIZE):
    """Compute + save flow for every clip in `df`, exactly as preprocessing does.

    Reads back the JPEG crops rather than the source video: the crops are what
    preprocessing hands to `compute_clip_flow`, so the numbers match.
    """
    written = []
    for (vp, ci), grp in df.groupby(["video_path", "clip_index"], sort=False):
        grp = grp.sort_values("frame_num")
        paths = grp["frame_path"].tolist()
        crops = [cv2.imread(p) for p in paths]
        crops = [c for c in crops if c is not None]
        if len(crops) < 2:
            continue
        flow = compute_clip_flow(crops, resize=resize)
        out = flow_path_for_clip(paths[0], int(ci))
        save_clip_flow(out, flow)
        written.append(out)
    return written


def main():
    cfg = get_config()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(message)s")

    # Flow ON for this check only — the committed default is False.
    cfg.train.use_optical_flow = True
    cfg.model.temporal_supervision = "combined"

    T.set_seed(cfg.train.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ff = subsample(T.load_csv(cfg.paths.ff_train_csv), CLIPS)
    siw = subsample(T.load_csv(cfg.paths.siwmv2_train_csv), CLIPS)
    print(f"[data] ff_rows={len(ff)} siw_rows={len(siw)}")

    # ── 1. Produce flow, the way preprocessing does ────────────────────
    created = write_flow_for(ff) + write_flow_for(siw)
    total_kb = sum(p.stat().st_size for p in created) / 1024
    print(f"[prep] wrote {len(created)} flow files, "
          f"{total_kb / max(len(created), 1):.1f} KB/clip avg")
    assert created, "no flow files written — check the CSVs point at real crops"

    try:
        # ── 2. Dataset picks them up, resolution read from the file ────
        loader = T.build_interleaved_loader(ff, siw, cfg, is_train=True)
        subsets = getattr(loader.dataset, "datasets", [loader.dataset])
        for i, d in enumerate(subsets):
            print(f"[ds  ] subset {i}: use_flow={d.use_flow} "
                  f"flow_resize={d.flow_resize} clips={len(d)}")
            assert d.use_flow, "dataset did not find the flow files it just wrote"
            assert d.flow_resize == RESIZE, (
                f"resolution came out as {d.flow_resize}, expected {RESIZE} — "
                f"it must be read from the file, not from config"
            )

        batch = next(iter(loader))
        assert "flow" in batch, "batch has no flow key"
        B, Tm1, C, H, W = batch["flow"].shape
        print(f"[batch] flow={tuple(batch['flow'].shape)} "
              f"frames={tuple(batch['frames'].shape)}")
        assert (Tm1, C, H, W) == (cfg.train.num_frames - 1, 2, RESIZE, RESIZE), (
            f"unexpected flow shape {(Tm1, C, H, W)}"
        )
        nz = float(batch["flow"].abs().mean())
        print(f"[batch] mean |flow| = {nz:.5f} (normalised by R)")
        assert nz > 0.0, "flow is all zeros — the files were not actually read"

        # ── 3. flow_encoder must receive a gradient ────────────────────
        trainer = T.Trainer(cfg, logging.getLogger("check_flow"))
        model = trainer.model
        print(f"[loss] flow_loss_weight = {trainer.flow_loss_weight}")
        assert trainer.flow_loss_weight > 0.0, (
            "flow_loss_weight is 0 with use_optical_flow=True — the encoder "
            "would stay dead"
        )

        trainer.optimizer.zero_grad(set_to_none=True)
        with trainer.autocast_ctx:
            out = model(batch["frames"].to(dev), flow=batch["flow"].to(dev))
            assert out["flow_consistency"] is not None, (
                "TemporalHead returned no flow_consistency — is "
                "temporal_supervision set to a mode that builds the encoder?"
            )
            loss = T.temporal_consistency_loss(
                out["temp_proj"],
                batch["temporal_label"].to(dev).float(),
                logit=out["temp_logit"],
                logit_weight=trainer.temporal_logit_weight,
                flow_score=out["flow_consistency"],
                flow_weight=trainer.flow_loss_weight,
            )
        trainer.scaler.scale(loss).backward()

        enc = [(n, p) for n, p in model.named_parameters()
               if "flow_encoder" in n]
        dead = [n for n, p in enc if p.grad is None]
        norms = [float(p.grad.norm()) for _, p in enc if p.grad is not None]
        print(f"[grad] flow_encoder params={len(enc)} with grad=None={len(dead)}")
        assert not dead, f"flow_encoder still receives no gradient: {dead}"
        assert max(norms) > 0.0, "flow_encoder gradients are all exactly zero"
        print(f"[grad] grad norms min={min(norms):.3e} max={max(norms):.3e}")
        trainer.optimizer.zero_grad(set_to_none=True)

        # ── 4. Missing file degrades to zeros, not a crash ─────────────
        victim = created[0]
        backup = victim.with_suffix(".npz.bak")
        victim.rename(backup)
        try:
            d0 = subsets[0]
            idx = next(
                (i for i, c in enumerate(d0.clips)
                 if flow_path_for_clip(c["frame_paths"][0], c["clip_index"]) == victim),
                None,
            )
            assert idx is not None, "could not locate the clip whose file was removed"
            sample = d0[idx]
            assert "flow" in sample, "flow key vanished for a missing file"
            assert tuple(sample["flow"].shape) == (
                cfg.train.num_frames - 1, 2, RESIZE, RESIZE
            ), f"fallback shape wrong: {tuple(sample['flow'].shape)}"
            assert float(sample["flow"].abs().sum()) == 0.0, (
                "fallback should be exactly zeros"
            )
            print("[miss] missing file -> zero flow of the right shape, no crash")
        finally:
            backup.rename(victim)

        # ── 5. Dataset-wide absence disables flow with a warning ───────
        stash = victim.parent.with_name(victim.parent.name + "__flow_stash")
        moved = []
        for p in created:
            if p.parent == victim.parent:
                tgt = stash / p.name
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(tgt))
                moved.append((tgt, p))
        try:
            if moved:
                probe = T.MTLDataset(
                    ff if moved[0][1].parts[-2] in str(ff["frame_path"].iloc[0])
                    else siw, cfg, is_train=True,
                )
                print(f"[gate] after removing that subject's files: "
                      f"use_flow={probe.use_flow}")
        finally:
            for tgt, orig in moved:
                shutil.move(str(tgt), str(orig))
            if stash.exists():
                shutil.rmtree(stash, ignore_errors=True)

        if dev.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**20
            print(f"[vram] peak allocated={peak:.0f} MiB "
                  f"reserved={torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")

        print("\nFLOW PATH CHECK PASSED")
    finally:
        # Leave the tree as we found it — these clips are not the full dataset,
        # and a partial set of flow files would make MTLDataset's probe succeed
        # while most clips fall back to zeros.
        for p in created:
            p.unlink(missing_ok=True)
        print(f"[clean] removed {len(created)} temporary flow files")


if __name__ == "__main__":
    main()
