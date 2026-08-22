"""Verification for the per-batch task quota — the run01 root cause.

run01 trained 14 epochs without a single anti-spoof sample. The loader built a
`WeightedRandomSampler` whose SiW-Mv2 weight was `(1 - ff_sample_ratio) / n_siw`,
and `ff_sample_ratio` was 1.0, so every SiW clip had weight exactly 0.0.
`torch.multinomial` never draws a zero-weight index. `loss_sp` was 0.0 from epoch
2 onward and the spoof head's output ceiling ended at 0.4978 — below the demo's
0.6 threshold, which is why the webcam branch could never light up.

This script asserts the replacement cannot fail the same way:

    python tests/check_sampler_balance.py     # ~40 s, no model, low VRAM

Three layers:
  1. the old weights, replayed, to record *that* the bug was real and is gone
  2. `InterleavedBatchSampler` over the real dataset sizes for 200 batches —
     exact quota, no index out of range, full coverage, epoch-dependent order
  3. ~12 real batches through the real DataLoader/MTLDataset, so the seam where
     `task_id` collates into a tensor is covered too
"""
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import src.train as T
from src.config import get_config

BATCHES = 200      # index-level scan; the sampler recycles well before this
REAL_BATCHES = 12  # batches actually decoded through MTLDataset
CLIPS = 24         # clips per label, per dataset, for the real-loader pass


def subsample(df, n_clips):
    """Keep n_clips whole clips per label (the CSVs are label-sorted)."""
    out = []
    for _, grp in df.groupby("label", sort=False):
        keys = grp.groupby(["video_path", "clip_index"], sort=False).ngroup()
        out.append(grp[keys < max(n_clips // 2, 1)])
    return pd.concat(out).copy()


def replay_old_sampler(n_ff, n_siw, ratio, draws=4000):
    """The pre-fix weighting, replayed. Returns how many SiW indices it drew."""
    w = torch.cat([
        torch.full((n_ff,), ratio / max(n_ff, 1)),
        torch.full((n_siw,), (1.0 - ratio) / max(n_siw, 1)),
    ])
    idx = torch.multinomial(w, draws, replacement=True)
    return int((idx >= n_ff).sum()), float(w[n_ff].item())


def main():
    cfg = get_config()
    tc = cfg.train
    T.set_seed(tc.seed)

    ff_rows = T.load_csv(cfg.paths.ff_train_csv)
    siw_rows = T.load_csv(cfg.paths.siwmv2_train_csv)
    n_ff_clips = ff_rows.groupby(["video_path", "clip_index"], sort=False).ngroups
    n_siw_clips = siw_rows.groupby(["video_path", "clip_index"], sort=False).ngroups
    print(f"[data] train clips: ff={n_ff_clips} siw={n_siw_clips}")

    bs = tc.batch_size
    ff_pb = max(1, min(bs - 1, int(round(tc.ff_sample_ratio * bs))))
    siw_pb = bs - ff_pb
    print(f"[conf] batch_size={bs} ff_sample_ratio={tc.ff_sample_ratio} "
          f"-> {ff_pb} FF + {siw_pb} SiW per batch")
    assert ff_pb >= 1 and siw_pb >= 1, "a task got zero slots per batch"

    # ── 1. the old failure mode, on the record ─────────────────────────
    n_siw_drawn, siw_w = replay_old_sampler(n_ff_clips, n_siw_clips, 1.0)
    print(f"[old ] WeightedRandomSampler at ratio=1.0: siw weight={siw_w} "
          f"-> {n_siw_drawn}/4000 SiW draws")
    assert n_siw_drawn == 0, (
        "the old sampler drew SiW samples here, so this script is not "
        "reproducing run01's configuration")

    # Any ratio must now keep both tasks, including the two degenerate ends.
    for r in (0.0, 0.01, 0.5, 0.99, 1.0):
        pb = max(1, min(bs - 1, int(round(r * bs))))
        assert 1 <= pb <= bs - 1, f"ratio {r} clamps to {pb} slots"
    print(f"[conf] clamp holds for ratio in 0.0..1.0 (never 0 or {bs} FF slots)")

    # ── 2. the quota itself, over many batches ─────────────────────────
    sampler = T.InterleavedBatchSampler(
        n_ff=n_ff_clips, n_siw=n_siw_clips,
        batch_size=bs, ff_per_batch=ff_pb, seed=tc.seed,
    )
    expect_len = max(-(-n_ff_clips // ff_pb), -(-n_siw_clips // siw_pb))
    print(f"[samp] len={len(sampler)} (expected {expect_len} = "
          f"max(ceil({n_ff_clips}/{ff_pb}), ceil({n_siw_clips}/{siw_pb})))")
    assert len(sampler) == expect_len, "epoch length is not the larger dataset"

    seen_ff, seen_siw = set(), set()
    sizes = Counter()
    for i, batch in enumerate(sampler):
        n_ff = sum(1 for j in batch if j < n_ff_clips)
        n_siw = len(batch) - n_ff
        sizes[(n_ff, n_siw)] += 1
        assert len(batch) == bs, f"batch {i} has {len(batch)} samples, want {bs}"
        assert (n_ff, n_siw) == (ff_pb, siw_pb), (
            f"batch {i} is {n_ff} FF + {n_siw} SiW, want {ff_pb} + {siw_pb}")
        assert len(set(batch)) == len(batch), f"batch {i} repeats an index"
        assert min(batch) >= 0 and max(batch) < n_ff_clips + n_siw_clips, (
            f"batch {i} indexes outside the ConcatDataset")
        seen_ff.update(j for j in batch if j < n_ff_clips)
        seen_siw.update(j - n_ff_clips for j in batch if j >= n_ff_clips)
        if i + 1 >= BATCHES:
            break
    print(f"[samp] {sum(sizes.values())} batches, composition seen: {dict(sizes)}")

    # One full epoch must show every clip of both datasets at least once — the
    # smaller one by wrapping around, not by the larger one being truncated.
    full_ff, full_siw = set(), set()
    for batch in sampler:
        full_ff.update(j for j in batch if j < n_ff_clips)
        full_siw.update(j - n_ff_clips for j in batch if j >= n_ff_clips)
    print(f"[samp] one epoch covers ff={len(full_ff)}/{n_ff_clips} "
          f"siw={len(full_siw)}/{n_siw_clips}")
    assert len(full_ff) == n_ff_clips, "some FF++ clips are never sampled"
    assert len(full_siw) == n_siw_clips, "some SiW-Mv2 clips are never sampled"

    first = next(iter(sampler))
    sampler.set_epoch(1)
    second = next(iter(sampler))
    print(f"[samp] set_epoch changes order: {first[:4]} -> {second[:4]}")
    assert first != second, "set_epoch did not reshuffle"
    sampler.set_epoch(0)
    assert next(iter(sampler)) == first, "epoch 0 is not reproducible"

    # An empty dataset must fail loudly here rather than produce silent all-FF
    # batches further down.
    for bad in ((0, n_siw_clips), (n_ff_clips, 0)):
        try:
            T.InterleavedBatchSampler(bad[0], bad[1], bs, ff_pb, tc.seed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"empty dataset accepted: n_ff/n_siw={bad}")
    print("[samp] empty dataset raises ValueError")

    # ── 3. through the real loader ─────────────────────────────────────
    ff = subsample(ff_rows, CLIPS)
    siw = subsample(siw_rows, CLIPS)
    loader = T.build_interleaved_loader(ff, siw, cfg, is_train=True)
    task_counts, ds_counts = Counter(), Counter()
    for i, b in enumerate(loader):
        tid = b["task_id"]
        assert isinstance(tid, torch.Tensor) and tid.dtype == torch.long, (
            f"task_id collated as {type(tid)} — it must be a long tensor to "
            f"index the loss masks")
        n_df = int((tid == T.TASK_DEEPFAKE).sum())
        n_sp = int((tid == T.TASK_SPOOF).sum())
        assert (n_df, n_sp) == (ff_pb, siw_pb), (
            f"real batch {i}: {n_df} deepfake + {n_sp} spoof, want "
            f"{ff_pb} + {siw_pb}")
        # task_id and the dataset string must agree, or the mask points at the
        # wrong rows while still looking balanced.
        for k, name in enumerate(b["dataset"]):
            want = T.TASK_DEEPFAKE if name == cfg.preprocess.ff_dataset_name \
                else T.TASK_SPOOF
            assert int(tid[k]) == want, (
                f"real batch {i} sample {k}: dataset={name} but task_id="
                f"{int(tid[k])}")
        task_counts.update(b["task"])
        ds_counts.update(b["dataset"])
        if i + 1 >= REAL_BATCHES:
            break
    print(f"[real] {REAL_BATCHES} batches: tasks={dict(task_counts)} "
          f"datasets={dict(ds_counts)}")
    # "deepfake"/"spoof" are the strings preprocessing writes into the CSV.
    assert task_counts["deepfake"] == ff_pb * REAL_BATCHES, dict(task_counts)
    assert task_counts["spoof"] == siw_pb * REAL_BATCHES, dict(task_counts)

    # Validation keeps sequential order over the whole ConcatDataset — no quota
    # there, or part of the val set would never be scored.
    val_loader = T.build_interleaved_loader(
        subsample(T.load_csv(cfg.paths.ff_val_csv), CLIPS),
        subsample(T.load_csv(cfg.paths.siwmv2_val_csv), CLIPS),
        cfg, is_train=False,
    )
    assert val_loader.batch_sampler.__class__ is not T.InterleavedBatchSampler, \
        "validation must not use the training quota sampler"
    n_val = len(val_loader.dataset)
    print(f"[val ] sequential loader: {len(val_loader)} batches over {n_val} clips")
    assert len(val_loader) * cfg.train.batch_size >= n_val, "val set truncated"

    print("\nSAMPLER CHECK PASSED")


if __name__ == "__main__":
    main()
