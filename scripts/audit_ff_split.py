"""
Read-only audit of the FaceForensics++ split options. Touches no data and writes
no files — it only reports what each `PreprocessConfig.ff_split_key` would do.

Run this before preprocessing whenever the FF++ folder changes:

    python scripts/audit_ff_split.py

Two independent leaks can put the same content in two splits:

  scenario  a fake reuses the room, lighting, framing and clothing of the real
            video it was generated from, differing only in the face region
  actor     the same person's face appears in more than one split

The DFD actor subset cannot remove both: actors are tied together pairwise by the
fakes (each fake names two), so the identity graph collapses into a few giant
components. This script prints the numbers rather than asserting a verdict, so
the choice in config.py can be checked against the data instead of assumed.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config
from src.preprocessing import (
    build_ff_identity_map,
    build_ff_subject_map,
    ff_identity_tokens,
    ff_scene_token,
)

SPLIT_TARGETS = {"train": "train_ratio", "val": "val_ratio"}


def greedy_split(subjects: dict[str, Counter], targets: dict[str, float],
                 seed: int) -> dict[str, str]:
    """
    The same label-balanced greedy `split_by_subject` runs, mirrored here.

    Kept byte-for-byte equivalent on purpose — including the seeded shuffle that
    breaks size ties — so the table below reports the split that preprocessing
    will actually write, rather than an approximation of it.
    """
    totals: Counter = Counter()
    for counts in subjects.values():
        totals.update(counts)
    ids = sorted(subjects)
    random.Random(seed).shuffle(ids)
    ids.sort(key=lambda s: -sum(subjects[s].values()))
    filled = {s: Counter() for s in targets}
    out: dict[str, str] = {}
    for sid in ids:
        counts = subjects[sid]
        best, bs = None, None
        for split, ratio in targets.items():
            cost = max((filled[split][lab] + counts[lab]) / max(ratio * totals[lab], 1e-9)
                       for lab in totals)
            if best is None or cost < best:
                best, bs = cost, split
        out[sid] = str(bs)
        filled[str(bs)].update(counts)
    return out


def main() -> int:
    cfg = get_config().preprocess
    root = cfg.raw_data_root / cfg.ff_dataset_name
    labelled: list[tuple[Path, str]] = []
    for label, subdir in (("real", cfg.ff_real_dir), ("fake", cfg.ff_fake_dir)):
        folder = root / subdir
        if not folder.exists():
            print(f"  [error] missing folder: {folder}")
            return 1
        for vid in sorted(folder.rglob("*")):
            if vid.suffix.lower() in cfg.video_extensions:
                labelled.append((vid, label))
    if not labelled:
        print(f"  [error] no videos under {root}")
        return 1

    paths = [v for v, _ in labelled]
    label_of = {str(v): lab for v, lab in labelled}
    n_real = sum(lab == "real" for lab in label_of.values())
    n_fake = len(paths) - n_real

    print("=" * 74)
    print("FaceForensics++ split audit  (read-only)")
    print("=" * 74)
    print(f"  root   : {root}")
    print(f"  videos : {len(paths)}  ({n_real} real / {n_fake} fake)")

    # ── structure ────────────────────────────────────────────────────────────
    _, ident_stats = build_ff_identity_map(paths)
    scenes = Counter(ff_scene_token(v.stem) for v in paths)
    print(f"\n  scenarios          : {len(scenes)}  "
          f"(videos per scenario: min {min(scenes.values())}, "
          f"max {max(scenes.values())})")
    print(f"  actor tokens       : {ident_stats['n_identities']}")
    print(f"  identity components: {ident_stats['n_components']}  "
          f"(largest holds {ident_stats['largest_component_videos']} videos, "
          f"{100.0 * ident_stats['largest_component_videos'] / len(paths):.1f}%)")
    print(f"  unparsed filenames : {ident_stats['n_unparsed']}")

    shared = sum(
        1 for v, lab in labelled
        if lab == "fake" and any(
            ff_scene_token(o.stem) == ff_scene_token(v.stem)
            and label_of[str(o)] == "real" for o in paths
        )
    )
    print(f"\n  fakes whose scenario also exists as a real video: "
          f"{shared}/{n_fake} ({100.0 * shared / max(n_fake, 1):.0f}%)")

    # ── what each key would produce ──────────────────────────────────────────
    targets = {"train": cfg.train_ratio, "val": cfg.val_ratio,
               "test": max(0.0, 1.0 - cfg.train_ratio - cfg.val_ratio)}
    targets = {k: v for k, v in targets.items() if v > 0}

    print(f"\n  Split outcomes at {'/'.join(f'{100*v:.0f}' for v in targets.values())}"
          f"  (greedy, label-balanced — same code path as split_by_subject)")
    header = (f"    {'key':9} {'subjects':>8}  {'real tr/va/te':>15} "
              f"{'fake tr/va/te':>15}  {'fake% per split':>17}  "
              f"{'actor leak':>10}  {'scene leak':>10}")
    print(header)
    print("    " + "-" * (len(header) - 4))

    for key in ("scene", "identity", "video"):
        subject_of_path, _ = build_ff_subject_map(paths, key)
        groups: dict[str, Counter] = defaultdict(Counter)
        for v in paths:
            groups[subject_of_path[str(v)]][label_of[str(v)]] += 1
        assigned = greedy_split(groups, targets, cfg.split_seed)
        split_of_path = {str(v): assigned[subject_of_path[str(v)]] for v in paths}

        rc, fc = Counter(), Counter()
        for v in paths:
            (rc if label_of[str(v)] == "real" else fc)[split_of_path[str(v)]] += 1

        # An actor/scenario leaks when its videos land in more than one split.
        actor_splits: dict[str, set[str]] = defaultdict(set)
        scene_splits: dict[str, set[str]] = defaultdict(set)
        for v in paths:
            sp = split_of_path[str(v)]
            for tok in ff_identity_tokens(v.stem):
                actor_splits[tok].add(sp)
            scene_splits[ff_scene_token(v.stem)].add(sp)
        actor_leak = sum(len(s) > 1 for s in actor_splits.values())
        scene_leak = sum(len(s) > 1 for s in scene_splits.values())

        order = list(targets)
        pct = " / ".join(
            f"{100.0 * fc[s] / max(rc[s] + fc[s], 1):.0f}%" for s in order
        )
        print(f"    {key:9} {len(groups):>8}  "
              f"{'/'.join(f'{rc[s]:>4}' for s in order)}  "
              f"{'/'.join(f'{fc[s]:>4}' for s in order)}  "
              f"{pct:>17}  "
              f"{actor_leak:>3}/{len(actor_splits):<6} "
              f"{scene_leak:>3}/{len(scene_splits):<6}")
        for s in order:
            if rc[s] == 0 or fc[s] == 0:
                print(f"      [warn] {key}: {s} split is single-class "
                      f"(real={rc[s]}, fake={fc[s]}) — its metrics would be empty")

    print(f"\n  config: ff_split_key = {cfg.ff_split_key!r}")
    print("  'actor leak' / 'scene leak' count units appearing in >1 split.")
    print("  Zero in both columns is not achievable here; pick which one matters.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
