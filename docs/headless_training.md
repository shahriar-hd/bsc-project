# Headless Training Guide (tmux + Ubuntu CLI)

Running training detached from the desktop session, and reclaiming the VRAM the
desktop holds. Written for the machine this project is developed on:

```
GPU            NVIDIA GeForce RTX 3050 Ti Laptop, 4096 MiB total
Desktop (GNOME/gdm3) holds   ~930 MiB
Free with desktop running    ~2820 MiB
Training needs               ~2100 MiB   (1290 MiB tensors + ~300 MiB CUDA context + cuDNN workspaces)
```

So training **does** fit alongside the desktop — with roughly 700 MiB of
headroom. That headroom is what disappears when a browser opens a video, so if a
run has to survive unattended for hours, [Option B](#option-b--drop-to-a-console)
buys back the desktop's 930 MiB.

---

## 0. One-time setup

```bash
sudo apt update && sudo apt install -y tmux      # not installed by default here
```

Grant read access to the Intel RAPL energy counters — **needs sudo and resets on
every boot**, so do it before detaching, or CPU power silently logs 0 W:

```bash
bash scripts/enable_rapl_access.sh
```

---

## 1. Pre-flight checklist

```bash
cd /home/shahriar/Documents/bsc-project
conda activate bsc_project

# a) Nothing else is on the GPU
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv

# b) resume=True is the default — check what it would resume FROM
ls checkpoints/
```

**(b) matters.** `TrainConfig.resume = True`, and `_get_run_id()` picks the
*highest-numbered* `run*/` directory. A leftover directory from a short test run
is resumed silently: the log prints `Resumed at epoch N` and training continues
from a checkpoint that never saw the full dataset. For a genuinely fresh run,
either empty `checkpoints/` or set `resume = False` in [config.py](../src/config.py).

```bash
# c) Confirm the knobs that decide the memory footprint
grep -n "batch_size\|num_frames\|grad_accum_steps\|amp_dtype\|grad_checkpointing" src/config.py
```

```
batch_size          4          per GPU
grad_accum_steps    4          effective batch = 16
num_frames          8          T per clip
amp_dtype           float16
grad_checkpointing  True       required — 2426 MiB → OOM without it
num_workers         2
use_optical_flow    False      True adds ~0 MiB VRAM (56×56 int8) but needs
                               preprocessing to have written the .npz files
```

```bash
# d) Verify the training step still passes end-to-end (~2 min, 1225 MiB peak)
python tests/check_train_step.py
```

> Both this and `check_fit_epoch.py` create a `checkpoints/runNN/` directory, and
> the latter leaves a ~200 MB checkpoint behind. `resume=True` picks the
> highest-numbered run, so clear them before the real launch — otherwise the
> detached run silently continues a 6-clip debug checkpoint.

---

## 2. Launch under tmux

```bash
bash scripts/train_headless.sh
```

That wraps the recipe below. To drive it by hand instead:

```bash
tmux new-session -s train                 # create + attach
conda activate bsc_project
cd /home/shahriar/Documents/bsc-project

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

python -u src/train.py 2>&1 | tee -a runs/train_$(date +%F_%H%M).log
```

Then **`Ctrl-b` then `d`** to detach. Training keeps running; closing the
terminal, logging out, or dropping an SSH connection no longer kills it.

| Action | Command |
|---|---|
| Reattach | `tmux attach -t train` |
| List sessions | `tmux ls` |
| Scroll back | `Ctrl-b` `[` , then arrows / `PgUp`, `q` to exit |
| Detach | `Ctrl-b` `d` |
| Kill the run | `tmux attach -t train`, then `Ctrl-c` |
| Kill the session | `tmux kill-session -t train` |

Why each variable:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — lets the caching
  allocator grow segments instead of stranding memory in fragments. On a 4 GB
  card this is the difference between finishing and a late-epoch OOM that the
  peak-memory numbers do not predict.
- `CUDA_VISIBLE_DEVICES=0` — one GPU, explicitly. Prevents any DataParallel
  path from being taken.
- `OMP_NUM_THREADS=4` — 2 dataloader workers × unbounded OpenMP threads
  oversubscribes the CPU and slows the loader down.
- `python -u` — unbuffered, so `tail -f` shows progress instead of nothing for
  minutes.

---

## 3. Monitor without attaching

```bash
# GPU memory + utilisation, one line per 5 s
nvidia-smi dmon -s um -d 5

# Live training log (path printed at startup as "Run directory")
tail -f checkpoints/run01/training.log

# Per-epoch metrics as they land
column -s, -t checkpoints/run01/results.csv

# Energy / CO2 samples
column -s, -t checkpoints/run01/power_train.csv | tail -20
```

---

## Option B — drop to a console

Frees the ~930 MiB GNOME holds, roughly doubling the headroom.

```bash
# Start tmux FIRST so the run survives the display manager stopping
tmux new-session -d -s train 'bash scripts/train_headless.sh --inline'

sudo systemctl isolate multi-user.target      # desktop down, ~930 MiB freed
```

You land on a text console — log in, then `tmux attach -t train`.

```bash
sudo systemctl isolate graphical.target       # bring the desktop back
```

Neither command reboots, and neither touches the training process: the run is
inside tmux, whose parent is systemd, not the session that started it.

Caveats:

- Do this **from a local TTY or an SSH connection**, not from a terminal inside
  the GNOME session you are about to stop — start tmux detached (`-d`) first, as
  above, and the ordering is safe either way.
- `graphical.target` does not persist across reboot; `systemctl get-default`
  still returns `graphical.target`, so a reboot comes back to the desktop.
- The webcam demo (`src/app_demo.py`) needs a display — bring the desktop back
  before running it.

Verify the reclaim:

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
# expect memory.used to drop to a few tens of MiB
```

---

## 4. If it OOMs anyway

There is **no automatic recovery** — nothing catches `torch.OutOfMemoryError` and
retries at a smaller batch, so the process dies. Lower these by hand, in order;
each step costs less accuracy than the one after it:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (if it wasn't set)
2. Option B — free the desktop's 930 MiB
3. `num_frames: 8 → 6` — shortens the clip; the temporal head still has
   adjacent-frame pairs to work with
4. `batch_size: 4 → 2` **and** `grad_accum_steps: 4 → 8` — keeps the effective
   batch at 16, so the optimizer trajectory is unchanged; only BatchNorm
   statistics get noisier
5. `use_optical_flow: False` if it was on — the flow tensor itself is small, but
   `flow_encoder`'s activations are not free
6. `use_pcgrad: False` — PCGrad holds one full gradient copy per task

Do **not** turn off `grad_checkpointing` to go faster: measured at
batch_size=4 / num_frames=8 with PCGrad + GradNorm, the run needs 2426 MiB
without it and OOMs.

A mid-run OOM is recoverable — `last.pth` is written every epoch, so fix the
config and restart with `resume = True`.

---

## 5. Resuming

```bash
# resume=True is already the default; it picks the highest-numbered run*/
tmux new-session -s train 'bash scripts/train_headless.sh --inline'
tmux attach -t train
```

The log confirms what was restored:

```
Resuming from: ./checkpoints/run01/last.pth
Restored GradNorm task weights: [1.237  0.7503 1.0128]
Resumed at epoch 7  |  best metric so far: 0.8134
```

If `Restored GradNorm task weights` is missing and a warning about the L(0)
baseline appears instead, the checkpoint predates GradNorm state being
persisted — task balancing restarts from `[1, 1, 1]` while the model weights
resume normally.
