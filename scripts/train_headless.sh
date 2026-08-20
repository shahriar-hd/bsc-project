#!/usr/bin/env bash
# Launch MTL training headless, with the environment a 4 GB card needs.
#
#   bash scripts/train_headless.sh            # create tmux session + attach
#   bash scripts/train_headless.sh --inline   # run in this shell (for `tmux new -d ... `)
#
# See docs/headless_training.md for the VRAM budget and the OOM ladder.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${TMUX_SESSION:-train}"
PYTHON="${PYTHON:-/home/shahriar/miniconda3/envs/bsc_project/bin/python}"

# Fragmentation is what kills a long run on 4 GB — the peak-memory numbers look
# fine right up until an allocation cannot find a contiguous segment.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# 2 dataloader workers with unbounded OpenMP threads oversubscribe the CPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ "${1:-}" == "--inline" ]]; then
    cd "$PROJECT_DIR"
    mkdir -p runs
    LOG="runs/train_$(date +%F_%H%M).log"

    echo "── $(date '+%F %T') ────────────────────────────────────────────"
    echo "python : $PYTHON"
    echo "alloc  : $PYTORCH_CUDA_ALLOC_CONF"
    echo "log    : $LOG"
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv || true
    # resume=True is the default and resumes the highest-numbered run*/ — show
    # what is there so a leftover test checkpoint is not picked up unnoticed.
    echo "ckpts  : $(ls checkpoints/ 2>/dev/null | tr '\n' ' ')"
    echo "───────────────────────────────────────────────────────────────"

    exec "$PYTHON" -u src/train.py 2>&1 | tee -a "$LOG"
fi

command -v tmux >/dev/null || { echo "tmux not installed: sudo apt install -y tmux" >&2; exit 1; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists — attaching. (Ctrl-b d to detach.)"
else
    tmux new-session -d -s "$SESSION" \
        "bash '${BASH_SOURCE[0]}' --inline; echo; echo '[exited — Ctrl-b d to detach, or exit]'; exec bash"
    echo "Started tmux session '$SESSION'. Ctrl-b then d to detach."
fi
exec tmux attach -t "$SESSION"
