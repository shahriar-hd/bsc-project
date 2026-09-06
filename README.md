# Multi-Task Face Anti-Spoofing & Deepfake Detection

A unified multi-task learning (MTL) framework that detects **presentation attacks** and **deepfakes** from video clips in a single model, on a shared backbone, designed to run on consumer GPUs with limited VRAM.

## Highlights

- **One shared backbone (EfficientNet-B2)** + **three task heads**:
  - Deepfake detection (`FaceForensics++`)
  - Presentation-attack / anti-spoofing detection (`SiW-Mv2`)
  - Temporal-consistency auxiliary task (self-supervised, 3 independently-supervised branches)
- **Robust MTL training**: GradNorm dynamic loss balancing + PCGrad gradient surgery + temporal shift module (TSM).
- **Solved partial-label problem** — per-task loss masking + quota-based batching make "a head trained on the other task's placeholder labels" structurally impossible.
- **Data-leakage-aware splits**: FF++ grouped by *scenario* (not video), with the residual actor leak reported explicitly.
- **4 GB-class training**: gradient checkpointing + float16 AMP + gradient accumulation bring a 3-head video model to a **1133 MiB peak** at an effective batch of 16.
- **Efficient preprocessing**: offline optical flow with int8 quantization (16.8 GB → ~350 MB) keeps flow out of the training loop.

## Results (held-out test set)

| Task | Clip AUC | Video AUC | Notes |
|---|---|---|---|
| Anti-spoofing (SiW-Mv2) | **0.9882** | 0.9867 | min ACER 0.0445 |
| Deepfake (FaceForensics++) | **0.8410** | 0.8333 | honest number after closing scenario leakage (was 0.9899 on a leaky video split) |
| Temporal consistency | **0.8840** | 0.9207 | 0.9503 on SiW vs 0.7522 on FF++ |

- Whole research cycle: **0.42 kWh**, ~**208 g CO₂**.
- Verdict latency ~1.45 s/clip; model inference is only ~1.3% of the latency — face detection is the real bottleneck.

## Installation

Requires Python 3.10 and an NVIDIA GPU.

```bash
conda create -n mtl python=3.10
conda activate mtl
pip install -r requirements.txt   # pins onnxruntime-gpu (do NOT also install plain onnxruntime)
python -c "import onnxruntime as o; print(o.get_available_providers())"  # expect CUDAExecutionProvider
```

## Data

Organize raw videos, then run preprocessing (face detection, clip sampling, per-attack-type balancing, optical flow).

```text
data/datasets/raw/FaceForensics++/{real,fake}/*.mp4
data/datasets/raw/SiW-Mv2/{live,spoof}/**/*.mp4
```

```bash
python src/preprocessing.py
```

## Training

```bash
python src/train.py
```

Resumes from the highest-numbered `checkpoints/run*/last.pth` by default — clear `checkpoints/` before a fresh run. On a low-VRAM desktop, run detached with `scripts/train_headless.sh` (see `docs/headless_training.md`).

## Demo (webcam / file)

```bash
python src/app_demo.py
```

Real-time anti-spoof / deepfake / temporal screening plus enrollment-based identity verification (deepfake 0.6, spoof 0.6, temporal 0.8, identity 0.75).

## Project Layout

```text
src/         preprocessing, training, demo, central config
src/utils/   shared modules (face, flow, logger, power monitor, repro)
scripts/     launchers & analysis helpers
tests/       regression tests that drive the real Trainer/dataset
docs/        headless training guide
data/        raw & processed datasets (with CSVs)
```

## Security & Notes

- Domain limitation: only *face-swap* deepfakes are covered; both detection and evaluation are in-dataset (no cross-dataset generalization yet).
- Detection score thresholds are not calibrated probabilities; see chapters on calibration before deployment.
- Energy/CO₂ logging uses Intel RAPL / NVML counters.


## License

Read LICENSE file.


## Contact

- Repo issue
- [shahriar.hd@outlook.com](mailto:shahriar.hd@outlook.com)
