#!/usr/bin/env python
"""Convert MTL checkpoints -> safetensors + ONNX, then upload to the Hugging Face Hub.

Reads ``checkpoints/run01/{best,last}.pth`` (the ``{"model": state_dict, ...}``
Trainer format), rebuilds the exact MTL architecture from ``src/``, writes:

    out/model_best.safetensors
    out/model_last.safetensors
    out/model_best.onnx            (opset 18, batch=1, T=8, verified against PyTorch)
    out/model_last.onnx
    out/model_best.metrics.json    (the Trainer's held-out metrics)
    out/config.json                (architecture + thresholds for the Space / web)
    out/README.md                  (model card, rendered from MODEL_CARD.md)

and uploads the whole ``out/`` folder to the Hub repo.

Usage
-----
    # convert only, do not touch the network
    python .upload/convert_and_upload.py --convert-only

    # convert + upload (token via env or --token)
    HF_TOKEN=hf_... python .upload/convert_and_upload.py --repo shahriarhd/mtl-face-auth

Run from the repository root so ``src`` is importable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.train import MTLModel  # noqa: E402

DEFAULT_REPO = "shahriarhd/mtl-face-auth"
ONNX_OPSET = 18  # torch >= 2.9 exporter only supports opset >= 18; ORT Web handles it

# Thresholds reported in the thesis demo (deepfake / spoof / temporal).
THRESHOLDS = {"deepfake": 0.6, "spoof": 0.6, "temporal": 0.8}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ExportWrapper(torch.nn.Module):
    """Return the three task logits as a flat tuple with stable ONNX names."""

    def __init__(self, model: MTLModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor):
        out = self.model(frames)
        return out["deepfake_logit"], out["spoof_logit"], out["temp_logit"]


def load_state(path: Path) -> tuple[dict, dict]:
    """Return (state_dict, full_checkpoint) from a Trainer .pth file."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        return ck["model"], ck
    return ck, {"model": ck}


def build_model(state: dict) -> MTLModel:
    cfg = get_config()
    model = MTLModel(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(f"state_dict mismatch: {len(missing)} missing keys, e.g. {missing[:4]}")
    if unexpected:
        print(f"[warn] ignoring {len(unexpected)} unexpected keys, e.g. {unexpected[:4]}")
    model.eval()
    return model


def save_safetensors(state: dict, path: Path) -> None:
    from safetensors.torch import save_file

    clean = {k: v.contiguous() for k, v in state.items()}
    save_file(clean, str(path), metadata={"format": "pt", "framework": "pytorch"})
    print(f"  safetensors -> {path}  ({path.stat().st_size / 1e6:.1f} MB, {len(clean)} tensors)")


def export_onnx(model: MTLModel, path: Path, size: int = 224, frames: int = 8) -> None:
    """Export a self-contained ONNX graph via the torch.export (dynamo) exporter.

    The legacy TorchScript exporter is deliberately avoided: on current torch it
    leaves the module in training mode and bakes dropout/BN-batch behaviour into
    the graph, producing scores that are ~1.0 off in sigmoid space. The dynamo
    exporter also needs ``external_data=False`` so the ~35 MB of weights stay in
    a single file that ONNX Runtime Web can fetch from one URL.
    """
    import copy

    wrapper = ExportWrapper(copy.deepcopy(model)).eval()
    dummy = torch.randn(1, frames, 3, size, size)
    common = dict(
        input_names=["frames"],
        output_names=["deepfake_logit", "spoof_logit", "temp_logit"],
        opset_version=ONNX_OPSET,
    )
    try:
        torch.onnx.export(wrapper, dummy, str(path), dynamo=True, external_data=False, **common)
    except TypeError:
        # older/other torch builds without `external_data`
        try:
            torch.onnx.export(wrapper, dummy, str(path), dynamo=True, **common)
        except Exception as e:  # missing onnxscript, etc.
            raise SystemExit(
                f"ONNX (dynamo) export failed: {e}\n"
                "Install the exporter backend: pip install 'onnxscript>=0.1'"
            ) from e
    except Exception as e:
        raise SystemExit(
            f"ONNX (dynamo) export failed: {e}\n"
            "Install the exporter backend: pip install 'onnxscript>=0.1'"
        ) from e

    side = Path(str(path) + ".data")
    if side.exists():
        raise SystemExit(f"export wrote external weights to {side}; expected one self-contained .onnx")
    print(f"  onnx        -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def verify_onnx(model: MTLModel, onnx_path: Path, size: int = 224, frames: int = 8) -> dict:
    """Compare ORT vs PyTorch on score-relevant inputs; abort on any real mismatch."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
    rng = np.random.RandomState(0)
    inputs = {
        "mean": np.zeros((1, frames, 3, size, size), dtype=np.float32),  # ImageNet-mean image
        "random": rng.randn(1, frames, 3, size, size).astype(np.float32),
    }
    checks = []
    for tag, x in inputs.items():
        onnx_logits = np.array([o.reshape(-1)[0] for o in sess.run(None, {in_name: x})], dtype=np.float64)
        with torch.no_grad():
            out = model(torch.from_numpy(x))
        torch_logits = np.array(
            [
                float(out["deepfake_logit"].reshape(-1)[0]),
                float(out["spoof_logit"].reshape(-1)[0]),
                float(out["temp_logit"].reshape(-1)[0]),
            ],
            dtype=np.float64,
        )
        score_diff = float(np.abs(sigmoid(onnx_logits) - sigmoid(torch_logits)).max())
        rel = float(np.abs(onnx_logits - torch_logits).max() / (np.abs(torch_logits).max() + 1e-6))
        checks.append({"input": tag, "score_max_diff": score_diff, "logit_rel_diff": rel})
        print(f"  verify[{tag:6s}] -> score |Δ| = {score_diff:.2e}   logit rel = {rel:.2e}")

    names = [o.name for o in sess.get_outputs()]
    worst = max(c["score_max_diff"] for c in checks)
    if worst > 0.01:
        raise SystemExit(
            f"ONNX export disagrees with PyTorch in score space (max |Δ|={worst:.3f}); "
            "the graph is not inference-safe — aborting."
        )
    return {
        "input": "frames",
        "outputs": names,
        "checks": checks,
        "score_max_diff": worst,
    }


def write_config(out: Path, repo: str, onnx_report: dict, ckpt_meta: dict) -> None:
    cfg = get_config()
    payload = {
        "architecture": "efficientnet_b2 + 3 heads (deepfake / spoof / temporal), TSM blocks 1/3/5",
        "backbone": cfg.model.backbone,
        "num_frames": cfg.train.num_frames,
        "image_size": 224,
        "feature_dim": 1408,
        "task_heads": {
            "deepfake": {"hidden": cfg.model.deepfake_hidden, "threshold": THRESHOLDS["deepfake"]},
            "spoof": {"hidden": cfg.model.spoof_hidden, "threshold": THRESHOLDS["spoof"]},
            "temporal": {
                "proj_dim": cfg.model.temporal_proj_dim,
                "hidden": cfg.model.temporal_hidden,
                "threshold": THRESHOLDS["temporal"],
            },
        },
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "onnx": {"opset": ONNX_OPSET, "input": "frames", "layout": "(1, T, 3, H, W)", **onnx_report},
        "source_checkpoint": {k: ckpt_meta.get(k) for k in ("epoch", "best_metric", "best_epoch")},
        "hf_repo": repo,
    }
    (out / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  config      -> {out / 'config.json'}")


def write_metrics(out: Path, tag: str, ckpt: dict) -> None:
    metrics = ckpt.get("metrics") or {}
    def _clean(d):
        return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in d.items()}
    (out / f"model_{tag}.metrics.json").write_text(
        json.dumps(_clean(metrics), indent=2) + "\n", encoding="utf-8"
    )
    print(f"  metrics     -> {out / f'model_{tag}.metrics.json'}")


def render_readme(template: Path, out: Path, repo: str, metrics: dict) -> None:
    text = template.read_text(encoding="utf-8")
    summary = metrics.get("monitored")
    text = text.replace("{{REPO_ID}}", repo)
    text = text.replace("{{COMPOSITE}}", f"{summary:.4f}" if isinstance(summary, (int, float)) else "n/a")
    # Compact headline table from the held-out metrics.
    def fmt(key, prefix=""):
        v = metrics.get(key)
        return f"{prefix}{v:.4f}" if isinstance(v, (int, float)) else "n/a"

    lines = [
        "| Task | AUC-ROC | Secondary |",
        "|---|---|---|",
        f"| Anti-spoofing (SiW-Mv2) | **{fmt('siw_sp_auc_roc')}** | ACER {fmt('siw_sp_acer')} |",
        f"| Deepfake (FaceForensics++) | **{fmt('ff_df_auc_roc')}** | EER {fmt('ff_df_eer')} |",
        f"| Temporal consistency | **{fmt('ff_tmp_auc_roc')}** | video AUC {fmt('siw_tmp_auc')} on SiW-Mv2 |",
    ]
    text = text.replace("{{METRICS_TABLE}}", "\n".join(lines))
    (out / "README.md").write_text(text, encoding="utf-8")
    print(f"  readme      -> {out / 'README.md'}")


def upload(out: Path, repo: str, token: str | None, private: bool) -> str:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 403:
            raise SystemExit(
                f"403 Forbidden: the token cannot create/write '{repo}'.\n"
                "The token is read-only (or fine-grained without write scope). Create a\n"
                "Write token at https://huggingface.co/settings/tokens (fine-grained: add\n"
                "'repo.content.write' + 'repo.manage' for your user), put it in .env as\n"
                "HF_TOKEN=hf_..., then re-run: bash .upload/upload.sh"
            ) from e
        raise
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(out),
        commit_message="Add MTL face anti-spoofing / deepfake checkpoints (safetensors + ONNX)",
    )
    url = f"https://huggingface.co/{repo}"
    print(f"[ok] uploaded {out} -> {url}")
    return url


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF model repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints" / "run01"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--token", default=None, help="HF token (falls back to $HF_TOKEN / cached login)")
    ap.add_argument("--convert-only", action="store_true", help="skip the Hub upload")
    ap.add_argument("--no-onnx", action="store_true", help="skip ONNX export/verification")
    ap.add_argument("--private", action="store_true", help="create the Hub repo as private")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best_pth, last_pth = ckpt_dir / "best.pth", ckpt_dir / "last.pth"
    for p in (best_pth, last_pth):
        if not p.exists():
            raise SystemExit(f"missing checkpoint: {p}")

    states, metas = {}, {}
    for tag, p in (("best", best_pth), ("last", last_pth)):
        print(f"[{tag}] {p}")
        state, ck = load_state(p)
        states[tag], metas[tag] = state, ck
        save_safetensors(state, out / f"model_{tag}.safetensors")
        write_metrics(out, tag, ck)

    onnx_report = {}
    if not args.no_onnx:
        model = build_model(states["best"])
        onnx_path = out / "model_best.onnx"
        export_onnx(model, onnx_path)
        onnx_report = verify_onnx(model, onnx_path)

        model_last = build_model(states["last"])
        export_onnx(model_last, out / "model_last.onnx")
        verify_onnx(model_last, out / "model_last.onnx")

    write_config(out, args.repo, onnx_report, metas["best"])
    template = Path(__file__).resolve().parent / "MODEL_CARD.md"
    if template.exists():
        render_readme(template, out, args.repo, metas["best"].get("metrics") or {})

    print(f"\n[done] artifacts in {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:32s} {f.stat().st_size / 1e6:8.2f} MB")

    if args.convert_only:
        print("\n--convert-only: skipping upload.")
        return

    url = upload(out, args.repo, args.token, args.private)
    print(f"\nSet MODEL_REPO_ID={args.repo} on the Space and web/config.js hfRepo='{args.repo}'.\n{url}")


if __name__ == "__main__":
    main()
