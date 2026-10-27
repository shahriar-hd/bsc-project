---
license: mit
library_name: pytorch
tags:
  - face-anti-spoofing
  - deepfake-detection
  - multi-task-learning
  - efficientnet
  - onnx
  - computer-vision
pipeline_tag: video-classification
---

# MTL Face Anti-Spoofing & Deepfake Detection

Single shared backbone (EfficientNet-B2) with **three heads** — deepfake, presentation-attack (spoof) and temporal consistency — trained jointly with GradNorm + PCGrad and a temporal shift module (TSM) on blocks 1/3/5.

- **Repository:** [`{{REPO_ID}}`](https://huggingface.co/{{REPO_ID}})
- **Input:** 8 RGB frames, 224×224, ImageNet-normalized, laid out `(1, 8, 3, 224, 224)`
- **Output:** three sigmoid scores — `deepfake`, `spoof`, `temporal`
- **Decision thresholds:** deepfake `0.6`, spoof `0.6`, temporal `0.8`
- **Composite (monitored) score:** `{{COMPOSITE}}`

## Held-out results

{{METRICS_TABLE}}

## Files

| File | Description |
|---|---|
| `model_best.safetensors` | Best checkpoint (`torch` state dict, 536 tensors) |
| `model_last.safetensors` | Last-epoch checkpoint |
| `model_best.onnx` / `model_last.onnx` | ONNX opset 18, fixed batch 1 × 8 frames, weights embedded — for ONNX Runtime / Web |
| `model_best.metrics.json` | Full held-out metrics from the Trainer |
| `config.json` | Architecture, normalization and thresholds |
| `README.md` | This model card |

## Quick start (PyTorch)

```python
import torch, numpy as np, json

# The architecture lives in the training repo: src/train.py -> MTLModel
model = MTLModel(get_config())              # see the project repo linked in the card
from safetensors.torch import load_file
model.load_state_dict(load_file("model_best.safetensors"), strict=False)
model.eval()

x = torch.randn(1, 8, 3, 224, 224)          # 8 frames, ImageNet-normalized
with torch.no_grad():
    out = model(x)                          # {"deepfake_logit", "spoof_logit", "temp_logit", ...}
scores = {k: torch.sigmoid(v).item() for k, v in out.items() if k.endswith("_logit")}
```

## Quick start (ONNX / browser)

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("model_best.onnx", providers=["CPUExecutionProvider"])
y = sess.run(None, {"frames": np.zeros((1, 8, 3, 224, 224), dtype=np.float32)})
# -> [deepfake_logit, spoof_logit, temp_logit]
```

The exported graph is the backend of the GitHub Pages demo, which runs **ONNX Runtime Web (WASM)** entirely on the user's machine — video never leaves the browser.

## Intended use & limitations

- Research / academic demonstration of joint anti-spoofing + deepfake detection.
- The deepfake head is trained on FaceForensics++ and reported **honestly** after closing the scenario-level split leakage (AUC ≈ 0.84). Do not treat it as a production security control.
- Performance depends on a detectable face and reasonable head stability; large motion triggers a "keep face still" rejection in the demo.
- Scores are per-clip probabilities of *attack*, not identity. Not a biometric matcher.
- No warranty; use at your own risk.
