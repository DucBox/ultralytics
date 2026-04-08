---
name: Face Detection Project — YOLO12 Face Task Extension
description: Extended Face Detection task to support YOLO12 (A2C2f attention backbone, NMS-based) architecture — only requires a single YAML file
type: project
---

# Face Detection with Landmarks — YOLO12

## Overview

Extended the existing `face` task to support **YOLO12** architecture. YOLO12 introduces **A2C2f (Area-Attention C2f)** blocks replacing standard `C3k2+SPPF+C2PSA` in the later backbone/head layers, while keeping the same `Pose` head and NMS-based pipeline as YOLO11. As a result, **no Python code changes were needed** — only a single model YAML file was added.

**Project root:** `/Users/ngoquangduc/Desktop/workspace/face_detection/ultralytics`

## Architecture Comparison: YOLO11 vs YOLO12 vs YOLO26

| Feature | YOLOv11 Face | YOLO12 Face | YOLO26 Face |
|---------|-------------|-------------|-------------|
| Head module | `Pose` | `Pose` | `Pose26` |
| End-to-end (NMS-free) | No | No | Yes |
| DFL reg_max | 16 | 16 | 1 |
| Loss class | `v8FaceLoss` | `v8FaceLoss` | `E2ELoss(FaceLoss26)` |
| Backbone innovation | C3k2 + SPPF + C2PSA | **A2C2f (area attention)** | C3k2 + full attention |
| Model YAML | `cfg/models/11/yolo11-face.yaml` | `cfg/models/12/yolo12-face.yaml` | `cfg/models/26/yolo26-face.yaml` |
| Best pretrained | `yolo11n-pose.pt` (97.3%) | `yolo12n.pt` (det, ~94%) | `yolo26n-pose.pt` (91.5%) |

## What Was Implemented for YOLO12

### 1. Model Config: `ultralytics/cfg/models/12/yolo12-face.yaml` [NEW — only change]

- Based on `yolo12-pose.yaml`
- `nc: 1` (face only), `kpt_shape: [5, 3]` (5 face landmarks + visibility)
- NMS-based, `reg_max` default = 16, `Pose` head
- Supports all scales: n/s/m/l/x (built-in via `scales:` block)

**No Python changes needed** — `FaceModel.init_criterion()` auto-selects:
```python
E2ELoss(self, FaceLoss26) if getattr(self, "end2end", False) else v8FaceLoss(self)
#                                      YOLO12 has end2end=False → v8FaceLoss ✓
```

## Pretrained Weights Note

> **IMPORTANT:** `yolo12n-pose.pt` does NOT exist publicly (Ultralytics has not released YOLO12 pose pretrained weights as of 2026-04).
>
> Best alternatives (in order of preference):
> 1. `yolo12n.pt` — detection pretrained, ~94% backbone/neck coverage
> 2. `yolo11n-pose.pt` — different architecture, lower coverage but has pose head
> 3. From scratch — works but slower convergence

| Pretrained Source | Approx Coverage | Notes |
|-------------------|----------------|-------|
| `yolo12n-pose.pt` | N/A | **Does not exist publicly** |
| `yolo12n.pt` (Detection) ⭐ | ~94% | Backbone/Neck ✓, cls/pose head layers mismatch |

## Files Summary (YOLO12-specific)

| File | Action | Description |
|------|--------|-------------|
| `ultralytics/cfg/models/12/yolo12-face.yaml` | **NEW** | YOLO12 face model config — Pose head, nc=1, kpt_shape=[5,3] |

**Shared infrastructure** (from YOLO11 face implementation, fully reused — no changes):
- Dataset config, FaceTrainer, FaceValidator, FacePredictor
- WingLoss, v8FaceLoss, FaceModel, task registration

## Training Commands

### Train from Scratch
```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/12/yolo12n-face.yaml', task='face')
model.train(
    data='ultralytics/cfg/datasets/widerface.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    task='face',
)
```

### Train with Pretrained Weights (Recommended)
```python
from ultralytics import YOLO

# Best option: detection pretrained (yolo12n-pose.pt does not exist publicly)
model = YOLO('ultralytics/cfg/models/12/yolo12n-face.yaml', task='face')
model.load('yolo12n.pt')  # detection pretrained — backbone/neck matched
model.train(
    data='ultralytics/cfg/datasets/widerface.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    task='face',
)
```

### Using Different Model Scales (s/m/l/x)
```python
from ultralytics import YOLO

# Nano
model = YOLO('ultralytics/cfg/models/12/yolo12n-face.yaml', task='face')
model.load('yolo12n.pt')

# Small
model = YOLO('ultralytics/cfg/models/12/yolo12s-face.yaml', task='face')
model.load('yolo12s.pt')

# Medium
model = YOLO('ultralytics/cfg/models/12/yolo12m-face.yaml', task='face')
model.load('yolo12m.pt')
```

### Resume Interrupted Training
```python
from ultralytics import YOLO

model = YOLO('runs/face/trainX/weights/last.pt', task='face')
model.train(resume=True)
```

## Inference

```python
from ultralytics import YOLO

model = YOLO('runs/face/trainX/weights/best.pt', task='face')
results = model.predict(source='path/to/image.jpg', conf=0.25)

for res in results:
    boxes = res.boxes.xyxy          # (N, 4) — bounding boxes
    scores = res.boxes.conf         # (N,) — confidence scores
    landmarks = res.keypoints.xy    # (N, 5, 2) — 5 landmark (x, y) coords
    visibility = res.keypoints.conf # (N, 5) — landmark visibility
```

## Verified Tests (CPU, fraction=0.005 ≈ 64 images)

| Test | Result |
|------|--------|
| T1: Forward pass (yolo12n-face.yaml) | ✅ PASS — output dict, 2.64M params |
| T2: Load yolo12n.pt detection pretrained | ✅ PASS — weights transferred |
| T3: Train 2 epochs from scratch | ✅ PASS — lmk_loss non-zero, non-NaN |
| T3b: Train 2 epochs with yolo12n.pt | ✅ PASS |
| T4: Predict from trained weights | ✅ PASS — keypoints shape (N, 5, 2) |

## Isolation from YOLO11/YOLO26

- All three architectures use separate YAML files
- Shared Python infrastructure (FaceModel, FaceTrainer, etc.) has no architecture-specific code
- No changes to any existing file — zero risk of regression
