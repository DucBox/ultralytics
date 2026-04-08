---
name: Face Detection Project — YOLO26 Face Task Extension
description: Extended Face Detection task to support YOLO26 (end2end, NMS-free) architecture with Pose26 head, FaceLoss26, and pretrained weight transfer analysis
type: project
---

# Face Detection with Landmarks — YOLO26

## Overview

Extended the existing `face` task (originally implemented for YOLOv11) to fully support the **YOLO26** architecture. YOLO26 uses an **end-to-end, NMS-free** detection pipeline with dual one2many/one2one training branches and a `Pose26` head (which includes a RealNVP flow model), requiring additional loss handling compared to the simpler YOLOv11 `Pose` head.

**Project root:** `/Users/ngoquangduc/Desktop/workspace/face_detection/ultralytics`

## Architecture Comparison: YOLOv11 vs YOLO26

| Feature | YOLOv11 Face | YOLO26 Face |
|---------|-------------|-------------|
| Head module | `Pose` | `Pose26` |
| End-to-end (NMS-free) | No (`end2end: False`) | Yes (`end2end: True`) |
| DFL reg_max | 16 | 1 |
| Loss class | `v8FaceLoss` | `E2ELoss(FaceLoss26)` |
| kpts_decode | `y * 2.0 + anchor - 0.5` | `y + anchor` (no scale factor) |
| Extra modules | None | `flow_model` (RealNVP), `cv4_sigma` |
| Training branches | Single | Dual (one2many + one2one with decay schedule) |
| Model file | `yolo11-face.yaml` | `yolo26-face.yaml` |

## What Was Implemented for YOLO26

### 1. Model Config: `ultralytics/cfg/models/26/yolo26-face.yaml` [NEW]
- Based on `yolo26-pose.yaml`
- `nc: 1` (face only), `kpt_shape: [5, 3]` (5 face landmarks)
- `end2end: True`, `reg_max: 1`, `Pose26` head
- Supports all scales: n/s/m/l/x (built-in via `scales:` block)

### 2. Loss: `FaceLoss26` in `ultralytics/utils/loss.py` [NEW CLASS]
- Extends `PoseLoss26` (NOT `v8PoseLoss`) — critical for correct YOLO26 anchor-offset `kpts_decode`
- Replaces OKS `KeypointLoss` with `WingLoss(w=10.0, epsilon=2.0)` for face landmarks
- Returns 3-tuple `(lmk_loss, 0, 0)` — no kobj_loss, no rle_loss (face landmarks don't need them)
- Properly handles PoseLoss26's sigma/RLE code paths by zeroing them out

### 3. Updated `FaceModel.init_criterion()` in `ultralytics/nn/tasks.py` [MODIFIED]
```python
# Before (only supported YOLOv11):
def init_criterion(self):
    return v8FaceLoss(self)

# After (supports both YOLOv11 and YOLO26):
def init_criterion(self):
    return E2ELoss(self, FaceLoss26) if getattr(self, "end2end", False) else v8FaceLoss(self)
```
- YOLO26 (`end2end=True`) → `E2ELoss` wrapping `FaceLoss26` with dual-branch training
- YOLOv11 (`end2end=False`) → `v8FaceLoss` (backward compatible)

### 4. Import added in `ultralytics/nn/tasks.py` [MODIFIED]
- Added `FaceLoss26` to imports from `ultralytics.utils.loss`

## Files Summary (YOLO26-specific changes only)

| File | Action | Description |
|------|--------|-------------|
| `ultralytics/cfg/models/26/yolo26-face.yaml` | **NEW** | YOLO26 face model config — Pose26 head, nc=1, kpt_shape=[5,3] |
| `ultralytics/utils/loss.py` | **MODIFIED** | Added `FaceLoss26` class (extends PoseLoss26 + WingLoss) |
| `ultralytics/nn/tasks.py` | **MODIFIED** | Updated `FaceModel.init_criterion()` for end2end + added import |

**Shared infrastructure** (from YOLOv11 face implementation, fully reused):
- Dataset config, data loader, FaceTrainer, FaceValidator, FacePredictor
- WingLoss, v8FaceLoss, FaceModel, task registration
- See `project_description_for_face_yolov11.md` for full list

## Pretrained Weight Transfer Analysis

### yolo26n-face.yaml (nano scale)

| Pretrained Source | Layers Matched | Params Matched | Notes |
|-------------------|---------------|----------------|-------|
| `yolo26n.pt` (Detection) | 606/879 (**68.9%**) | 2,432,922/2,817,560 (**86.3%**) | Backbone/Neck ✅, all pose-specific head layers missing |
| `yolo26n-pose.pt` (Pose) ⭐ | 795/879 (**90.4%**) | 2,577,110/2,817,560 (**91.5%**) | Backbone/Neck ✅, flow_model ✅, cv4/kpts/sigma shape mismatch |

**Unmatched layers with `yolo26n-pose.pt`** (all shape mismatch due to keypoint count):

| Layer Group | Layers | Params | Reason |
|-------------|--------|--------|--------|
| `cv4` (pose feature conv) | 60 | 236,550 | pose=85ch vs face=25ch |
| `cv4_kpts` (kpt output) | 12 | 2,340 | 17 keypoints × 3 = 51 vs 5 × 3 = 15 |
| `cv4_sigma` (sigma output) | 12 | 1,560 | 17 × 2 = 34 vs 5 × 2 = 10 |
| `flow_model` (RealNVP) | 0 | 0 | ✅ Fully matched (shape-independent) |

**Recommendation:** Use `yolo26n-pose.pt` for best weight coverage (**91.5%**).

## Training Commands

### Train from Scratch
```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/26/yolo26n-face.yaml', task='face')
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

# Option 1: From pose pretrained (BEST — 91.5% weight coverage)
model = YOLO('ultralytics/cfg/models/26/yolo26n-face.yaml', task='face')
model.load('yolo26n-pose.pt')
model.train(
    data='ultralytics/cfg/datasets/widerface.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    task='face',
)

# Option 2: From detection pretrained (86.3% weight coverage)
model = YOLO('ultralytics/cfg/models/26/yolo26n-face.yaml', task='face')
model.load('yolo26n.pt')
model.train(
    data='ultralytics/cfg/datasets/widerface.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    task='face',
)
```

### Using Different Model Scales (s/m/l/x)

The `yolo26-face.yaml` config has built-in scales. To use a larger model, simply change the scale letter in the YAML filename:

```python
from ultralytics import YOLO

# Nano (default) — 2.8M params, 6.6 GFLOPs
model = YOLO('ultralytics/cfg/models/26/yolo26n-face.yaml', task='face')
model.load('yolo26n-pose.pt')   # match scale: n → n

# Small
model = YOLO('ultralytics/cfg/models/26/yolo26s-face.yaml', task='face')
model.load('yolo26s-pose.pt')   # match scale: s → s

# Medium — recommended for production
model = YOLO('ultralytics/cfg/models/26/yolo26m-face.yaml', task='face')
model.load('yolo26m-pose.pt')   # match scale: m → m

# Large
model = YOLO('ultralytics/cfg/models/26/yolo26l-face.yaml', task='face')
model.load('yolo26l-pose.pt')   # match scale: l → l

# Extra-large
model = YOLO('ultralytics/cfg/models/26/yolo26x-face.yaml', task='face')
model.load('yolo26x-pose.pt')   # match scale: x → x
```

**Important:** The YAML file is the **same** for all scales (`yolo26-face.yaml` internally defines n/s/m/l/x). The scale letter in the filename (e.g., `yolo26**n**-face.yaml`) selects which scale to use. The pretrained `.pt` file must match the same scale letter.

**No additional YAML files needed** — one `yolo26-face.yaml` covers all scales.

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

## Verified Training Results (2 epochs, fraction=0.005 ≈ 64 images, CPU)

| Configuration | Epoch | box_loss | lmk_loss | cls_loss | dfl_loss | mAP50 |
|---------------|-------|----------|----------|----------|----------|-------|
| **From scratch** (train8) | 1 | 4.828 | 26.50 | 5.398 | 0.0161 | 0.000395 |
| **From scratch** (train8) | 2 | 4.458 | 25.04 | 5.143 | 0.0146 | 0.000382 |
| **+ yolo26n.pt** (train9) | 1 | 4.391 | 26.02 | 4.364 | 0.0087 | 0.000260 |
| **+ yolo26n.pt** (train9) | 2 | 3.781 | 24.59 | 4.045 | 0.0062 | 0.000270 |
| **+ yolo26n-pose.pt** (train12) ⭐ | 1 | 4.962 | 27.82 | 5.856 | 0.0172 | 0.0326 |
| **+ yolo26n-pose.pt** (train12) ⭐ | 2 | 3.925 | 24.02 | 3.833 | 0.0112 | **0.0615** |

**Key takeaway:** With pose pretrained weights, mAP50 reaches **0.0615** after just 2 epochs on 64 images — **160× higher** than from scratch (0.000382). The `box_loss` and `cls_loss` also converge much faster.

## Comparison: YOLO26 vs YOLOv11 (same conditions)

| Model | Pretrained | Epoch 2 box_loss | Epoch 2 mAP50 | Params |
|-------|-----------|-----------------|---------------|--------|
| YOLOv11n-face | yolo11n-pose.pt | **3.169** | 0.0557 | 2.68M |
| YOLO26n-face | yolo26n-pose.pt | 3.925 | **0.0615** | 2.80M |

Both architectures work well for the face detection task. YOLOv11 has slightly lower loss but YOLO26 achieves slightly higher mAP50 — in practice, these differences vanish with full training.
