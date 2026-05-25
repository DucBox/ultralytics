---
name: Face Detection Project — YOLOv11 Face Task
description: WIDER FACE training with 5-point landmarks integrated into Ultralytics YOLO11 as a "face" task
type: project
---

# Face Detection with Landmarks — YOLOv11

## Overview

A custom `face` task added to the Ultralytics framework, enabling **face detection** with **5-point facial landmark regression** (left eye, right eye, nose, left mouth corner, right mouth corner). Built on top of the existing `pose` infrastructure using the `Pose` head module with `kpt_shape=[5, 3]`.

**Project root:** `/Users/ngoquangduc/Desktop/workspace/face_detection/ultralytics`

## Architecture

- **Base:** YOLOv11 Detection model (non–end-to-end, NMS-based)
- **Head:** Standard `Pose` module with `kpt_shape=[5, 3]`
- **Loss:** `v8FaceLoss` — extends `v8PoseLoss`, replaces OKS-based `KeypointLoss` with `WingLoss(w=10.0, ε=2.0)` for better face landmark regression
- **No kobj loss** — landmark visibility is treated as a data property, not predicted
- **reg_max:** 16 (standard DFL)

## Key Design Decisions

| Decision                                   | Rationale                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------ |
| Reuse `Pose` head with `kpt_shape=[5,3]`   | No new head module needed; face landmarks ≡ keypoints                                |
| `kpt_shape=[5,3]` not `[5,2]`              | 3rd dim (visibility) needed to mask invalid (`-1`) landmarks in loss                 |
| `WingLoss` instead of OKS `KeypointLoss`   | Logarithmic penalty for small errors → better sub-pixel precision for face landmarks |
| `FaceTrainer` skips `PoseTrainer.__init__` | Prevents forced `task="pose"` override                                               |
| `v8FaceLoss` inherits `v8PoseLoss`         | Correct `kpts_decode` for YOLOv11's `y*2.0 + anchor - 0.5` scheme                    |

## Data Structure

```
WIDER_train/
├── images/0--Parade/*.jpg, 1--Handshaking/*.jpg, ...
└── labels/0--Parade/*.txt, 1--Handshaking/*.txt, ...   (created by prepare_widerface.py)

widerface/
├── images/val/0--Parade/*.jpg, ...
└── labels/val/0--Parade/*.txt, ...                      (converted by prepare_widerface.py)
```

**Label format** (20 values per line):

```
class cx cy w h lm1x lm1y v1 lm2x lm2y v2 lm3x lm3y v3 lm4x lm4y v4 lm5x lm5y v5
```

- All coordinates are normalized `[0, 1]`
- Landmark order: `left_eye(0)`, `right_eye(1)`, `nose(2)`, `left_mouth(3)`, `right_mouth(4)`
- Visibility: `1` = visible, `0` = not visible (coordinates set to `0 0`)

## Files Modified/Created

### New Files

| File                                         | Description                                                      |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `ultralytics/cfg/models/11/yolo11-face.yaml` | Model architecture config — nc=1, kpt_shape=[5,3], Pose head     |
| `ultralytics/cfg/datasets/widerface.yaml`    | Dataset config pointing to WIDER FACE paths                      |
| `ultralytics/models/yolo/face/__init__.py`   | Package init, exports Trainer/Validator/Predictor                |
| `ultralytics/models/yolo/face/train.py`      | `FaceTrainer` — custom task="face", uses `FaceModel`             |
| `ultralytics/models/yolo/face/val.py`        | `FaceValidator` — extends `PoseValidator`                        |
| `ultralytics/models/yolo/face/predict.py`    | `FacePredictor` — extends `PosePredictor`                        |
| `prepare_widerface.py`                       | Data prep script, converts WIDER FACE annotations to YOLO format |

### Modified Files

| File                                  | Changes                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `ultralytics/utils/loss.py`           | Added `WingLoss` class + `v8FaceLoss` class                  |
| `ultralytics/nn/tasks.py`             | Added `FaceModel` (extends `PoseModel`), import `v8FaceLoss` |
| `ultralytics/data/dataset.py`         | `use_keypoints` now includes `"face"` task                   |
| `ultralytics/models/yolo/__init__.py` | Added `face` import                                          |
| `ultralytics/models/yolo/model.py`    | Added `FaceModel` import + `task_map["face"]` entry          |
| `ultralytics/cfg/__init__.py`         | Added `"face"` to `TASKS` and `TASK2*` dicts                 |

## Pretrained Weight Transfer Analysis

### yolo11n-face.yaml (nano scale)

| Pretrained Source           | Layers Matched      | Params Matched                  | Notes                                                       |
| --------------------------- | ------------------- | ------------------------------- | ----------------------------------------------------------- |
| `yolo11n.pt` (Detection)    | 448/541 (**82.8%**) | 2,560,305/2,678,071 (**95.6%**) | Backbone/Neck ✅, cls head shape mismatch (80→1 class)      |
| `yolo11n-pose.pt` (Pose) ⭐ | 505/541 (**93.3%**) | 2,605,498/2,678,071 (**97.3%**) | Backbone/Neck ✅, only cv4 kpt layers mismatch (17kpt→5kpt) |

**Unmatched layers with `yolo11n-pose.pt`** (all in Head, shape mismatch):

- `model.23.cv4` — pose feature conv: 51ch → 16ch (due to `nk = 17*3=51` vs `5*3=15`)
- These are randomly initialized and will be learned during training

**Recommendation:** Use `yolo11n-pose.pt` for best weight coverage (**97.3%**).

## Training Commands

### Train from Scratch

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11n-face.yaml", task="face")
model.train(
    data="ultralytics/cfg/datasets/widerface.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    task="face",
)
```

### Train with Pretrained Weights (Recommended)

```python
from ultralytics import YOLO

# Option 1: From pose pretrained (BEST — 97.3% weight coverage)
model = YOLO("ultralytics/cfg/models/11/yolo11n-face.yaml", task="face")
model.load("yolo11n-pose.pt")
model.train(
    data="ultralytics/cfg/datasets/widerface.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    task="face",
)

# Option 2: From detection pretrained (95.6% weight coverage)
model = YOLO("ultralytics/cfg/models/11/yolo11n-face.yaml", task="face")
model.load("yolo11n.pt")
model.train(
    data="ultralytics/cfg/datasets/widerface.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    task="face",
)
```

### Using Different Model Scales (s/m/l/x)

The `yolo11-face.yaml` config has built-in scales. To use a larger model, simply change the scale letter in the YAML filename:

```python
from ultralytics import YOLO

# Nano (default) — 2.7M params, 6.7 GFLOPs
model = YOLO("ultralytics/cfg/models/11/yolo11n-face.yaml", task="face")
model.load("yolo11n-pose.pt")  # or yolo11n.pt

# Small — more params, better accuracy
model = YOLO("ultralytics/cfg/models/11/yolo11s-face.yaml", task="face")
model.load("yolo11s-pose.pt")  # or yolo11s.pt

# Medium — recommended for production
model = YOLO("ultralytics/cfg/models/11/yolo11m-face.yaml", task="face")
model.load("yolo11m-pose.pt")  # or yolo11m.pt

# Large
model = YOLO("ultralytics/cfg/models/11/yolo11l-face.yaml", task="face")
model.load("yolo11l-pose.pt")  # or yolo11l.pt

# Extra-large
model = YOLO("ultralytics/cfg/models/11/yolo11x-face.yaml", task="face")
model.load("yolo11x-pose.pt")  # or yolo11x.pt
```

**Important:** The YAML file is the **same** for all scales (`yolo11-face.yaml` internally defines n/s/m/l/x). The scale letter in the filename (e.g., `yolo11**n**-face.yaml`) selects which scale to use. The pretrained `.pt` file must match the same scale letter.

**No additional YAML files needed** — one `yolo11-face.yaml` covers all scales.

### Resume Interrupted Training

```python
from ultralytics import YOLO

model = YOLO("runs/face/trainX/weights/last.pt", task="face")
model.train(resume=True)
```

## Inference

```python
from ultralytics import YOLO

model = YOLO("runs/face/trainX/weights/best.pt", task="face")
results = model.predict(source="path/to/image.jpg", conf=0.25)

for res in results:
    boxes = res.boxes.xyxy  # (N, 4) — bounding boxes
    scores = res.boxes.conf  # (N,) — confidence scores
    landmarks = res.keypoints.xy  # (N, 5, 2) — 5 landmark (x, y) coords
    visibility = res.keypoints.conf  # (N, 5) — landmark visibility
```

## Verified Training Results (2 epochs, fraction=0.005 ≈ 64 images, CPU)

| Configuration                      | Epoch | box_loss | lmk_loss | cls_loss | dfl_loss | mAP50      |
| ---------------------------------- | ----- | -------- | -------- | -------- | -------- | ---------- |
| **From scratch** (train11)         | 1     | 5.641    | 43.60    | 13.49    | 3.673    | 0.000171   |
| **From scratch** (train11)         | 2     | 6.176    | 42.75    | 11.81    | 4.054    | 0.000171   |
| **+ yolo11n-pose.pt** (train10) ⭐ | 1     | 4.344    | 45.05    | 7.473    | 2.563    | 0.0430     |
| **+ yolo11n-pose.pt** (train10) ⭐ | 2     | 3.169    | 36.90    | 2.601    | 1.378    | **0.0557** |

**Key takeaway:** With pose pretrained weights, initial mAP50 is **325× higher** than from scratch (0.0557 vs 0.000171), and losses converge significantly faster — especially `box_loss` (3.17 vs 6.18) and `cls_loss` (2.60 vs 11.81).
