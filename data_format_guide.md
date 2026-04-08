# Data Format Guide — Ultralytics Face Task (5-point Landmarks)

## 1. Cấu trúc thư mục

```
dataset_root/
├── images/
│   ├── train/
│   │   ├── category_A/
│   │   │   ├── img001.jpg
│   │   │   └── img002.jpg
│   │   └── category_B/
│   │       └── img003.jpg
│   └── val/
│       ├── category_A/
│       │   └── img004.jpg
│       └── category_B/
│           └── img005.jpg
└── labels/
    ├── train/
    │   ├── category_A/
    │   │   ├── img001.txt   ← tên file phải khớp với ảnh
    │   │   └── img002.txt
    │   └── category_B/
    │       └── img003.txt
    └── val/
        ├── category_A/
        │   └── img004.txt
        └── category_B/
            └── img005.txt
```

**Quy tắc bắt buộc:**
- Cấu trúc thư mục `labels/` phải **mirror** `images/` hoàn toàn
- Tên file `.txt` phải **khớp tên** với file ảnh (chỉ đổi extension)
- Ultralytics tự suy ra path labels bằng cách thay `/images/` → `/labels/` trong path

---

## 2. Dataset YAML

```yaml
# Đường dẫn gốc của dataset (tuyệt đối hoặc tương đối với working dir)
path: /absolute/path/to/dataset_root

# Đường dẫn train/val tương đối so với path
train: images/train    # hoặc WIDER_train/images nếu nested
val: images/val

# Keypoint configuration — BẮT BUỘC cho face task
kpt_shape: [5, 3]      # [số landmarks, (x, y, visibility)] — PHẢI là [5, 3]

# Landmark flip index — dùng khi horizontal flip augmentation
# Thứ tự: left_eye(0), right_eye(1), nose(2), left_mouth(3), right_mouth(4)
# Sau hflip:  right_eye(1), left_eye(0), nose(2), right_mouth(4), left_mouth(3)
flip_idx: [1, 0, 2, 4, 3]

# Classes
nc: 1
names:
  0: face
```

**Lưu ý quan trọng:**
- `kpt_shape: [5, 3]` — số `3` là `(x, y, visibility)`. Nếu để `[5, 2]` thì không có visibility → loss không mask được invalid landmarks
- `path` trên server **phải là đường dẫn đúng trên server**, không phải máy local
- Thiếu `kpt_shape` trong YAML → `KeyError` khi train

---

## 3. Format Label

### Cấu trúc 1 dòng = 1 face

```
class cx cy w h  lm1x lm1y v1  lm2x lm2y v2  lm3x lm3y v3  lm4x lm4y v4  lm5x lm5y v5
```

| Field | Mô tả | Range |
|-------|-------|-------|
| `class` | Class index | `0` (face) |
| `cx cy` | Tâm bbox, normalized | `[0.0, 1.0]` |
| `w h` | Width/height bbox, normalized | `[0.0, 1.0]` |
| `lmNx lmNy` | Tọa độ landmark N, normalized | `[0.0, 1.0]` |
| `vN` | Visibility landmark N | `0` hoặc `1` |

**Thứ tự 5 landmarks:**
```
lm1: left_eye
lm2: right_eye
lm3: nose
lm4: left_mouth
lm5: right_mouth
```

### Ví dụ

```
# Face với đủ 5 landmarks visible
0 0.5014 0.2964 0.0361 0.0746  0.4964 0.2896 1  0.5137 0.2896 1  0.5068 0.2981 1  0.4986 0.3136 1  0.5137 0.3126 1

# Face mà landmarks không nhìn thấy (bị che, quá nhỏ...)
0 0.4150 0.3323 0.0371 0.0702  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0
```

---

## 4. Các lỗi gây "corrupt" label

Ultralytics đánh dấu label là **corrupt** nếu vi phạm bất kỳ điều kiện nào:

### ❌ Sai số cột
```
# Sai: 15 values (format YOLOv6, chưa convert)
0 0.50 0.29 0.036 0.074  0.496 0.289  0.513 0.289  0.506 0.298  0.498 0.313  0.513 0.312

# Đúng: 20 values
0 0.50 0.29 0.036 0.074  0.496 0.289 1  0.513 0.289 1  0.506 0.298 1  0.498 0.313 1  0.513 0.312 1
```

### ❌ Tọa độ âm
```
# Sai: bất kỳ giá trị nào < 0
0 0.41 0.33 0.037 0.070  -0.001 -0.001 0  ...

# Đúng: landmark không visible → set coords = 0, visibility = 0
0 0.41 0.33 0.037 0.070  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0
```

### ❌ Tọa độ > 1.0
```
# Sai: giá trị vượt quá 1.0
0 0.95 0.50 0.12 0.08  1.002 0.48 1  ...

# Đúng: clamp về [0, 1]
0 0.95 0.50 0.12 0.08  1.0 0.48 1  ...
```

### ❌ Visibility sai giá trị
```
# Sai: visibility = -1 (convention của YOLOv6, không dùng được cho Ultralytics)
0 0.41 0.33 0.037 0.070  0 0 -1  ...

# Đúng: visibility chỉ nhận 0 hoặc 1
0 0.41 0.33 0.037 0.070  0 0 0  ...
```

---

## 5. Chuyển đổi từ WIDER FACE / YOLOv6 format

**YOLOv6 format (15 values, dùng `-1` cho invalid):**
```
0 cx cy w h  lm1x lm1y  lm2x lm2y  lm3x lm3y  lm4x lm4y  lm5x lm5y
```

**Quy tắc convert sang Ultralytics format (20 values):**

```python
for i in range(5):
    lmx, lmy = float(parts[5 + i*2]), float(parts[5 + i*2 + 1])
    if lmx < 0 or lmy < 0:
        # Invalid/invisible landmark (WIDER FACE dùng -1 pixel, hoặc out-of-bounds)
        result.extend(["0", "0", "0"])        # x=0, y=0, v=0
    else:
        lmx = min(1.0, max(0.0, lmx))        # clamp về [0, 1]
        lmy = min(1.0, max(0.0, lmy))
        result.extend([f"{lmx:.10f}", f"{lmy:.10f}", "1"])   # v=1
```

**Tại sao clamp `< 0` thành v=0 (không phải clamp coords về 0 với v=1)?**

Trong WIDER FACE:
- `-1` pixel (= `-1/img_width` sau normalize ≈ `-0.001`) → landmark **không được annotate**
- `-0.032`, `-0.005`... → landmark **thật nhưng nằm ngoài biên ảnh** (bị crop)

Cả 2 trường hợp đều treat là v=0 vì landmark ra ngoài ảnh thì không quan sát được — behavior giống YOLOv6 (`np.where(< 0, -1, coord)` rồi loss bỏ qua).

---

## 6. Kiểm tra nhanh trước khi train

```bash
# 1. Đếm số cột (phải là 20)
head -1 labels/train/category/*.txt | awk '{print NF}'

# 2. Tìm giá trị âm
grep -r " -" labels/train/ | head -5

# 3. Tìm giá trị > 1.0
awk '{for(i=2;i<=NF;i++) if($i+0 > 1.0) print FILENAME, NR, i, $i}' labels/train/**/*.txt | head -10

# 4. Xóa cache cũ sau khi sửa labels
find labels/ -name "*.cache" -delete
```

---

## 7. Visibility trong loss

Với `kpt_shape=[5, 3]`, WingLoss chỉ tính loss trên những landmark có `v > 0`:

```python
# Trong v8FaceLoss / WingLoss:
mask = (kpt_gt[..., 2] > 0)   # v=1 → tính loss, v=0 → bỏ qua
loss = wing_loss(pred[mask], gt[mask])
```

Đây là lý do tọa độ `0 0` của invisible landmarks **không ảnh hưởng** đến training — chúng bị mask ra hoàn toàn.
