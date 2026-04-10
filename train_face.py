"""
train_face.py — Train face detection with 5-point landmarks using Ultralytics YOLO.

Supports YOLO11, YOLO12, YOLO26 with the custom face task (WingLoss + Pose head).

Usage examples:
    # YOLO12n from scratch
    python3 train_face.py --model ultralytics/cfg/models/12/yolo12n-face.yaml

    # YOLO12n with detection pretrained weights
    python3 train_face.py --model ultralytics/cfg/models/12/yolo12n-face.yaml --weights yolo12n.pt

    # YOLO11n with pose pretrained weights (best coverage)
    python3 train_face.py --model ultralytics/cfg/models/11/yolo11n-face.yaml --weights yolo11n-pose.pt

    # YOLO26n with pose pretrained weights
    python3 train_face.py --model ultralytics/cfg/models/26/yolo26n-face.yaml --weights yolo26n-pose.pt

    # Full training with custom params
    python3 train_face.py \\
        --model ultralytics/cfg/models/12/yolo12n-face.yaml \\
        --weights yolo12n.pt \\
        --epochs 100 --batch 32 --imgsz 640 --device 0

    # Resume interrupted training
    python3 train_face.py --resume runs/face/trainX/weights/last.pt

Pretrained weights availability:
    yolo11n/s/m/l/x-pose.pt   — available (best for yolo11-face, 97% coverage)
    yolo12n/s/m/l/x.pt        — available (detection only; yolo12-pose.pt does NOT exist)
    yolo26n/s/m/l/x-pose.pt   — available (best for yolo26-face, 91% coverage)
"""

import argparse
import sys
from pathlib import Path

# Use local ultralytics package, not the pip-installed one
sys.path.insert(0, str(Path(__file__).parent / "ultralytics"))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO face detection with 5-point landmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Core ──────────────────────────────────────────────────────────────────
    core = parser.add_argument_group("Core")
    core.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/12/yolo12n-face.yaml",
        help=(
            "Model path with scale letter in filename. "
            "Ultralytics strips the scale letter to find the actual YAML on disk. "
            "e.g. 'yolo26n-face.yaml' → loads 'yolo26-face.yaml' with scale=n. "
            "Valid: yolo11[n/s/m/l/x]-face.yaml, yolo12[n/s/m/l/x]-face.yaml, yolo26[n/s/m/l/x]-face.yaml"
        ),
    )
    core.add_argument(
        "--task",
        type=str,
        default="face",
        help="Task type: 'face' (WingLoss + VFL + GIoU + RepulsionLoss) or 'pose' (OKS + BCE + CIoU)",
    )
    core.add_argument(
        "--face-loss",
        type=str,
        default="auto",
        choices=["auto", "v8", "v6", "26"],
        help=(
            "Face loss variant (only used when task='face'):\n"
            "  auto — auto-detect from head type (Pose→v8FaceLoss, Pose26→v8YOLOv6FaceLoss)\n"
            "  v8   — v8FaceLoss: WingLoss + BCE + CIoU (for YOLO11/12)\n"
            "  v6   — v8YOLOv6FaceLoss: WingLoss + VFL + GIoU + RepulsionLoss (YOLOv6 clone)\n"
            "  26   — FaceLoss26: WingLoss + BCE + CIoU with Pose26 decode (YOLO26 native)"
        ),
    )
    core.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Pretrained weights to load before training (.pt file). "
             "None = train from scratch. "
             "Recommended: yolo11n-pose.pt / yolo12n.pt / yolo26n-pose.pt",
    )
    core.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="LAST_PT",
        help="Resume training from last.pt checkpoint path. "
             "When set, --model and --weights are ignored.",
    )
    core.add_argument(
        "--data",
        type=str,
        default="ultralytics/cfg/datasets/widerface.yaml",
        help="Dataset YAML config path",
    )
    core.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: '' = auto, '0' = GPU 0, '0,1' = multi-GPU, 'cpu'",
    )

    # ── Training schedule ─────────────────────────────────────────────────────
    sched = parser.add_argument_group("Training schedule")
    sched.add_argument("--epochs",       type=int,   default=100,   help="Total training epochs")
    sched.add_argument("--batch",        type=int,   default=16,    help="Batch size (per GPU)")
    sched.add_argument("--imgsz",        type=int,   default=640,   help="Input image size")
    sched.add_argument("--fraction",     type=float, default=1.0,   help="Dataset fraction to use (0.0-1.0); useful for quick smoke tests")
    sched.add_argument("--patience",     type=int,   default=100,   help="Early stopping patience (epochs with no improvement)")
    sched.add_argument("--save-period",  type=int,   default=-1,    help="Save checkpoint every N epochs (-1 = only best/last)")
    sched.add_argument("--workers",      type=int,   default=8,     help="DataLoader worker threads")
    sched.add_argument("--seed",         type=int,   default=0,     help="Random seed for reproducibility")
    sched.add_argument("--amp",          action="store_true", default=True, help="Use Automatic Mixed Precision (AMP)")
    sched.add_argument("--no-amp",       dest="amp", action="store_false",  help="Disable AMP")
    sched.add_argument("--close-mosaic", type=int,   default=10,    help="Disable mosaic augmentation in last N epochs")
    sched.add_argument("--freeze",       type=int,   default=None,  help="Freeze first N backbone layers (None = freeze nothing)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt = parser.add_argument_group("Optimizer")
    opt.add_argument("--optimizer",       type=str,   default="auto",  help="Optimizer: auto, SGD, Adam, AdamW, RMSProp")
    opt.add_argument("--lr0",             type=float, default=0.01,    help="Initial learning rate")
    opt.add_argument("--lrf",             type=float, default=0.01,    help="Final LR = lr0 * lrf")
    opt.add_argument("--momentum",        type=float, default=0.937,   help="SGD momentum / Adam beta1")
    opt.add_argument("--weight-decay",    type=float, default=0.0005,  help="Optimizer weight decay")
    opt.add_argument("--warmup-epochs",   type=float, default=3.0,     help="Warmup epochs")
    opt.add_argument("--warmup-momentum", type=float, default=0.8,     help="Warmup initial momentum")
    opt.add_argument("--warmup-bias-lr",  type=float, default=0.1,     help="Warmup initial bias lr")
    opt.add_argument("--cos-lr",          action="store_true",         help="Use cosine LR scheduler (default: linear)")

    # ── Loss weights ──────────────────────────────────────────────────────────
    loss = parser.add_argument_group("Loss weights")
    loss.add_argument("--box",  type=float, default=7.5,   help="Box regression loss weight")
    loss.add_argument("--cls",  type=float, default=0.5,   help="Classification loss weight")
    loss.add_argument("--dfl",  type=float, default=1.5,   help="DFL loss weight")
    loss.add_argument("--pose", type=float, default=12.0,  help="Landmark (WingLoss) weight")
    loss.add_argument("--kobj", type=float, default=1.0,   help="Keypoint objectness loss weight (always 0 for face task)")
    loss.add_argument("--rle",  type=float, default=1.0,   help="RLE loss weight (pose tasks; unused for face task)")

    # ── Augmentation ──────────────────────────────────────────────────────────
    aug = parser.add_argument_group("Augmentation")
    aug.add_argument("--hsv-h",      type=float, default=0.015, help="Hue augmentation fraction")
    aug.add_argument("--hsv-s",      type=float, default=0.7,   help="Saturation augmentation fraction")
    aug.add_argument("--hsv-v",      type=float, default=0.4,   help="Value (brightness) augmentation fraction")
    aug.add_argument("--degrees",    type=float, default=0.0,   help="Rotation range (degrees)")
    aug.add_argument("--translate",  type=float, default=0.1,   help="Translation range (fraction of image size)")
    aug.add_argument("--scale",      type=float, default=0.5,   help="Scale range (+/- fraction)")
    aug.add_argument("--shear",      type=float, default=0.0,   help="Shear range (degrees)")
    aug.add_argument("--perspective",type=float, default=0.0,   help="Perspective distortion (0-0.001)")
    aug.add_argument("--flipud",     type=float, default=0.0,   help="Vertical flip probability")
    aug.add_argument("--fliplr",     type=float, default=0.5,   help="Horizontal flip probability")
    aug.add_argument("--bgr",        type=float, default=0.0,   help="RGB↔BGR channel swap probability")
    aug.add_argument("--mosaic",     type=float, default=1.0,   help="Mosaic augmentation probability")
    aug.add_argument("--mixup",      type=float, default=0.0,   help="MixUp augmentation probability")
    aug.add_argument("--cutmix",     type=float, default=0.0,   help="CutMix augmentation probability")
    aug.add_argument("--copy-paste", type=float, default=0.0,   help="Copy-paste augmentation probability (not recommended for face — keypoints not handled)")
    aug.add_argument("--multi-scale",type=float, default=0.0,   help="Multi-scale range as fraction of imgsz; sizes rounded to stride multiples")

    # ── Output ────────────────────────────────────────────────────────────────
    out = parser.add_argument_group("Output")
    out.add_argument("--project", type=str, default=None, help="Save results to project/name (default: runs/face)")
    out.add_argument("--name",    type=str, default=None, help="Experiment name (default: train)")
    out.add_argument("--exist-ok",action="store_true",    help="Overwrite existing experiment directory")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Resume mode ───────────────────────────────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming training from: {resume_path}")
        model = YOLO(str(resume_path), task=args.task)
        model.train(resume=True)
        return

    # ── Normal training ───────────────────────────────────────────────────────
    import re
    model_path = Path(args.model)

    # Ultralytics convention: scale letter (n/s/m/l/x) is embedded in the filename passed by the user,
    # but the actual file on disk has the letter stripped.
    # e.g. user passes "yolo26n-face.yaml" → Ultralytics loads "yolo26-face.yaml" + sets scale="n"
    # So we must validate the scale-stripped path, not the raw path.
    scale_match = re.search(r"yolo(e-)?[v]?\d+([nslmx])", model_path.stem)
    if not scale_match:
        raise ValueError(
            f"Model filename '{model_path.name}' is missing the scale letter (n/s/m/l/x).\n"
            "Pass the scale letter in the filename — Ultralytics will strip it to find the actual YAML.\n"
            "Examples: yolo11n-face.yaml, yolo12s-face.yaml, yolo26m-face.yaml"
        )

    # Derive the actual file path on disk (scale letter stripped)
    scale_letter = scale_match.group(2)
    actual_path = model_path.parent / model_path.name.replace(scale_letter + "-", "-")
    if not actual_path.exists():
        raise FileNotFoundError(
            f"Model YAML not found on disk: {actual_path}\n"
            "Available face configs:\n"
            "  ultralytics/cfg/models/11/yolo11-face.yaml  → pass as yolo11n/s/m/l/x-face.yaml\n"
            "  ultralytics/cfg/models/12/yolo12-face.yaml  → pass as yolo12n/s/m/l/x-face.yaml\n"
            "  ultralytics/cfg/models/26/yolo26-face.yaml  → pass as yolo26n/s/m/l/x-face.yaml"
        )

    print(f"Model:   {model_path}")
    print(f"Weights: {args.weights or 'None (train from scratch)'}")
    print(f"Data:    {args.data}")
    print(f"Task:    {args.task}")
    if args.task == "face":
        print(f"Loss:    {args.face_loss}")
    print(f"Device:  {args.device or 'auto'}")
    print()

    model = YOLO(str(model_path), task=args.task)

    # Set face loss variant on the inner model so FaceModel.init_criterion() can read it
    if args.task == "face" and hasattr(model, "model") and hasattr(model.model, "model"):
        model.model.face_loss_type = args.face_loss

    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            # Let Ultralytics try to auto-download (e.g. yolo12n.pt)
            print(f"Weights file not found locally, attempting auto-download: {args.weights}")
        model.load(args.weights)

    train_kwargs = dict(
        data=args.data,
        task=args.task,
        # Schedule
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        fraction=args.fraction,
        patience=args.patience,
        save_period=args.save_period,
        workers=args.workers,
        seed=args.seed,
        amp=args.amp,
        close_mosaic=args.close_mosaic,
        # Optimizer
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        cos_lr=args.cos_lr,
        # Loss weights
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        pose=args.pose,
        kobj=args.kobj,
        rle=args.rle,
        # Augmentation
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        bgr=args.bgr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        cutmix=args.cutmix,
        copy_paste=args.copy_paste,
        multi_scale=args.multi_scale,
        # Device
        device=args.device,
    )

    # Optional output overrides
    if args.project:
        train_kwargs["project"] = args.project
    if args.name:
        train_kwargs["name"] = args.name
    if args.exist_ok:
        train_kwargs["exist_ok"] = args.exist_ok
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    results = model.train(**train_kwargs)
    print(f"\nTraining complete. Results saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
