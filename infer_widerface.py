"""
infer_widerface.py — Run Ultralytics face model on WIDER FACE val set and write
predictions in the exact format consumed by widerface_evaluate/evaluation.py.

Prediction format (matches YOLOv6 inferer.py --save-txt-widerface):
  output_dir/
    EVENT_NAME/
      image_stem.txt        ← one file per image
        line 0: image_stem  (e.g. "0_Parade_marchingband_1_480")
        line 1: N           (number of detections, int)
        lines 2+: x1 y1 w h score   (pixel coords, float)

Usage:
    # Basic (defaults to widerface/images/val)
    python infer_widerface.py --weights runs/face/train/weights/best.pt

    # Full
    python infer_widerface.py \\
        --weights runs/face/train/weights/best.pt \\
        --source  widerface/images/val \\
        --output  widerface_preds \\
        --conf 0.02 --iou 0.45 --imgsz 640 --device 0

    # Then evaluate:
    cd YOLOv6-0.3.1/widerface_evaluate
    python evaluation.py -p ../../widerface_preds -g ./ground_truth/
"""

import argparse
import os
import sys
from pathlib import Path

# Use local ultralytics package
sys.path.insert(0, str(Path(__file__).parent / "ultralytics"))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Ultralytics face model → WiderFace prediction format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained .pt weights (task=face)")
    parser.add_argument("--source", type=str,
                        default="widerface/images/val",
                        help="Root of WIDER val images. Must have EVENT/*.jpg structure.")
    parser.add_argument("--output", type=str,
                        default="widerface_preds",
                        help="Output directory for prediction txt files")
    parser.add_argument("--conf",   type=float, default=0.02,
                        help="Confidence threshold (use low value like 0.02 for eval)")
    parser.add_argument("--iou",    type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--imgsz",  type=int,   default=640,
                        help="Inference image size")
    parser.add_argument("--device", type=str,   default="",
                        help="Device: '' = auto, '0' = GPU 0, 'cpu'")
    parser.add_argument("--batch",  type=int,   default=1,
                        help="Batch size per predict call")
    return parser.parse_args()


def collect_images(source: Path) -> dict[str, list[Path]]:
    """Walk source/EVENT/*.{jpg,jpeg,png} and return {event_name: [img_path, ...]}."""
    events = {}
    for event_dir in sorted(source.iterdir()):
        if not event_dir.is_dir():
            continue
        imgs = sorted(
            p for p in event_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if imgs:
            events[event_dir.name] = imgs
    return events


def main():
    args = parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    output = Path(args.output)

    print(f"Weights : {args.weights}")
    print(f"Source  : {source}")
    print(f"Output  : {output}")
    print(f"conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}")
    print()

    model = YOLO(args.weights, task="face")

    events = collect_images(source)
    if not events:
        raise RuntimeError(f"No images found under {source}. "
                           "Expected structure: source/EVENT_NAME/*.jpg")

    total_images = sum(len(v) for v in events.values())
    print(f"Found {len(events)} events, {total_images} images total.\n")

    processed = 0
    for event_name, img_paths in events.items():
        event_out = output / event_name
        event_out.mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                save=False,
            )
            result = results[0]  # single image

            txt_path = event_out / (img_path.stem + ".txt")
            with open(txt_path, "w") as f:
                f.write(img_path.stem + "\n")

                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    f.write("0\n")
                else:
                    xyxy  = boxes.xyxy.cpu().numpy()   # (N, 4) pixel coords
                    confs = boxes.conf.cpu().numpy()   # (N,)

                    # Sort by confidence descending
                    order = confs.argsort()[::-1]
                    xyxy  = xyxy[order]
                    confs = confs[order]

                    f.write(f"{len(xyxy)}\n")
                    for (x1, y1, x2, y2), score in zip(xyxy, confs):
                        w = x2 - x1
                        h = y2 - y1
                        score = min(float(score), 1.0)
                        f.write(f"{x1:.1f} {y1:.1f} {w:.1f} {h:.1f} {score:.6f}\n")

            processed += 1
            if processed % 500 == 0:
                print(f"  [{processed}/{total_images}] {event_name}/{img_path.name}")

    print(f"\nDone. {processed} prediction files written to: {output}")
    print("\nNext step — evaluate:")
    print(f"  cd YOLOv6-0.3.1/widerface_evaluate")
    print(f"  python evaluation.py -p ../../{output} -g ./ground_truth/")


if __name__ == "__main__":
    main()
