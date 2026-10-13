"""Train a YOLOv8 model on the Roboflow 'Lego Dude' dataset.

Usage:  python train.py [--epochs 100] [--model yolov8n.pt]

Some labels in this dataset are polygons and a few are plain boxes. A -seg model
rejects that mix, so we train a plain detection model; Ultralytics converts the
polygons to bounding boxes automatically.
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATASET = HERE / "Lego Dude.v1i.yolov8"


def write_data_yaml() -> Path:
    # Roboflow's data.yaml uses "../train/images" paths that resolve outside the
    # dataset folder, so write a copy with absolute paths.
    path = HERE / "data_abs.yaml"
    path.write_text(
        f"train: {DATASET / 'train' / 'images'}\n"
        f"val: {DATASET / 'valid' / 'images'}\n"
        f"test: {DATASET / 'test' / 'images'}\n"
        "nc: 1\n"
        "names: ['Lego-Dude']\n"
    )
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training on device: {device}")

    data = write_data_yaml()
    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=30,
        project=str(HERE / "runs"),
        name="lego_dude",
        exist_ok=True,
        workers=2,
        # MPS has no deterministic version of some ops; skip the warning spam.
        deterministic=False,
    )

    # Evaluate the best weights on the held-out test split.
    best = HERE / "runs" / "lego_dude" / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(
        data=str(data), split="test", device=device,
        project=str(HERE / "runs"), name="lego_dude_test", exist_ok=True,
    )
    print(f"Test mAP50 (box): {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    print(f"Best weights saved to: {best}")


if __name__ == "__main__":
    main()
