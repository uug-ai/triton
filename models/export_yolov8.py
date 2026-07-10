#!/usr/bin/env python3
"""Export a YOLOv8 detector to ONNX for the Triton model repository.

YOLOv8 emits raw head predictions ([84, 8400] for COCO); clients must run NMS.

Usage:
    python export_yolov8.py [--weights yolov8n.pt] [--imgsz 640] [--opset 12]

Requires: pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# .../triton/models/export_yolov8.py -> .../triton
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model-repository" / "yolov8" / "1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov8n.pt", help="Ultralytics weights to export")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, dynamic=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / "model.onnx"
    shutil.copyfile(exported, dest)
    print(f"Wrote {dest}")
    print("Verify I/O with: polygraphy inspect model", dest)


if __name__ == "__main__":
    main()
