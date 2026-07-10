#!/usr/bin/env python3
"""Export a YOLO26 detector to ONNX for the Triton model repository.

YOLO26 is end-to-end / NMS-free: the exported model emits final detections
directly ([300, 6] = x1, y1, x2, y2, score, class), so no client-side NMS is
needed.

Usage:
    python export_yolo26.py [--weights yolo26n.pt] [--imgsz 640] [--opset 12]

Requires: pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# .../triton/models/export_yolo26.py -> .../triton
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model-repository" / "yolo26" / "1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n.pt", help="Ultralytics weights to export")
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
