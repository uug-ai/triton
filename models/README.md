# Model export scripts

Scripts and notes for producing the model files that live in the Triton
[`../model-repository`](../model-repository). The exported weights are **not**
committed (see the model repository's `.gitignore`); regenerate them here.

## Setup

```bash
pip install -r requirements.txt
```

## Export the example models

```bash
python export_yolov8.py   # -> ../model-repository/yolov8/1/model.onnx
python export_yolo26.py   # -> ../model-repository/yolo26/1/model.onnx
```

Each script accepts `--weights`, `--imgsz` and `--opset` (run with `-h` for
details). After exporting, verify the tensor names/shapes match the model's
`config.pbtxt`:

```bash
polygraphy inspect model ../model-repository/yolov8/1/model.onnx
```

## YOLOv8 vs YOLO26

- **YOLOv8** produces raw head predictions (`[84, 8400]`); the client must run
  NMS.
- **YOLO26** is end-to-end / **NMS-free**: it emits final detections directly
  (`[300, 6]` = `x1, y1, x2, y2, score, class`), so no client-side NMS is needed.

## Optional: TensorRT for production

ONNX is portable and a good default. For maximum throughput, convert the ONNX
model to a TensorRT engine on a machine with the **same GPU/driver** as the
serving nodes, then switch the model's `config.pbtxt` to
`platform: "tensorrt_plan"` with a `model.plan` file:

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:8x3x640x640 \
  --maxShapes=images:16x3x640x640
```

`coco-labels.txt` lists the 80 COCO class names (index order) for mapping the
class ids returned by the detectors.
