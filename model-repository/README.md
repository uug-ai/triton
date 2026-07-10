# Triton model repository

This directory is Triton's **model repository** — the single source of truth for
which models the server hosts. Triton is pointed at it with
`--model-repository=/models` (see [`../deploy/deployment.yaml`](../deploy/deployment.yaml)).

## Layout

```
model-repository/
  <model-name>/
    config.pbtxt        # model configuration (backend, I/O, batching, ...)
    1/                  # version directory (integer)
      model.onnx        # the actual weights (NOT committed — see below)
```

Two example detectors are provided:

| Model    | Backend     | Post-processing        | Notes                                   |
| -------- | ----------- | ---------------------- | --------------------------------------- |
| `yolov8` | onnxruntime | client-side NMS        | Raw head output `[84, 8400]`            |
| `yolo26` | onnxruntime | none (NMS-free/end2end)| Emits final detections `[300, 6]`       |

## Model weights are not committed

The `*.onnx` / `*.plan` weight files are large and reproducible, so they are
git-ignored (see [`.gitignore`](.gitignore)). Only the `config.pbtxt` files and
version-directory placeholders live in git. Generate the weights with the export
scripts in [`../models`](../models):

```bash
cd ../models
pip install -r requirements.txt
python export_yolov8.py   # writes model-repository/yolov8/1/model.onnx
python export_yolo26.py   # writes model-repository/yolo26/1/model.onnx
```

After exporting, **verify the tensor names and shapes** in each `config.pbtxt`
against the real ONNX file — an export with different `imgsz`/`opset`/class count
will change them:

```bash
polygraphy inspect model yolov8/1/model.onnx
```

## Loading the repository into the cluster

The Deployment mounts this repository read-only from a PVC. Populate it with one
of:

- **`kubectl cp`** the exported models into the PVC (via a helper pod), or
- host it on **S3/MinIO** and switch Triton to `--model-repository=s3://...`
  (the stack already runs MinIO) — see the commented options in
  [`../deploy/deployment.yaml`](../deploy/deployment.yaml).

With `--model-control-mode=poll` Triton hot-reloads new versions dropped into the
repository without a restart.
