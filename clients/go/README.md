# Triton Go client

A small, **dependency-free** Go client for the Triton Inference Server, plus
helpers to run YOLO-style object detection against the models hosted in this
repo's [`model-repository`](../../model-repository) (`yolov8`, `yolo26`).

Its purpose is to let a worker move its model **out of its own image** and call
the shared Triton GPU backend instead. Program against the `Detector` interface
and choose the backend at runtime, so the in-image model becomes **optional**.

```
import triton "github.com/uug-ai/triton/clients/go"
```

## Decoupling an embedded model

`DetectorFromEnv` returns a Triton-backed detector when `TRITON_URL` is set, or
`(nil, false, nil)` so the caller falls back to the model built into the image:

```go
det, useTriton, err := triton.DetectorFromEnv()
if err != nil {
    return err
}
if !useTriton {
    det = myEmbeddedDetector() // the ONNX model baked into the worker image
}

boxes, err := det.Detect(ctx, frame) // []triton.Detection in source-image pixels
```

The Triton path is pure Go (stdlib `net/http`) — **no `onnxruntime`, OpenCV or
`-tags onnx`** — so the default static worker image can do GPU inference just by
pointing `TRITON_URL` at the server. Setting no env keeps the embedded model, so
this is additive and the default behaviour is unchanged.

### Configuration (env)

| Variable             | Default                     | Purpose                              |
| -------------------- | --------------------------- | ------------------------------------ |
| `TRITON_URL`         | —                           | Base URL; **set to enable Triton**   |
| `TRITON_MODEL`       | `yolov8`                    | Model name in the repository         |
| `TRITON_MODEL_HEAD`  | `yolov8`                    | Output layout: `yolov8` or `yolo26`  |
| `TRITON_INPUT_NAME`  | `images`                    | Input tensor name                    |
| `TRITON_OUTPUT_NAME` | `output0`                   | Output tensor name                   |
| `TRITON_INPUT_SIZE`  | `640`                       | Square model input edge              |
| `TRITON_NUM_CLASSES` | `80`                        | Class count (yolov8 head)            |
| `TRITON_CONF`        | `0.25`                      | Confidence threshold                 |
| `TRITON_IOU`         | `0.45`                      | NMS IoU threshold (yolov8 head)      |

## Lower-level client

`Client` speaks the KServe v2 REST API directly if you need custom tensors:

```go
c := triton.New("http://triton.triton:8000")
if err := c.Ready(ctx); err != nil { /* server not ready */ }

resp, err := c.Infer(ctx, "yolov8", triton.InferRequest{
    Inputs: []triton.InferInput{{
        Name: "images", Shape: []int{1, 3, 640, 640}, Datatype: "FP32", Data: pixels,
    }},
    Outputs: []triton.RequestedOutput{{Name: "output0"}},
})
```

Helpers: `Letterbox` (frame → NCHW FP32 + a `Meta` for mapping boxes back),
`DecodeYOLOv8` (raw head + NMS), `DecodeYOLO26` (end-to-end / NMS-free), and
`NMS`.

## Scope

- FP32 tensors over HTTP (`:8000`). For large tensors the KServe **binary tensor
  extension** or **gRPC** (`:8001`) is more efficient; both can be added behind
  the same `Detector` interface without changing callers.
- Nearest-neighbour letterbox resize (fine for detection).

## Test

```bash
cd clients/go
go test ./...
```

The tests are stdlib-only (`net/http/httptest`) and need no running Triton server.

## Wiring it into a worker

`hub-anpr` already has the pluggable-engine seam (`plate.Recognizer` /
`plate.Localizer`, selected by env, with the embedded ONNX behind `-tags onnx`).
A Triton backend slots in as another adapter: build a detector with
`DetectorFromEnv` and return it from `NewLocalizer` when `TRITON_URL` is set,
keeping the embedded detector as the fallback. `hub-objecttracking` (the
designated `yolov8-detect` consumer) can use the `Detector` directly in its
tracker.
