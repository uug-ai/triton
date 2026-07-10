package triton

import (
	"context"
	"fmt"
	"image"
	"os"
	"strconv"
	"strings"
)

// Detector runs object detection on a single frame. Workers program against
// this interface so the model backend is swappable at runtime: an embedded,
// in-process model (the model baked into the worker image) or a TritonDetector
// that calls the shared Triton GPU backend. See DetectorFromEnv for the toggle.
type Detector interface {
	Detect(ctx context.Context, img image.Image) ([]Detection, error)
}

// DetectorConfig configures a TritonDetector. Only Model is required; the rest
// default to a standard Ultralytics export served on Triton.
type DetectorConfig struct {
	Model      string  // Triton model name, e.g. "yolov8" or "yolo26"
	Head       Head    // output layout; defaults to HeadYOLOv8
	InputName  string  // input tensor name; defaults to "images"
	OutputName string  // output tensor name; defaults to "output0"
	InputSize  int     // square model input edge; defaults to 640
	NumClasses int     // number of classes (HeadYOLOv8 only); defaults to 80
	ConfThres  float64 // minimum score to keep; defaults to 0.25
	IoUThres   float64 // NMS IoU threshold (HeadYOLOv8 only); defaults to 0.45
}

func (c *DetectorConfig) applyDefaults() {
	if c.Head == "" {
		c.Head = HeadYOLOv8
	}
	if c.InputName == "" {
		c.InputName = "images"
	}
	if c.OutputName == "" {
		c.OutputName = "output0"
	}
	if c.InputSize <= 0 {
		c.InputSize = 640
	}
	if c.NumClasses <= 0 {
		c.NumClasses = 80
	}
	if c.ConfThres <= 0 {
		c.ConfThres = 0.25
	}
	if c.IoUThres <= 0 {
		c.IoUThres = 0.45
	}
}

// TritonDetector detects objects by calling a YOLO model hosted on Triton. It
// preprocesses the frame (letterbox → NCHW FP32), runs inference over KServe v2,
// and decodes the output into source-image detections.
type TritonDetector struct {
	Client *Client
	Config DetectorConfig
}

// NewDetector builds a TritonDetector for the given client and config.
func NewDetector(client *Client, cfg DetectorConfig) *TritonDetector {
	cfg.applyDefaults()
	return &TritonDetector{Client: client, Config: cfg}
}

// Detect implements Detector: it letterboxes the frame to the model input, runs
// inference on Triton, decodes the output for the configured head, and maps the
// boxes back to source-image coordinates.
func (d *TritonDetector) Detect(ctx context.Context, img image.Image) ([]Detection, error) {
	if d.Client == nil {
		return nil, fmt.Errorf("triton: detector has no client")
	}
	cfg := d.Config
	cfg.applyDefaults()

	data, meta := Letterbox(img, cfg.InputSize)
	req := InferRequest{
		Inputs: []InferInput{{
			Name:     cfg.InputName,
			Shape:    []int{1, 3, cfg.InputSize, cfg.InputSize},
			Datatype: "FP32",
			Data:     data,
		}},
		Outputs: []RequestedOutput{{Name: cfg.OutputName}},
	}

	resp, err := d.Client.Infer(ctx, cfg.Model, req)
	if err != nil {
		return nil, err
	}
	out, ok := resp.Output(cfg.OutputName)
	if !ok {
		return nil, fmt.Errorf("triton: response missing output %q", cfg.OutputName)
	}

	var dets []Detection
	switch cfg.Head {
	case HeadYOLO26:
		dets = DecodeYOLO26(out.Data, cfg.ConfThres)
	default:
		dets = DecodeYOLOv8(out.Data, cfg.NumClasses, cfg.ConfThres, cfg.IoUThres)
	}

	// Map every box back to the source frame's coordinate space.
	for i := range dets {
		dets[i].Box = meta.ToSource(dets[i].Box)
	}
	return dets, nil
}

// DetectorFromEnv builds a Triton-backed Detector from the environment, or
// returns (nil, false, nil) when TRITON_URL is unset — signalling the caller to
// fall back to its embedded, in-image model. This is the seam that makes the
// baked-in model optional:
//
//	det, useTriton, err := triton.DetectorFromEnv()
//	if err != nil { return err }
//	if !useTriton {
//	    det = myEmbeddedDetector() // the model built into the image
//	}
//	boxes, err := det.Detect(ctx, frame)
//
// Environment:
//
//	TRITON_URL           base URL, e.g. http://triton.triton:8000 (enables Triton)
//	TRITON_MODEL         model name (default "yolov8")
//	TRITON_MODEL_HEAD    "yolov8" | "yolo26" (default "yolov8")
//	TRITON_INPUT_NAME    input tensor (default "images")
//	TRITON_OUTPUT_NAME   output tensor (default "output0")
//	TRITON_INPUT_SIZE    square input edge (default 640)
//	TRITON_NUM_CLASSES   class count for yolov8 (default 80)
//	TRITON_CONF          confidence threshold (default 0.25)
//	TRITON_IOU           NMS IoU threshold (default 0.45)
func DetectorFromEnv() (Detector, bool, error) {
	base := strings.TrimSpace(os.Getenv("TRITON_URL"))
	if base == "" {
		return nil, false, nil
	}
	cfg := DetectorConfig{
		Model:      envOr("TRITON_MODEL", "yolov8"),
		Head:       Head(strings.ToLower(strings.TrimSpace(os.Getenv("TRITON_MODEL_HEAD")))),
		InputName:  strings.TrimSpace(os.Getenv("TRITON_INPUT_NAME")),
		OutputName: strings.TrimSpace(os.Getenv("TRITON_OUTPUT_NAME")),
		InputSize:  envInt("TRITON_INPUT_SIZE", 0),
		NumClasses: envInt("TRITON_NUM_CLASSES", 0),
		ConfThres:  envFloat("TRITON_CONF", 0),
		IoUThres:   envFloat("TRITON_IOU", 0),
	}
	return NewDetector(New(base), cfg), true, nil
}

func envOr(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}
