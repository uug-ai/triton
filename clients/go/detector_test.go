package triton

import (
	"context"
	"encoding/json"
	"image"
	"image/color"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTritonDetectorDetect(t *testing.T) {
	// Fake Triton returning a single yolov8 detection (numClasses=1, anchors=1):
	// centre (320,320), size 100x50, class 0 score 0.9.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(InferResponse{
			ModelName: "yolov8",
			Outputs: []InferOutput{{
				Name: "output0", Shape: []int{1, 5, 1}, Datatype: "FP32",
				Data: []float32{320, 320, 100, 50, 0.9},
			}},
		})
	}))
	defer srv.Close()

	det := NewDetector(New(srv.URL), DetectorConfig{Model: "yolov8", NumClasses: 1})
	// A 640x640 frame letterboxes 1:1 (scale 1, no pad), so boxes come back unchanged.
	frame := solid(640, 640, color.Black)
	dets, err := det.Detect(context.Background(), frame)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1: %+v", len(dets), dets)
	}
	if want := image.Rect(270, 295, 370, 345); dets[0].Box != want {
		t.Errorf("box = %v, want %v", dets[0].Box, want)
	}
}

func TestDetectorFromEnv(t *testing.T) {
	t.Setenv("TRITON_URL", "")
	det, useTriton, err := DetectorFromEnv()
	if err != nil {
		t.Fatalf("DetectorFromEnv: %v", err)
	}
	if useTriton || det != nil {
		t.Errorf("with TRITON_URL unset: useTriton=%v det=%v, want false/nil", useTriton, det)
	}

	t.Setenv("TRITON_URL", "http://triton.triton:8000")
	t.Setenv("TRITON_MODEL", "yolo26")
	t.Setenv("TRITON_MODEL_HEAD", "yolo26")
	det, useTriton, err = DetectorFromEnv()
	if err != nil {
		t.Fatalf("DetectorFromEnv: %v", err)
	}
	if !useTriton || det == nil {
		t.Fatal("with TRITON_URL set: expected a Triton detector")
	}
	td, ok := det.(*TritonDetector)
	if !ok {
		t.Fatalf("detector type = %T, want *TritonDetector", det)
	}
	if td.Config.Model != "yolo26" || td.Config.Head != HeadYOLO26 {
		t.Errorf("config = %+v, want model yolo26 / head yolo26", td.Config)
	}
	if td.Client.BaseURL() != "http://triton.triton:8000" {
		t.Errorf("base URL = %q", td.Client.BaseURL())
	}
}
