package triton

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestInfer(t *testing.T) {
	var gotPath, gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(InferResponse{
			ModelName: "yolov8",
			Outputs: []InferOutput{{
				Name: "output0", Shape: []int{1, 5, 1}, Datatype: "FP32",
				Data: []float32{320, 320, 100, 50, 0.9},
			}},
		})
	}))
	defer srv.Close()

	c := New(srv.URL)
	resp, err := c.Infer(context.Background(), "yolov8", InferRequest{
		Inputs:  []InferInput{{Name: "images", Shape: []int{1, 3, 2, 2}, Datatype: "FP32", Data: []float32{0, 0, 0, 0}}},
		Outputs: []RequestedOutput{{Name: "output0"}},
	})
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if gotPath != "/v2/models/yolov8/infer" {
		t.Errorf("path = %q, want /v2/models/yolov8/infer", gotPath)
	}
	if !strings.Contains(gotBody, `"datatype":"FP32"`) || !strings.Contains(gotBody, `"name":"images"`) {
		t.Errorf("request body missing input tensor: %s", gotBody)
	}
	out, ok := resp.Output("output0")
	if !ok || len(out.Data) != 5 {
		t.Fatalf("output0 = %+v, ok=%v", out, ok)
	}
}

func TestInferNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	}))
	defer srv.Close()

	_, err := New(srv.URL).Infer(context.Background(), "missing", InferRequest{})
	if err == nil {
		t.Fatal("expected error for 404 response")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("error = %v, want it to mention 404", err)
	}
}

func TestReady(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v2/health/ready" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	c := New(srv.URL + "/") // trailing slash should be trimmed
	if c.BaseURL() != srv.URL {
		t.Errorf("BaseURL = %q, want %q", c.BaseURL(), srv.URL)
	}
	if err := c.Ready(context.Background()); err != nil {
		t.Errorf("Ready: %v", err)
	}
	if err := c.ModelReady(context.Background(), "yolov8"); err == nil {
		t.Error("ModelReady expected error (503)")
	}
}
