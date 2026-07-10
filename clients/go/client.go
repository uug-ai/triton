// Package triton is a small, dependency-free client for NVIDIA Triton Inference
// Server, plus helpers to run YOLO-style object detection against models hosted
// on Triton (see the model-repository in this repo: yolov8, yolo26).
//
// It exists so a worker can move its model OUT of its own image and call the
// shared Triton GPU backend instead: program against the Detector interface and
// pick the backend at runtime (an embedded in-process model, or TritonDetector),
// so the in-image model becomes optional. See DetectorFromEnv.
//
// The wire protocol is Triton's KServe v2 REST API over HTTP (:8000):
//
//	POST {base}/v2/models/{model}/infer
//	GET  {base}/v2/health/ready
//	GET  {base}/v2/models/{model}/ready
//
// Only the FP32 tensor path needed for image models is implemented; large
// tensors would be more efficient over the binary tensor extension or gRPC
// (:8001), which callers can add later behind the same Detector interface.
package triton

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultTimeout bounds a single HTTP call when no client is supplied.
const DefaultTimeout = 30 * time.Second

// Client talks to a Triton server over the KServe v2 REST protocol. It is safe
// for concurrent use; the zero value is not usable — build one with New.
type Client struct {
	baseURL string
	http    *http.Client
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient sets the underlying HTTP client (for custom timeouts, transports
// or TLS). A nil client is ignored.
func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) {
		if h != nil {
			c.http = h
		}
	}
}

// WithTimeout sets the per-call timeout on a default HTTP client.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		if d > 0 {
			c.http = &http.Client{Timeout: d}
		}
	}
}

// New builds a Client for a Triton base URL such as
// "http://triton.triton:8000". A trailing slash is trimmed.
func New(baseURL string, opts ...Option) *Client {
	c := &Client{
		baseURL: strings.TrimRight(strings.TrimSpace(baseURL), "/"),
		http:    &http.Client{Timeout: DefaultTimeout},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// BaseURL returns the server base URL the client was built with.
func (c *Client) BaseURL() string { return c.baseURL }

// InferInput is one input tensor of an inference request. Data is the tensor
// values flattened in row-major order; Shape includes the batch dimension.
type InferInput struct {
	Name     string    `json:"name"`
	Shape    []int     `json:"shape"`
	Datatype string    `json:"datatype"`
	Data     []float32 `json:"data"`
}

// RequestedOutput names an output tensor the caller wants returned.
type RequestedOutput struct {
	Name string `json:"name"`
}

// InferRequest is the body of a KServe v2 infer call.
type InferRequest struct {
	Inputs  []InferInput      `json:"inputs"`
	Outputs []RequestedOutput `json:"outputs,omitempty"`
}

// InferOutput is one output tensor in the response.
type InferOutput struct {
	Name     string    `json:"name"`
	Shape    []int     `json:"shape"`
	Datatype string    `json:"datatype"`
	Data     []float32 `json:"data"`
}

// InferResponse is the body of a KServe v2 infer response.
type InferResponse struct {
	ModelName string        `json:"model_name"`
	Outputs   []InferOutput `json:"outputs"`
}

// Output returns the named output tensor, or false when the response does not
// contain it.
func (r InferResponse) Output(name string) (InferOutput, bool) {
	for _, o := range r.Outputs {
		if o.Name == name {
			return o, true
		}
	}
	return InferOutput{}, false
}

// Ready reports whether the server is ready to serve (GET /v2/health/ready).
func (c *Client) Ready(ctx context.Context) error {
	return c.get(ctx, "/v2/health/ready")
}

// ModelReady reports whether a specific model is loaded and ready
// (GET /v2/models/{model}/ready).
func (c *Client) ModelReady(ctx context.Context, model string) error {
	return c.get(ctx, "/v2/models/"+model+"/ready")
}

// Infer runs inference on the named model and returns its outputs.
func (c *Client) Infer(ctx context.Context, model string, req InferRequest) (InferResponse, error) {
	if model == "" {
		return InferResponse{}, fmt.Errorf("triton: empty model name")
	}
	body, err := json.Marshal(req)
	if err != nil {
		return InferResponse{}, fmt.Errorf("triton: encode infer request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v2/models/"+model+"/infer", bytes.NewReader(body))
	if err != nil {
		return InferResponse{}, fmt.Errorf("triton: build infer request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return InferResponse{}, fmt.Errorf("triton: infer %q: %w", model, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return InferResponse{}, fmt.Errorf("triton: infer %q: status %s: %s", model, resp.Status, strings.TrimSpace(string(snippet)))
	}

	var decoded InferResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return InferResponse{}, fmt.Errorf("triton: decode infer response: %w", err)
	}
	return decoded, nil
}

// get issues a GET and treats any non-2xx status as an error, reading a short
// snippet of the body for context.
func (c *Client) get(ctx context.Context, path string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return fmt.Errorf("triton: build request %s: %w", path, err)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("triton: GET %s: %w", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return fmt.Errorf("triton: GET %s: status %s: %s", path, resp.Status, strings.TrimSpace(string(snippet)))
	}
	return nil
}
