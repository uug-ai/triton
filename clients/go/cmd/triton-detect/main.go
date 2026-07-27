// Command triton-detect is a tiny example service for the Triton Go SDK: it
// checks the server/model are ready and, given an image path or URL, runs object
// detection through Triton and prints the results.
//
// Its real purpose is to demonstrate that a service built on this SDK needs NO
// Python, ONNX runtime or OpenCV — just a static Go binary — so it can ship in a
// minimal, CVE-minimal image (see the Dockerfile alongside this file).
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg" // register JPEG decoder for image.Decode
	_ "image/png"  // register PNG decoder for image.Decode
	"io"
	"net/http"
	"os"
	"time"

	triton "github.com/uug-ai/triton/clients/go"
)

func main() {
	url := flag.String("url", envOr("TRITON_URL", "http://triton.triton:8000"), "Triton base URL")
	model := flag.String("model", envOr("TRITON_MODEL", "yolov8"), "model name")
	head := flag.String("head", envOr("TRITON_MODEL_HEAD", "yolov8"), "output head: yolov8|yolo26")
	imagePath := flag.String("image", "", "path to a JPEG/PNG frame to run detection on (optional)")
	imageURL := flag.String("image-url", "", "URL of a JPEG/PNG frame to download and run detection on (optional)")
	timeout := flag.Duration("timeout", 30*time.Second, "overall timeout")
	flag.Parse()
	if *imagePath != "" && *imageURL != "" {
		fatal("set only one of -image or -image-url")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	client := triton.New(*url)
	if err := client.Ready(ctx); err != nil {
		fatal("server not ready: %v", err)
	}
	if err := client.ModelReady(ctx, *model); err != nil {
		fatal("model %q not ready: %v", *model, err)
	}

	// No image: just report readiness (a handy container smoke test).
	if *imagePath == "" && *imageURL == "" {
		fmt.Printf("triton %s: server ready, model %q ready\n", *url, *model)
		return
	}

	imageSource := *imagePath
	var img image.Image
	var err error
	if *imageURL != "" {
		imageSource = *imageURL
		img, err = loadImageURL(ctx, *imageURL)
	} else {
		img, err = loadImage(*imagePath)
	}
	if err != nil {
		fatal("%v", err)
	}

	det := triton.NewDetector(client, triton.DetectorConfig{Model: *model, Head: triton.Head(*head)})
	dets, err := det.Detect(ctx, img)
	if err != nil {
		fatal("detect: %v", err)
	}

	fmt.Printf("%d detection(s) on %s:\n", len(dets), imageSource)
	for _, d := range dets {
		fmt.Printf("  class=%d score=%.3f box=%v\n", d.Class, d.Score, d.Box)
	}
}

const maxImageBytes = 20 << 20

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode image %q: %w", path, err)
	}
	return img, nil
}

func loadImageURL(ctx context.Context, url string) (image.Image, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create image request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download image: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("download image: unexpected HTTP status %s", resp.Status)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxImageBytes+1))
	if err != nil {
		return nil, fmt.Errorf("download image: %w", err)
	}
	if len(data) > maxImageBytes {
		return nil, fmt.Errorf("download image: response exceeds %d bytes", maxImageBytes)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image %q: %w", url, err)
	}
	return img, nil
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "triton-detect: "+format+"\n", args...)
	os.Exit(1)
}
