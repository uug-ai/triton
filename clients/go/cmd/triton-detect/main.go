// Command triton-detect is a tiny example service for the Triton Go SDK: it
// checks the server/model are ready and, given an image, runs object detection
// through Triton and prints the results.
//
// Its real purpose is to demonstrate that a service built on this SDK needs NO
// Python, ONNX runtime or OpenCV — just a static Go binary — so it can ship in a
// minimal, CVE-minimal image (see the Dockerfile alongside this file).
package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg" // register JPEG decoder for image.Decode
	_ "image/png"  // register PNG decoder for image.Decode
	"os"
	"time"

	triton "github.com/uug-ai/triton/clients/go"
)

func main() {
	url := flag.String("url", envOr("TRITON_URL", "http://triton.triton:8000"), "Triton base URL")
	model := flag.String("model", envOr("TRITON_MODEL", "yolov8"), "model name")
	head := flag.String("head", envOr("TRITON_MODEL_HEAD", "yolov8"), "output head: yolov8|yolo26")
	imagePath := flag.String("image", "", "path to a JPEG/PNG frame to run detection on (optional)")
	timeout := flag.Duration("timeout", 30*time.Second, "overall timeout")
	flag.Parse()

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
	if *imagePath == "" {
		fmt.Printf("triton %s: server ready, model %q ready\n", *url, *model)
		return
	}

	img, err := loadImage(*imagePath)
	if err != nil {
		fatal("%v", err)
	}

	det := triton.NewDetector(client, triton.DetectorConfig{Model: *model, Head: triton.Head(*head)})
	dets, err := det.Detect(ctx, img)
	if err != nil {
		fatal("detect: %v", err)
	}

	fmt.Printf("%d detection(s) on %s:\n", len(dets), *imagePath)
	for _, d := range dets {
		fmt.Printf("  class=%d score=%.3f box=%v\n", d.Class, d.Score, d.Box)
	}
}

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
