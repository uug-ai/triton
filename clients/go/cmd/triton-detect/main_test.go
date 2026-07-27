package main

import (
	"context"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLoadImageURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		img := image.NewRGBA(image.Rect(0, 0, 2, 3))
		img.Set(1, 2, color.RGBA{R: 255, A: 255})
		if err := png.Encode(w, img); err != nil {
			t.Errorf("encode PNG: %v", err)
		}
	}))
	defer server.Close()

	img, err := loadImageURL(context.Background(), server.URL)
	if err != nil {
		t.Fatalf("loadImageURL: %v", err)
	}
	if got, want := img.Bounds(), image.Rect(0, 0, 2, 3); got != want {
		t.Fatalf("bounds = %v, want %v", got, want)
	}
}

func TestLoadImageURLRejectsHTTPError(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()

	_, err := loadImageURL(context.Background(), server.URL)
	if err == nil || !strings.Contains(err.Error(), "404 Not Found") {
		t.Fatalf("error = %v, want 404 status", err)
	}
}

func TestLoadImageURLRejectsOversizedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(make([]byte, maxImageBytes+1))
	}))
	defer server.Close()

	_, err := loadImageURL(context.Background(), server.URL)
	if err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("error = %v, want size limit error", err)
	}
}
