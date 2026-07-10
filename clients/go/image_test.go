package triton

import (
	"image"
	"image/color"
	"testing"
)

func solid(w, h int, c color.Color) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

func TestLetterboxDimensions(t *testing.T) {
	img := solid(100, 50, color.RGBA{R: 255, A: 255})
	data, meta := Letterbox(img, 640)

	if len(data) != 3*640*640 {
		t.Fatalf("data len = %d, want %d", len(data), 3*640*640)
	}
	// 100x50 into 640: scale = min(6.4, 12.8) = 6.4 -> 640x320, padded 160 top/bottom.
	if meta.Scale != 6.4 {
		t.Errorf("scale = %v, want 6.4", meta.Scale)
	}
	if meta.PadX != 0 || meta.PadY != 160 {
		t.Errorf("pad = (%d,%d), want (0,160)", meta.PadX, meta.PadY)
	}
	// The padded rows at the very top must be the neutral gray fill.
	if got := data[0]; got != float32(114)/255 {
		t.Errorf("top-left R = %v, want padding %v", got, float32(114)/255)
	}
	// A pixel inside the placed image should carry the source red channel (=1.0).
	area := 640 * 640
	idx := (meta.PadY+10)*640 + 10
	if data[idx] != 1 {
		t.Errorf("R at placed pixel = %v, want 1", data[idx])
	}
	if data[area+idx] != 0 {
		t.Errorf("G at placed pixel = %v, want 0", data[area+idx])
	}
}

func TestMetaToSource(t *testing.T) {
	img := solid(100, 50, color.White)
	_, meta := Letterbox(img, 640)

	// A box covering the whole placed region should map back to ~the full source.
	full := image.Rect(meta.PadX, meta.PadY, meta.PadX+640, meta.PadY+320)
	got := meta.ToSource(full)
	want := image.Rect(0, 0, 100, 50)
	if got != want {
		t.Errorf("ToSource(full) = %v, want %v", got, want)
	}
}
