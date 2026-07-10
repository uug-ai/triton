package triton

import (
	"image"
	"math"
)

// Letterbox resizes img to fit a square size×size model input while preserving
// aspect ratio, padding the remainder with a neutral gray (114/255, the YOLO
// convention). It returns the input tensor as float32 in NCHW order (RGB,
// channel-first, pixel values scaled to 0..1) and a Meta describing the
// transform so detections can be mapped back to the source image.
//
// The output length is 3*size*size. Sampling is nearest-neighbour, which is
// sufficient for detection; callers that need higher-quality resizing can
// pre-resize the image.
func Letterbox(img image.Image, size int) ([]float32, Meta) {
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	meta := Meta{Size: size, SrcMinX: b.Min.X, SrcMinY: b.Min.Y, SrcW: srcW, SrcH: srcH}
	data := make([]float32, 3*size*size)
	// Fill with the padding colour.
	const pad = float32(114) / 255
	for i := range data {
		data[i] = pad
	}
	if srcW <= 0 || srcH <= 0 || size <= 0 {
		return data, meta
	}

	scale := math.Min(float64(size)/float64(srcW), float64(size)/float64(srcH))
	newW := int(math.Round(float64(srcW) * scale))
	newH := int(math.Round(float64(srcH) * scale))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}
	padX := (size - newW) / 2
	padY := (size - newH) / 2
	meta.Scale = scale
	meta.PadX = padX
	meta.PadY = padY

	area := size * size
	for y := 0; y < newH; y++ {
		sy := b.Min.Y + int(float64(y)/scale)
		if sy >= b.Max.Y {
			sy = b.Max.Y - 1
		}
		for x := 0; x < newW; x++ {
			sx := b.Min.X + int(float64(x)/scale)
			if sx >= b.Max.X {
				sx = b.Max.X - 1
			}
			r, g, bl, _ := img.At(sx, sy).RGBA() // 16-bit per channel
			idx := (padY+y)*size + (padX + x)
			data[idx] = float32(r>>8) / 255
			data[area+idx] = float32(g>>8) / 255
			data[2*area+idx] = float32(bl>>8) / 255
		}
	}
	return data, meta
}

// Meta records the Letterbox transform so a detection box expressed in the
// model's input coordinate space can be mapped back to the source image.
type Meta struct {
	Size             int     // model input edge (e.g. 640)
	Scale            float64 // source→input scale factor
	PadX, PadY       int     // padding added on the left/top
	SrcMinX, SrcMinY int     // source image origin (Bounds().Min)
	SrcW, SrcH       int     // source image dimensions
}

// ToSource maps a rectangle from the model input space back to source-image
// pixel coordinates, undoing the scale and padding and clamping to the source
// bounds.
func (m Meta) ToSource(r image.Rectangle) image.Rectangle {
	if m.Scale <= 0 {
		return r
	}
	x1 := m.SrcMinX + int(math.Round(float64(r.Min.X-m.PadX)/m.Scale))
	y1 := m.SrcMinY + int(math.Round(float64(r.Min.Y-m.PadY)/m.Scale))
	x2 := m.SrcMinX + int(math.Round(float64(r.Max.X-m.PadX)/m.Scale))
	y2 := m.SrcMinY + int(math.Round(float64(r.Max.Y-m.PadY)/m.Scale))
	out := image.Rect(x1, y1, x2, y2)
	return out.Intersect(image.Rect(m.SrcMinX, m.SrcMinY, m.SrcMinX+m.SrcW, m.SrcMinY+m.SrcH))
}
