package triton

import (
	"image"
	"sort"
)

// Detection is one object returned by a detector, in source-image pixel
// coordinates.
type Detection struct {
	Box   image.Rectangle
	Score float64
	Class int
}

// Head identifies how a model's raw output tensor is laid out, so the right
// decoder is used.
type Head string

const (
	// HeadYOLOv8 is the raw Ultralytics YOLOv8 head: an [4+numClasses, anchors]
	// tensor of [cx, cy, w, h, class scores...] that still needs NMS.
	HeadYOLOv8 Head = "yolov8"
	// HeadYOLO26 is the end-to-end / NMS-free head: an [numDet, 6] tensor of
	// [x1, y1, x2, y2, score, class]; no NMS required.
	HeadYOLO26 Head = "yolo26"
)

// DecodeYOLOv8 turns a raw YOLOv8 output tensor ([4+numClasses, anchors],
// row-major) into detections in the model's input coordinate space. Boxes are
// converted from centre form to corner form, thresholded by confidence, and
// reduced with per-class non-maximum suppression at iouThres. Map the boxes back
// to the source image with Meta.ToSource.
func DecodeYOLOv8(data []float32, numClasses int, confThres, iouThres float64) []Detection {
	channels := 4 + numClasses
	if numClasses <= 0 || channels <= 4 || len(data) == 0 || len(data)%channels != 0 {
		return nil
	}
	anchors := len(data) / channels
	at := func(c, a int) float32 { return data[c*anchors+a] }

	var dets []Detection
	for a := 0; a < anchors; a++ {
		bestClass, bestScore := 0, float32(0)
		for c := 0; c < numClasses; c++ {
			if s := at(4+c, a); s > bestScore {
				bestScore, bestClass = s, c
			}
		}
		if float64(bestScore) < confThres {
			continue
		}
		cx, cy, w, h := float64(at(0, a)), float64(at(1, a)), float64(at(2, a)), float64(at(3, a))
		dets = append(dets, Detection{
			Box:   image.Rect(int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)),
			Score: float64(bestScore),
			Class: bestClass,
		})
	}
	return NMS(dets, iouThres)
}

// DecodeYOLO26 turns an end-to-end / NMS-free output tensor ([numDet, 6],
// row-major, each row [x1, y1, x2, y2, score, class]) into detections in the
// model's input coordinate space, keeping rows scoring at least confThres. No
// NMS is applied — the model already emits final detections.
func DecodeYOLO26(data []float32, confThres float64) []Detection {
	const stride = 6
	if len(data) == 0 || len(data)%stride != 0 {
		return nil
	}
	var dets []Detection
	for i := 0; i+stride <= len(data); i += stride {
		score := float64(data[i+4])
		if score < confThres {
			continue
		}
		dets = append(dets, Detection{
			Box:   image.Rect(int(data[i]), int(data[i+1]), int(data[i+2]), int(data[i+3])),
			Score: score,
			Class: int(data[i+5]),
		})
	}
	return dets
}

// NMS applies per-class non-maximum suppression: detections are sorted by score
// and a box is dropped when it overlaps an already-kept box of the same class by
// more than iouThres.
func NMS(dets []Detection, iouThres float64) []Detection {
	if len(dets) < 2 {
		return dets
	}
	sorted := make([]Detection, len(dets))
	copy(sorted, dets)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Score > sorted[j].Score })

	var kept []Detection
	suppressed := make([]bool, len(sorted))
	for i := range sorted {
		if suppressed[i] {
			continue
		}
		kept = append(kept, sorted[i])
		for j := i + 1; j < len(sorted); j++ {
			if suppressed[j] || sorted[j].Class != sorted[i].Class {
				continue
			}
			if iou(sorted[i].Box, sorted[j].Box) > iouThres {
				suppressed[j] = true
			}
		}
	}
	return kept
}

// iou is the intersection-over-union of two rectangles.
func iou(a, b image.Rectangle) float64 {
	inter := a.Intersect(b)
	if inter.Empty() {
		return 0
	}
	interArea := float64(inter.Dx() * inter.Dy())
	union := float64(a.Dx()*a.Dy()) + float64(b.Dx()*b.Dy()) - interArea
	if union <= 0 {
		return 0
	}
	return interArea / union
}
