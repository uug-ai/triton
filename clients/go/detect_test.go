package triton

import (
	"image"
	"testing"
)

func TestDecodeYOLOv8(t *testing.T) {
	// numClasses=2, anchors=2, channels=6. Layout is value(c,a)=data[c*anchors+a].
	// Anchor 0 is a strong class-0 detection; anchor 1 is below threshold.
	numClasses, anchors := 2, 2
	data := make([]float32, (4+numClasses)*anchors)
	set := func(c, a int, v float32) { data[c*anchors+a] = v }
	set(0, 0, 100) // cx
	set(1, 0, 100) // cy
	set(2, 0, 40)  // w
	set(3, 0, 20)  // h
	set(4, 0, 0.9) // class 0 score
	set(5, 0, 0.1) // class 1 score
	set(4, 1, 0.1)
	set(5, 1, 0.1)

	dets := DecodeYOLOv8(data, numClasses, 0.5, 0.45)
	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1: %+v", len(dets), dets)
	}
	d := dets[0]
	if d.Class != 0 || d.Score != float64(float32(0.9)) {
		t.Errorf("class/score = %d/%v, want 0/0.9", d.Class, d.Score)
	}
	if want := image.Rect(80, 90, 120, 110); d.Box != want {
		t.Errorf("box = %v, want %v", d.Box, want)
	}
}

func TestDecodeYOLO26(t *testing.T) {
	// Two rows [x1,y1,x2,y2,score,class]; only the first clears the threshold.
	data := []float32{
		10, 20, 110, 220, 0.8, 3,
		0, 0, 0, 0, 0.1, 0,
	}
	dets := DecodeYOLO26(data, 0.5)
	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1: %+v", len(dets), dets)
	}
	if want := image.Rect(10, 20, 110, 220); dets[0].Box != want {
		t.Errorf("box = %v, want %v", dets[0].Box, want)
	}
	if dets[0].Class != 3 {
		t.Errorf("class = %d, want 3", dets[0].Class)
	}
}

func TestNMS(t *testing.T) {
	// Two heavily overlapping same-class boxes: the lower-scoring one is dropped.
	a := Detection{Box: image.Rect(0, 0, 100, 100), Score: 0.9, Class: 0}
	b := Detection{Box: image.Rect(5, 5, 105, 105), Score: 0.8, Class: 0}
	// A distant box of the same class survives; an overlapping box of a different
	// class also survives (NMS is per-class).
	c := Detection{Box: image.Rect(500, 500, 600, 600), Score: 0.7, Class: 0}
	d := Detection{Box: image.Rect(0, 0, 100, 100), Score: 0.6, Class: 1}

	kept := NMS([]Detection{a, b, c, d}, 0.5)
	if len(kept) != 3 {
		t.Fatalf("kept %d, want 3: %+v", len(kept), kept)
	}
	for _, k := range kept {
		if k.Box == b.Box && k.Class == 0 {
			t.Errorf("expected overlapping lower-score box to be suppressed")
		}
	}
}
