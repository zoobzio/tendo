package testing

import (
	"testing"

	"github.com/zoobzio/tendo"
)

func TestShapesEqual(t *testing.T) {
	tests := []struct {
		name     string
		a        []int
		b        []int
		expected bool
	}{
		{"equal shapes", []int{2, 3, 4}, []int{2, 3, 4}, true},
		{"different lengths", []int{2, 3}, []int{2, 3, 4}, false},
		{"different values", []int{2, 3, 4}, []int{2, 3, 5}, false},
		{"empty shapes", []int{}, []int{}, true},
		{"nil and empty", nil, []int{}, true},
		{"single dimension", []int{5}, []int{5}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ShapesEqual(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("ShapesEqual(%v, %v) = %v, expected %v", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestApproxEqual(t *testing.T) {
	tests := []struct {
		name      string
		a         float32
		b         float32
		tolerance float64
		expected  bool
	}{
		{"exact equal", 1.0, 1.0, 1e-6, true},
		{"within tolerance", 1.0, 1.0000001, 1e-6, true},
		{"outside tolerance", 1.0, 1.001, 1e-6, false},
		{"zero values", 0.0, 0.0, 1e-6, true},
		{"negative values", -1.0, -1.0000001, 1e-6, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ApproxEqual(tt.a, tt.b, tt.tolerance)
			if result != tt.expected {
				t.Errorf("ApproxEqual(%v, %v, %v) = %v, expected %v",
					tt.a, tt.b, tt.tolerance, result, tt.expected)
			}
		})
	}
}

func TestApproxEqualRel(t *testing.T) {
	tests := []struct {
		name     string
		a        float32
		b        float32
		relTol   float64
		expected bool
	}{
		{"exact equal", 1.0, 1.0, 0.01, true},
		{"1% tolerance pass", 100.0, 100.5, 0.01, true},
		{"1% tolerance fail", 100.0, 102.0, 0.01, false},
		{"zero values", 0.0, 0.0, 0.01, true},
		{"small relative diff", 1000.0, 1005.0, 0.01, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ApproxEqualRel(tt.a, tt.b, tt.relTol)
			if result != tt.expected {
				t.Errorf("ApproxEqualRel(%v, %v, %v) = %v, expected %v",
					tt.a, tt.b, tt.relTol, result, tt.expected)
			}
		})
	}
}

func TestTensorData(t *testing.T) {
	t.Run("nil tensor", func(t *testing.T) {
		result := TensorData(nil)
		if result != nil {
			t.Errorf("TensorData(nil) = %v, expected nil", result)
		}
	})

	t.Run("simple tensor", func(t *testing.T) {
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		result := TensorData(tensor)
		expected := []float32{1, 2, 3, 4}
		if len(result) != len(expected) {
			t.Fatalf("TensorData length = %d, expected %d", len(result), len(expected))
		}
		for i := range expected {
			if result[i] != expected[i] {
				t.Errorf("TensorData[%d] = %v, expected %v", i, result[i], expected[i])
			}
		}
	})
}

func TestRange(t *testing.T) {
	tensor := Range(5)
	data := TensorData(tensor)
	expected := []float32{0, 1, 2, 3, 4}

	if len(data) != len(expected) {
		t.Fatalf("Range(5) length = %d, expected %d", len(data), len(expected))
	}

	for i := range expected {
		if data[i] != expected[i] {
			t.Errorf("Range(5)[%d] = %v, expected %v", i, data[i], expected[i])
		}
	}
}

func TestRangeWithShape(t *testing.T) {
	tensor := RangeWithShape(2, 3)

	// Check shape
	shape := tensor.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Errorf("RangeWithShape(2, 3) shape = %v, expected [2, 3]", shape)
	}

	// Check data
	data := TensorData(tensor)
	expected := []float32{0, 1, 2, 3, 4, 5}
	for i := range expected {
		if data[i] != expected[i] {
			t.Errorf("RangeWithShape(2, 3)[%d] = %v, expected %v", i, data[i], expected[i])
		}
	}
}

func TestConstant(t *testing.T) {
	tensor := Constant(3.14, 2, 2)
	data := TensorData(tensor)

	for i, v := range data {
		if v != 3.14 {
			t.Errorf("Constant(3.14, 2, 2)[%d] = %v, expected 3.14", i, v)
		}
	}
}

func TestMaxAbsDiff(t *testing.T) {
	a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)
	b := tendo.MustFromSlice([]float32{1, 2.5, 3}, 3)

	diff := MaxAbsDiff(a, b)
	if diff != 0.5 {
		t.Errorf("MaxAbsDiff = %v, expected 0.5", diff)
	}
}

func TestAllClose(t *testing.T) {
	a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)
	b := tendo.MustFromSlice([]float32{1.0001, 2.0001, 3.0001}, 3)

	if !AllClose(a, b, 0.001) {
		t.Error("AllClose should return true for tensors within tolerance")
	}

	if AllClose(a, b, 0.00001) {
		t.Error("AllClose should return false for tensors outside tolerance")
	}
}
