// Package testing provides test utilities and helpers for tendo tensor operations.
//
// This package includes tensor comparison functions, test data generators,
// and assertion helpers to make testing tensor operations easier.
package testing

import (
	"fmt"
	"math"
	"testing"

	"github.com/zoobzio/tendo"
)

// DefaultTolerance is the default tolerance for floating-point comparisons.
const DefaultTolerance = 1e-6

// AssertTensorEqual checks that two tensors have the same shape and approximately equal values.
func AssertTensorEqual(t *testing.T, expected, actual *tendo.Tensor, tolerance float64) {
	t.Helper()

	if expected == nil && actual == nil {
		return
	}
	if expected == nil || actual == nil {
		t.Fatalf("tensor mismatch: expected=%v, actual=%v", expected, actual)
	}

	// Check shapes match
	if !ShapesEqual(expected.Shape(), actual.Shape()) {
		t.Fatalf("shape mismatch: expected %v, got %v", expected.Shape(), actual.Shape())
	}

	// Get data
	expectedData := TensorData(expected)
	actualData := TensorData(actual)

	if len(expectedData) != len(actualData) {
		t.Fatalf("data length mismatch: expected %d, got %d", len(expectedData), len(actualData))
	}

	// Compare values with tolerance
	for i := range expectedData {
		if !ApproxEqual(expectedData[i], actualData[i], tolerance) {
			t.Errorf("value mismatch at index %d: expected %v, got %v (tolerance=%v)",
				i, expectedData[i], actualData[i], tolerance)
		}
	}
}

// AssertTensorShape checks that a tensor has the expected shape.
func AssertTensorShape(t *testing.T, tensor *tendo.Tensor, expectedShape ...int) {
	t.Helper()

	if tensor == nil {
		t.Fatal("tensor is nil")
	}

	if !ShapesEqual(tensor.Shape(), expectedShape) {
		t.Fatalf("shape mismatch: expected %v, got %v", expectedShape, tensor.Shape())
	}
}

// AssertTensorValues checks that a tensor has approximately the expected values.
func AssertTensorValues(t *testing.T, tensor *tendo.Tensor, expected []float32, tolerance float64) {
	t.Helper()

	if tensor == nil {
		t.Fatal("tensor is nil")
	}

	actual := TensorData(tensor)
	if len(actual) != len(expected) {
		t.Fatalf("data length mismatch: expected %d, got %d", len(expected), len(actual))
	}

	for i := range expected {
		if !ApproxEqual(expected[i], actual[i], tolerance) {
			t.Errorf("value mismatch at index %d: expected %v, got %v", i, expected[i], actual[i])
		}
	}
}

// AssertScalar checks that a scalar tensor has the expected value.
func AssertScalar(t *testing.T, tensor *tendo.Tensor, expected float32, tolerance float64) {
	t.Helper()

	if tensor == nil {
		t.Fatal("tensor is nil")
	}

	if tensor.Numel() != 1 {
		t.Fatalf("expected scalar tensor, got shape %v", tensor.Shape())
	}

	actual := TensorData(tensor)[0]
	if !ApproxEqual(expected, actual, tolerance) {
		t.Errorf("scalar mismatch: expected %v, got %v", expected, actual)
	}
}

// TensorData extracts the logical data from a tensor as a flat slice.
// Handles non-contiguous tensors and views (slices) correctly.
func TensorData(t *tendo.Tensor) []float32 {
	if t == nil {
		return nil
	}

	// Always make contiguous to get correct logical data
	// This handles slices, transposes, and other views
	cont := t.Contiguous()

	storage := cont.Storage()
	if cpu, ok := storage.(*tendo.CPUStorage); ok {
		// Return a copy of only the logical elements
		data := cpu.Data()
		numel := cont.Numel()
		if numel > len(data) {
			numel = len(data)
		}
		result := make([]float32, numel)
		copy(result, data[:numel])
		return result
	}

	return nil
}

// ShapesEqual checks if two shapes are identical.
func ShapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ApproxEqual checks if two float32 values are approximately equal within tolerance.
func ApproxEqual(a, b float32, tolerance float64) bool {
	diff := math.Abs(float64(a - b))
	return diff <= tolerance
}

// ApproxEqualRel checks if two values are approximately equal using relative tolerance.
func ApproxEqualRel(a, b float32, relTol float64) bool {
	if a == b {
		return true
	}
	maxAbs := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	if maxAbs == 0 {
		return true
	}
	return math.Abs(float64(a-b))/maxAbs <= relTol
}

// MustTensor creates a tensor from a slice, panicking on error.
// Useful for test setup where errors indicate test bugs.
func MustTensor(data []float32, shape ...int) *tendo.Tensor {
	return tendo.MustFromSlice(data, shape...)
}

// Range creates a tensor with values from 0 to n-1.
func Range(n int) *tendo.Tensor {
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i)
	}
	return tendo.MustFromSlice(data, n)
}

// RangeWithShape creates a tensor with sequential values and the given shape.
func RangeWithShape(shape ...int) *tendo.Tensor {
	numel := 1
	for _, s := range shape {
		numel *= s
	}
	data := make([]float32, numel)
	for i := range data {
		data[i] = float32(i)
	}
	return tendo.MustFromSlice(data, shape...)
}

// Constant creates a tensor filled with a constant value.
func Constant(value float32, shape ...int) *tendo.Tensor {
	return tendo.MustFull(value, shape...)
}

// PrintTensor prints a tensor's shape and data for debugging.
func PrintTensor(t *tendo.Tensor, name string) {
	if t == nil {
		fmt.Printf("%s: nil\n", name)
		return
	}
	data := TensorData(t)
	fmt.Printf("%s: shape=%v, data=%v\n", name, t.Shape(), data)
}

// MaxAbsDiff returns the maximum absolute difference between two tensors.
func MaxAbsDiff(a, b *tendo.Tensor) float64 {
	dataA := TensorData(a)
	dataB := TensorData(b)

	if len(dataA) != len(dataB) {
		return math.Inf(1)
	}

	maxDiff := 0.0
	for i := range dataA {
		diff := math.Abs(float64(dataA[i] - dataB[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

// AllClose checks if all elements of two tensors are within tolerance.
func AllClose(a, b *tendo.Tensor, tolerance float64) bool {
	return MaxAbsDiff(a, b) <= tolerance
}
