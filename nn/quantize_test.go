package nn

import (
	"math"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cpu"
)

func TestQuantType(t *testing.T) {
	if QuantInt8 != 8 {
		t.Errorf("QuantInt8 = %d, want 8", QuantInt8)
	}
	if QuantInt4 != 4 {
		t.Errorf("QuantInt4 = %d, want 4", QuantInt4)
	}
}

func TestQuantize_PerChannel(t *testing.T) {
	// Create a simple weight tensor [2, 4]
	data := []float32{
		1.0, 2.0, 3.0, 4.0,
		-1.0, -2.0, -3.0, -4.0,
	}
	weight, err := tendo.FromSlice(data, 2, 4)
	if err != nil {
		t.Fatalf("FromSlice() error: %v", err)
	}

	// Quantize per-channel (groupSize = 0)
	qt, err := Quantize(weight, 0)
	if err != nil {
		t.Fatalf("Quantize() error: %v", err)
	}

	// Verify structure
	if qt.Type != QuantInt8 {
		t.Errorf("Type = %d, want QuantInt8 (%d)", qt.Type, QuantInt8)
	}
	if qt.GroupSize != 0 {
		t.Errorf("GroupSize = %d, want 0", qt.GroupSize)
	}
	if len(qt.Shape) != 2 || qt.Shape[0] != 2 || qt.Shape[1] != 4 {
		t.Errorf("Shape = %v, want [2, 4]", qt.Shape)
	}
	if qt.Data == nil {
		t.Error("Data tensor is nil")
	}
	if qt.Scale == nil {
		t.Error("Scale tensor is nil")
	}

	// Verify scale shape [out_features]
	scaleShape := qt.Scale.Shape()
	if len(scaleShape) != 1 || scaleShape[0] != 2 {
		t.Errorf("Scale shape = %v, want [2]", scaleShape)
	}
}

func TestQuantize_PerGroup(t *testing.T) {
	// Create weight tensor [2, 8] with groupSize=4
	data := make([]float32, 16)
	for i := range data {
		data[i] = float32(i - 8) // Range from -8 to 7
	}
	weight, err := tendo.FromSlice(data, 2, 8)
	if err != nil {
		t.Fatalf("FromSlice() error: %v", err)
	}

	// Quantize per-group
	qt, err := Quantize(weight, 4)
	if err != nil {
		t.Fatalf("Quantize() error: %v", err)
	}

	if qt.GroupSize != 4 {
		t.Errorf("GroupSize = %d, want 4", qt.GroupSize)
	}

	// Verify scale shape [out_features, num_groups] = [2, 2]
	scaleShape := qt.Scale.Shape()
	if len(scaleShape) != 2 || scaleShape[0] != 2 || scaleShape[1] != 2 {
		t.Errorf("Scale shape = %v, want [2, 2]", scaleShape)
	}
}

func TestQuantize_InvalidDimensions(t *testing.T) {
	// 1D tensor
	data1D, _ := tendo.FromSlice([]float32{1, 2, 3}, 3)
	_, err := Quantize(data1D, 0)
	if err == nil {
		t.Error("Quantize() with 1D tensor should fail")
	}

	// 3D tensor
	data3D, _ := tendo.FromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2)
	_, err = Quantize(data3D, 0)
	if err == nil {
		t.Error("Quantize() with 3D tensor should fail")
	}
}

func TestQuantize_InvalidGroupSize(t *testing.T) {
	// Create [2, 7] tensor - 7 not divisible by group size 4
	data := make([]float32, 14)
	weight, _ := tendo.FromSlice(data, 2, 7)

	_, err := Quantize(weight, 4)
	if err == nil {
		t.Error("Quantize() with non-divisible group size should fail")
	}
}

func TestDequantize_PerChannel(t *testing.T) {
	// Create and quantize a tensor
	data := []float32{
		1.27, 2.54, 3.81, 5.08,
		-1.27, -2.54, -3.81, -5.08,
	}
	weight, _ := tendo.FromSlice(data, 2, 4)

	qt, err := Quantize(weight, 0)
	if err != nil {
		t.Fatalf("Quantize() error: %v", err)
	}

	// Dequantize
	deq, err := qt.Dequantize()
	if err != nil {
		t.Fatalf("Dequantize() error: %v", err)
	}

	// Verify shape
	deqShape := deq.Shape()
	if len(deqShape) != 2 || deqShape[0] != 2 || deqShape[1] != 4 {
		t.Errorf("Dequantized shape = %v, want [2, 4]", deqShape)
	}

	// Verify values are approximately equal (within quantization error)
	deqData, err := deq.Data()
	if err != nil {
		t.Fatalf("Data() error: %v", err)
	}

	for i := range data {
		diff := math.Abs(float64(deqData[i] - data[i]))
		tolerance := math.Abs(float64(data[i])) * 0.02 // 2% tolerance
		if tolerance < 0.1 {
			tolerance = 0.1
		}
		if diff > tolerance {
			t.Errorf("Dequantized[%d] = %f, original = %f, diff = %f (tolerance %f)",
				i, deqData[i], data[i], diff, tolerance)
		}
	}
}

func TestDequantize_PerGroup(t *testing.T) {
	data := make([]float32, 16)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	weight, _ := tendo.FromSlice(data, 2, 8)

	qt, err := Quantize(weight, 4)
	if err != nil {
		t.Fatalf("Quantize() error: %v", err)
	}

	deq, err := qt.Dequantize()
	if err != nil {
		t.Fatalf("Dequantize() error: %v", err)
	}

	deqShape := deq.Shape()
	if len(deqShape) != 2 || deqShape[0] != 2 || deqShape[1] != 8 {
		t.Errorf("Dequantized shape = %v, want [2, 8]", deqShape)
	}
}

func TestQuantize_ZeroValues(t *testing.T) {
	// All zeros should not cause division by zero
	data := make([]float32, 8)
	weight, _ := tendo.FromSlice(data, 2, 4)

	qt, err := Quantize(weight, 0)
	if err != nil {
		t.Fatalf("Quantize() with zeros error: %v", err)
	}

	deq, err := qt.Dequantize()
	if err != nil {
		t.Fatalf("Dequantize() error: %v", err)
	}

	deqData, _ := deq.Data()
	for i, v := range deqData {
		if v != 0 {
			t.Errorf("Dequantized[%d] = %f, want 0", i, v)
		}
	}
}

func TestNewQuantizedLinear(t *testing.T) {
	weight, _ := tendo.FromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)
	bias, _ := tendo.FromSlice([]float32{0.1, 0.2}, 2)

	linear, err := NewLinear(weight, bias)
	if err != nil {
		t.Fatalf("NewLinear() error: %v", err)
	}

	ql, err := NewQuantizedLinear(linear, 0)
	if err != nil {
		t.Fatalf("NewQuantizedLinear() error: %v", err)
	}

	if ql.Weight == nil {
		t.Error("QuantizedLinear.Weight is nil")
	}
	if ql.Bias == nil {
		t.Error("QuantizedLinear.Bias is nil")
	}
	if ql.InFeatures() != 4 {
		t.Errorf("InFeatures() = %d, want 4", ql.InFeatures())
	}
	if ql.OutFeatures() != 2 {
		t.Errorf("OutFeatures() = %d, want 2", ql.OutFeatures())
	}
}

func TestNewQuantizedLinearFromTensors(t *testing.T) {
	data, _ := tendo.FromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)
	scale, _ := tendo.FromSlice([]float32{0.1, 0.1}, 2)
	bias, _ := tendo.FromSlice([]float32{0.1, 0.2}, 2)

	ql := NewQuantizedLinearFromTensors(data, scale, []int{2, 4}, 0, bias)

	if ql == nil {
		t.Fatal("NewQuantizedLinearFromTensors() returned nil")
	}
	if ql.Weight.Data != data {
		t.Error("Weight.Data does not match input")
	}
	if ql.Weight.Scale != scale {
		t.Error("Weight.Scale does not match input")
	}
	if ql.Bias != bias {
		t.Error("Bias does not match input")
	}
	if ql.Weight.GroupSize != 0 {
		t.Errorf("GroupSize = %d, want 0", ql.Weight.GroupSize)
	}
}

func TestQuantizedTensor_ToDevice(t *testing.T) {
	backend := cpu.NewBackend()

	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	weight, _ := tendo.FromSlice(data, 2, 4)

	qt, err := Quantize(weight, 0)
	if err != nil {
		t.Fatalf("Quantize() error: %v", err)
	}

	qtDevice, err := qt.ToDevice(backend)
	if err != nil {
		t.Fatalf("ToDevice() error: %v", err)
	}

	if qtDevice.Data == nil {
		t.Error("ToDevice() Data is nil")
	}
	if qtDevice.Scale == nil {
		t.Error("ToDevice() Scale is nil")
	}
	if qtDevice.GroupSize != qt.GroupSize {
		t.Errorf("GroupSize = %d, want %d", qtDevice.GroupSize, qt.GroupSize)
	}
}

func TestQuantizedLinear_ToDevice(t *testing.T) {
	backend := cpu.NewBackend()

	weight, _ := tendo.FromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)
	bias, _ := tendo.FromSlice([]float32{0.1, 0.2}, 2)
	linear, _ := NewLinear(weight, bias)

	ql, err := NewQuantizedLinear(linear, 0)
	if err != nil {
		t.Fatalf("NewQuantizedLinear() error: %v", err)
	}

	qlDevice, err := ql.ToDevice(backend)
	if err != nil {
		t.Fatalf("ToDevice() error: %v", err)
	}

	if qlDevice.Weight == nil {
		t.Error("ToDevice() Weight is nil")
	}
	if qlDevice.Bias == nil {
		t.Error("ToDevice() Bias is nil")
	}
	if qlDevice.InFeatures() != 4 {
		t.Errorf("InFeatures() = %d, want 4", qlDevice.InFeatures())
	}
}
