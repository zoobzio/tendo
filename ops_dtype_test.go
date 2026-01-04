package tendo_test

import (
	"context"
	"math"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cuda"
)

func TestToFloat16(t *testing.T) {
	ctx := context.Background()

	// Create a float32 tensor
	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := tendo.MustFromSlice(data, 2, 2)

	// Convert to float16
	result, err := tendo.ToFloat16().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("ToFloat16 failed: %v", err)
	}

	// Check dtype
	if result.DType() != tendo.Float16 {
		t.Errorf("Expected dtype Float16, got %v", result.DType())
	}

	// Check shape preserved
	if !shapesEqual(result.Shape(), []int{2, 2}) {
		t.Errorf("Shape mismatch: got %v, expected [2, 2]", result.Shape())
	}

	// Check data preserved via public API
	resultData := result.MustData()
	for i, v := range resultData {
		if math.Abs(float64(v-data[i])) > 0.01 {
			t.Errorf("Data mismatch at index %d: got %v, expected %v", i, v, data[i])
		}
	}
}

func TestToBFloat16(t *testing.T) {
	ctx := context.Background()

	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := tendo.MustFromSlice(data, 2, 2)

	result, err := tendo.ToBFloat16().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("ToBFloat16 failed: %v", err)
	}

	if result.DType() != tendo.BFloat16 {
		t.Errorf("Expected dtype BFloat16, got %v", result.DType())
	}

	if !shapesEqual(result.Shape(), []int{2, 2}) {
		t.Errorf("Shape mismatch: got %v, expected [2, 2]", result.Shape())
	}
}

func TestToFloat32FromFloat16(t *testing.T) {
	ctx := context.Background()

	// Create a float32 tensor, convert to float16, then back to float32
	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := tendo.MustFromSlice(data, 2, 2)

	// Convert to float16
	f16, err := tendo.ToFloat16().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("ToFloat16 failed: %v", err)
	}

	// Convert back to float32
	result, err := tendo.ToFloat32().Process(ctx, f16)
	if err != nil {
		t.Fatalf("ToFloat32 failed: %v", err)
	}

	if result.DType() != tendo.Float32 {
		t.Errorf("Expected dtype Float32, got %v", result.DType())
	}

	resultData := result.MustData()
	for i, v := range resultData {
		if math.Abs(float64(v-data[i])) > 0.01 {
			t.Errorf("Data mismatch at index %d: got %v, expected %v", i, v, data[i])
		}
	}
}

func TestToSameDtype(t *testing.T) {
	ctx := context.Background()

	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := tendo.MustFromSlice(data, 2, 2)

	// Converting to same dtype should return same tensor
	result, err := tendo.ToFloat32().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("ToFloat32 failed: %v", err)
	}

	// Should be the same tensor (no conversion needed)
	if result != tensor {
		t.Error("Expected same tensor when converting to same dtype")
	}
}

func TestToDTypeGeneric(t *testing.T) {
	ctx := context.Background()

	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := tendo.MustFromSlice(data, 2, 2)

	// Use generic ToDType() function
	result, err := tendo.ToDType(tendo.Float16).Process(ctx, tensor)
	if err != nil {
		t.Fatalf("ToDType(Float16) failed: %v", err)
	}

	if result.DType() != tendo.Float16 {
		t.Errorf("Expected dtype Float16, got %v", result.DType())
	}
}

func TestDTypeAliases(t *testing.T) {
	ctx := context.Background()

	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := tendo.MustFromSlice(data, 2, 2)

	// Test Half() alias
	result, err := tendo.Half().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("Half() failed: %v", err)
	}
	if result.DType() != tendo.Float16 {
		t.Errorf("Half() expected Float16, got %v", result.DType())
	}

	// Test BFloat() alias
	result, err = tendo.BFloat().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("BFloat() failed: %v", err)
	}
	if result.DType() != tendo.BFloat16 {
		t.Errorf("BFloat() expected BFloat16, got %v", result.DType())
	}

	// Test Float() alias
	result, err = tendo.Float().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("Float() failed: %v", err)
	}
	if result.DType() != tendo.Float32 {
		t.Errorf("Float() expected Float32, got %v", result.DType())
	}
}

func TestDTypeConversionChain(t *testing.T) {
	ctx := context.Background()

	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := tendo.MustFromSlice(data, 2, 2)

	// Chain multiple dtype conversions
	f16, err := tendo.ToFloat16().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("ToFloat16 failed: %v", err)
	}

	bf16, err := tendo.ToBFloat16().Process(ctx, f16)
	if err != nil {
		t.Fatalf("ToBFloat16 failed: %v", err)
	}

	result, err := tendo.ToFloat32().Process(ctx, bf16)
	if err != nil {
		t.Fatalf("ToFloat32 failed: %v", err)
	}

	if result.DType() != tendo.Float32 {
		t.Errorf("Expected dtype Float32, got %v", result.DType())
	}

	resultData := result.MustData()
	for i, v := range resultData {
		if math.Abs(float64(v-data[i])) > 0.01 {
			t.Errorf("Data mismatch at index %d: got %v, expected %v", i, v, data[i])
		}
	}
}

func TestGPUDtypeFloat16(t *testing.T) {
	if !cuda.IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}

	ctx := context.Background()

	// Create a CPU tensor and move to GPU
	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := tendo.MustFromSlice(data, 2, 2)

	// Convert to float16
	f16, err := tendo.ToFloat16().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("ToFloat16 failed: %v", err)
	}

	// Move to GPU
	gpuTensor, err := tendo.ToGPU(0).Process(ctx, f16)
	if err != nil {
		t.Fatalf("GPU transfer failed: %v", err)
	}

	if gpuTensor.Device().Type != tendo.CUDA {
		t.Errorf("Expected CUDA device, got %v", gpuTensor.Device())
	}

	if gpuTensor.DType() != tendo.Float16 {
		t.Errorf("Expected Float16 dtype, got %v", gpuTensor.DType())
	}

	// Copy back to CPU and verify
	cpuTensor, err := tendo.ToCPU().Process(ctx, gpuTensor)
	if err != nil {
		t.Fatalf("CPU transfer failed: %v", err)
	}

	resultData := cpuTensor.MustData()
	for i, v := range resultData {
		diff := math.Abs(float64(v - data[i]))
		if diff > 0.01 { // Float16 has limited precision
			t.Errorf("Data mismatch at index %d: got %v, expected %v (diff=%v)", i, v, data[i], diff)
		}
	}
}

func TestGPUDtypeBFloat16(t *testing.T) {
	if !cuda.IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}

	ctx := context.Background()

	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := tendo.MustFromSlice(data, 2, 2)

	// Convert to bfloat16
	bf16, err := tendo.ToBFloat16().Process(ctx, tensor)
	if err != nil {
		t.Fatalf("ToBFloat16 failed: %v", err)
	}

	// Move to GPU
	gpuTensor, err := tendo.ToGPU(0).Process(ctx, bf16)
	if err != nil {
		t.Fatalf("GPU transfer failed: %v", err)
	}

	if gpuTensor.DType() != tendo.BFloat16 {
		t.Errorf("Expected BFloat16 dtype, got %v", gpuTensor.DType())
	}

	// Copy back and verify
	cpuTensor, err := tendo.ToCPU().Process(ctx, gpuTensor)
	if err != nil {
		t.Fatalf("CPU transfer failed: %v", err)
	}

	resultData := cpuTensor.MustData()
	for i, v := range resultData {
		diff := math.Abs(float64(v - data[i]))
		if diff > 0.01 {
			t.Errorf("Data mismatch at index %d: got %v, expected %v (diff=%v)", i, v, data[i], diff)
		}
	}
}
