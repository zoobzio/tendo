package tendo

import (
	"context"
	"testing"
)

func TestWithTraining(t *testing.T) {
	ctx := context.Background()

	// Default should be inference mode
	if IsTraining(ctx) {
		t.Error("Default context should not be in training mode")
	}

	// With training enabled
	trainCtx := WithTraining(ctx)
	if !IsTraining(trainCtx) {
		t.Error("WithTraining context should be in training mode")
	}

	// With inference explicitly set
	inferCtx := WithInference(ctx)
	if IsTraining(inferCtx) {
		t.Error("WithInference context should not be in training mode")
	}
}

func TestWithPool(t *testing.T) {
	ctx := context.Background()
	pool := NewPool()

	poolCtx := WithPool(ctx, pool)
	retrieved := PoolFromContext(poolCtx)

	if retrieved != pool {
		t.Error("PoolFromContext should return the pool set with WithPool")
	}
}

func TestPoolFromContext_Default(t *testing.T) {
	ctx := context.Background()
	pool := PoolFromContext(ctx)

	if pool == nil {
		t.Error("PoolFromContext should return default pool when none set")
	}
}

func TestWithTraceID(t *testing.T) {
	ctx := context.Background()

	// Default should be empty
	if TraceID(ctx) != "" {
		t.Error("Default TraceID should be empty")
	}

	// With trace ID set
	traceCtx := WithTraceID(ctx, "test-trace-123")
	if TraceID(traceCtx) != "test-trace-123" {
		t.Errorf("TraceID = %q, want %q", TraceID(traceCtx), "test-trace-123")
	}
}

func TestAllocCPUFromContext(t *testing.T) {
	pool := NewPool()
	ctx := WithPool(context.Background(), pool)

	storage := AllocCPUFromContext(ctx, 100, Float32)

	if storage == nil {
		t.Fatal("AllocCPUFromContext should return non-nil storage")
	}

	if storage.Len() != 100 {
		t.Errorf("Storage Len() = %d, want 100", storage.Len())
	}
}

func TestNewTensorFromPool(t *testing.T) {
	pool := NewPool()
	tensor := NewTensorFromPool(pool, []int{2, 3}, Float32)

	if tensor == nil {
		t.Fatal("NewTensorFromPool should return non-nil tensor")
	}

	if tensor.Numel() != 6 {
		t.Errorf("Tensor Numel() = %d, want 6", tensor.Numel())
	}

	if tensor.Pool() != pool {
		t.Error("Tensor should have the pool set")
	}
}

func TestNewTensorFromContext(t *testing.T) {
	pool := NewPool()
	ctx := WithPool(context.Background(), pool)

	tensor := NewTensorFromContext(ctx, []int{4, 5}, Float32)

	if tensor == nil {
		t.Fatal("NewTensorFromContext should return non-nil tensor")
	}

	if tensor.Numel() != 20 {
		t.Errorf("Tensor Numel() = %d, want 20", tensor.Numel())
	}
}

func TestDeviceError(t *testing.T) {
	err := &DeviceError{Expected: CPU, Got: CUDA}

	expected := "expected device cpu, got cuda"
	if err.Error() != expected {
		t.Errorf("Error() = %q, want %q", err.Error(), expected)
	}

	// Test Is method
	other := &DeviceError{Expected: CUDA, Got: CPU}
	if !err.Is(other) {
		t.Error("Is() should return true for any DeviceError")
	}
}

func TestDeviceMismatchError(t *testing.T) {
	err := &DeviceMismatchError{
		A: CPUDevice(),
		B: CUDADevice(0),
	}

	expected := "device mismatch: cpu vs cuda:0"
	if err.Error() != expected {
		t.Errorf("Error() = %q, want %q", err.Error(), expected)
	}

	// Test Is method
	other := &DeviceMismatchError{}
	if !err.Is(other) {
		t.Error("Is() should return true for any DeviceMismatchError")
	}
}

func TestShapeError(t *testing.T) {
	err := &ShapeError{
		Op:      "matmul",
		Message: "incompatible shapes",
		ShapeA:  []int{2, 3},
		ShapeB:  []int{4, 5},
	}

	expected := "matmul: incompatible shapes"
	if err.Error() != expected {
		t.Errorf("Error() = %q, want %q", err.Error(), expected)
	}

	// Test Is method
	other := &ShapeError{}
	if !err.Is(other) {
		t.Error("Is() should return true for any ShapeError")
	}
}
