package tendo_test

import (
	"context"
	"errors"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/pkg/cuda"
)

func TestTo(t *testing.T) {
	ctx := context.Background()

	t.Run("To same device returns same tensor", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		cpuDevice := tendo.Device{Type: tendo.CPU, Index: 0}

		result, err := tendo.To(cpuDevice).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result != a {
			t.Error("To same device should return same tensor")
		}
	})

	t.Run("To different CPU index clones", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		cpuDevice := tendo.Device{Type: tendo.CPU, Index: 1}

		result, err := tendo.To(cpuDevice).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should be different tensor (cloned)
		if result == a {
			t.Error("To different CPU should clone")
		}

		// Data should be same
		origData := a.MustData()
		resultData := result.MustData()
		for i := range origData {
			if origData[i] != resultData[i] {
				t.Errorf("at index %d: expected %v, got %v", i, origData[i], resultData[i])
			}
		}
	})

	t.Run("To CUDA returns error when CUDA unavailable", func(t *testing.T) {
		if cuda.IsCUDAAvailable() {
			t.Skip("CUDA is available, skipping stub test")
		}

		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)

		_, err := tendo.ToGPU(0).Process(ctx, a)
		if err == nil {
			t.Error("expected error when CUDA unavailable")
		}
	})
}

func TestToCPU(t *testing.T) {
	ctx := context.Background()

	t.Run("ToCPU on CPU tensor", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)

		result, err := tendo.ToCPU().Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result != a {
			t.Error("ToCPU on CPU tensor should return same tensor")
		}
	})
}

func TestToGPU(t *testing.T) {
	ctx := context.Background()

	t.Run("ToGPU returns error when CUDA unavailable", func(t *testing.T) {
		if cuda.IsCUDAAvailable() {
			t.Skip("CUDA is available, skipping stub test")
		}

		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)

		_, err := tendo.ToGPU(0).Process(ctx, a)
		if err == nil {
			t.Error("expected error when CUDA unavailable")
		}

		// Should be the specific CUDA error (wrapped by pipz)
		if !errors.Is(err, tendo.ErrCUDANotAvailable) {
			t.Errorf("expected ErrCUDANotAvailable, got %v", err)
		}
	})
}

func TestMakeContiguous(t *testing.T) {
	ctx := context.Background()

	t.Run("MakeContiguous on contiguous tensor", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

		result, err := tendo.MakeContiguous().Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result != a {
			t.Error("MakeContiguous on contiguous tensor should return same tensor")
		}
	})

	t.Run("MakeContiguous on non-contiguous tensor", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		// Create non-contiguous view (transposed)
		transposed, err := tendo.NewPermute(1, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error during transpose: %v", err)
		}

		result, err := tendo.MakeContiguous().Process(ctx, transposed)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result == transposed {
			t.Error("MakeContiguous on non-contiguous should create new tensor")
		}

		if !result.IsContiguous() {
			t.Error("result should be contiguous")
		}
	})
}

func TestSync(t *testing.T) {
	ctx := context.Background()

	t.Run("Sync on CPU is no-op", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)

		result, err := tendo.Sync().Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result != a {
			t.Error("Sync on CPU should return same tensor")
		}
	})
}

func TestIsCUDAAvailable(t *testing.T) {
	t.Run("IsCUDAAvailable in stub mode", func(t *testing.T) {
		// In stub mode (no -tags cuda), should return false
		// This test documents expected behavior
		available := cuda.IsCUDAAvailable()
		t.Logf("IsCUDAAvailable: %v", available)
	})
}

func TestCUDADeviceCount(t *testing.T) {
	t.Run("CUDADeviceCount in stub mode", func(t *testing.T) {
		count := cuda.CUDADeviceCount()
		if !cuda.IsCUDAAvailable() && count != 0 {
			t.Errorf("expected 0 devices when CUDA unavailable, got %d", count)
		}
	})
}
