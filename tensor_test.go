package tendo

import (
	"testing"
)

func TestNewTensor(t *testing.T) {
	t.Run("With nil stride", func(t *testing.T) {
		storage := NewCPUStorage(12, Float32)
		tensor := NewTensor(storage, []int{3, 4}, nil)

		expectedShape := []int{3, 4}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}

		expectedStride := []int{4, 1}
		if !shapesEqual(tensor.Stride(), expectedStride) {
			t.Errorf("expected stride %v, got %v", expectedStride, tensor.Stride())
		}
	})

	t.Run("With custom stride", func(t *testing.T) {
		storage := NewCPUStorage(12, Float32)
		customStride := []int{1, 3} // transposed
		tensor := NewTensor(storage, []int{3, 4}, customStride)

		if !shapesEqual(tensor.Stride(), customStride) {
			t.Errorf("expected stride %v, got %v", customStride, tensor.Stride())
		}
	})
}

func TestTensorShape(t *testing.T) {
	t.Run("Shape returns copy", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		shape := tensor.Shape()
		shape[0] = 999

		// Original should be unchanged
		if tensor.Shape()[0] != 3 {
			t.Error("Shape should return a copy")
		}
	})
}

func TestTensorStride(t *testing.T) {
	t.Run("Stride returns copy", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		stride := tensor.Stride()
		stride[0] = 999

		// Original should be unchanged
		if tensor.Stride()[0] != 4 {
			t.Error("Stride should return a copy")
		}
	})
}

func TestTensorDim(t *testing.T) {
	t.Run("0D tensor", func(t *testing.T) {
		tensor := MustFromSlice([]float32{1})
		reshaped, err := tensor.Reshape()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if reshaped.Dim() != 0 {
			t.Errorf("expected 0 dimensions, got %d", reshaped.Dim())
		}
	})

	t.Run("1D tensor", func(t *testing.T) {
		tensor := MustZeros(5)
		if tensor.Dim() != 1 {
			t.Errorf("expected 1 dimension, got %d", tensor.Dim())
		}
	})

	t.Run("3D tensor", func(t *testing.T) {
		tensor := MustZeros(2, 3, 4)
		if tensor.Dim() != 3 {
			t.Errorf("expected 3 dimensions, got %d", tensor.Dim())
		}
	})
}

func TestTensorSize(t *testing.T) {
	t.Run("Positive index", func(t *testing.T) {
		tensor := MustZeros(2, 3, 4)
		if tensor.Size(0) != 2 {
			t.Errorf("expected 2, got %d", tensor.Size(0))
		}
		if tensor.Size(1) != 3 {
			t.Errorf("expected 3, got %d", tensor.Size(1))
		}
		if tensor.Size(2) != 4 {
			t.Errorf("expected 4, got %d", tensor.Size(2))
		}
	})

	t.Run("Negative index", func(t *testing.T) {
		tensor := MustZeros(2, 3, 4)
		if tensor.Size(-1) != 4 {
			t.Errorf("expected 4, got %d", tensor.Size(-1))
		}
		if tensor.Size(-2) != 3 {
			t.Errorf("expected 3, got %d", tensor.Size(-2))
		}
	})

	t.Run("Out of range", func(t *testing.T) {
		tensor := MustZeros(2, 3)
		if tensor.Size(5) != 0 {
			t.Errorf("expected 0 for out of range, got %d", tensor.Size(5))
		}
	})
}

func TestTensorNumel(t *testing.T) {
	t.Run("1D", func(t *testing.T) {
		tensor := MustZeros(5)
		if tensor.Numel() != 5 {
			t.Errorf("expected 5, got %d", tensor.Numel())
		}
	})

	t.Run("3D", func(t *testing.T) {
		tensor := MustZeros(2, 3, 4)
		if tensor.Numel() != 24 {
			t.Errorf("expected 24, got %d", tensor.Numel())
		}
	})
}

func TestTensorDevice(t *testing.T) {
	t.Run("CPU tensor", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		device := tensor.Device()
		if device.Type != CPU {
			t.Errorf("expected CPU, got %v", device.Type)
		}
	})
}

func TestTensorDType(t *testing.T) {
	t.Run("Default dtype", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		if tensor.DType() != Float32 {
			t.Errorf("expected Float32, got %v", tensor.DType())
		}
	})
}

func TestTensorStorage(t *testing.T) {
	t.Run("Returns storage", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		storage := tensor.Storage()
		if storage == nil {
			t.Error("expected non-nil storage")
		}
		if storage.Len() != 12 {
			t.Errorf("expected 12 elements, got %d", storage.Len())
		}
	})
}

func TestTensorOffset(t *testing.T) {
	t.Run("Fresh tensor has zero offset", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		if tensor.Offset() != 0 {
			t.Errorf("expected 0 offset, got %d", tensor.Offset())
		}
	})
}

func TestTensorIsContiguous(t *testing.T) {
	t.Run("Fresh tensor is contiguous", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		if !tensor.IsContiguous() {
			t.Error("expected contiguous")
		}
	})

	t.Run("Transposed tensor is not contiguous", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		// Manually create transposed view
		transposed := &Tensor{
			storage: tensor.storage,
			shape:   []int{4, 3},
			stride:  []int{1, 4},
			offset:  0,
		}
		if transposed.IsContiguous() {
			t.Error("expected non-contiguous")
		}
	})
}

func TestTensorClone(t *testing.T) {
	t.Run("Clone creates independent copy", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		cloned := original.Clone()

		// Modify original
		original.MustData()[0] = 999

		// Clone should be unchanged
		if cloned.MustData()[0] == 999 {
			t.Error("clone should be independent of original")
		}
	})

	t.Run("Clone preserves shape", func(t *testing.T) {
		original := MustZeros(2, 3, 4)
		cloned := original.Clone()

		if !shapesEqual(cloned.Shape(), original.Shape()) {
			t.Errorf("expected shape %v, got %v", original.Shape(), cloned.Shape())
		}
	})

	t.Run("Clone preserves data", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		cloned := original.Clone()

		origData := original.MustData()
		clonedData := cloned.MustData()

		for i := range origData {
			if origData[i] != clonedData[i] {
				t.Errorf("at index %d: expected %v, got %v", i, origData[i], clonedData[i])
			}
		}
	})
}

func TestTensorFree(t *testing.T) {
	t.Run("Free sets storage to nil", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		tensor.Free()

		if tensor.storage != nil {
			t.Error("expected nil storage after Free")
		}
	})

	t.Run("Double free is safe", func(t *testing.T) {
		tensor := MustZeros(3, 4)
		tensor.Free()
		tensor.Free() // should not panic
	})
}

func TestTensorString(t *testing.T) {
	t.Run("String representation", func(t *testing.T) {
		tensor := MustZeros(2, 3)
		str := tensor.String()

		if str == "" {
			t.Error("expected non-empty string")
		}
	})
}

func TestTensorViewMethod(t *testing.T) {
	t.Run("View creates shared storage", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)
		viewed, err := original.View(2, 3)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if viewed.Storage() != original.Storage() {
			t.Error("view should share storage")
		}

		expectedShape := []int{2, 3}
		if !shapesEqual(viewed.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, viewed.Shape())
		}
	})

	t.Run("View with -1 inference", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)
		viewed, err := original.View(2, -1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3}
		if !shapesEqual(viewed.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, viewed.Shape())
		}
	})

	t.Run("View incompatible shape error", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)
		_, err := original.View(2, 4)
		if err == nil {
			t.Error("expected error for incompatible shape")
		}
	})

	t.Run("View non-contiguous error", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		// Create non-contiguous by manually setting strides
		transposed := &Tensor{
			storage: original.storage,
			shape:   []int{3, 2},
			stride:  []int{1, 3},
			offset:  0,
		}
		_, err := transposed.View(6)
		if err == nil {
			t.Error("expected error for non-contiguous tensor")
		}
	})
}

func TestTensorReshapeMethod(t *testing.T) {
	t.Run("Reshape contiguous tensor", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)
		reshaped, err := original.Reshape(2, 3)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should share storage for contiguous
		if reshaped.Storage() != original.Storage() {
			t.Error("reshape of contiguous should share storage")
		}
	})

	t.Run("Reshape non-contiguous copies", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		// Create non-contiguous
		transposed := &Tensor{
			storage: original.storage,
			shape:   []int{3, 2},
			stride:  []int{1, 3},
			offset:  0,
		}

		reshaped, err := transposed.Reshape(6)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should not share storage (had to copy)
		if reshaped.Storage() == transposed.Storage() {
			t.Error("reshape of non-contiguous should copy")
		}
	})
}

func TestTensorContiguous(t *testing.T) {
	t.Run("Already contiguous returns same", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		cont := original.Contiguous()

		if cont != original {
			t.Error("contiguous tensor should return itself")
		}
	})

	t.Run("Non-contiguous creates copy", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		// Create non-contiguous (transposed)
		transposed := &Tensor{
			storage: original.storage,
			shape:   []int{3, 2},
			stride:  []int{1, 3},
			offset:  0,
		}

		cont := transposed.Contiguous()

		if cont.Storage() == transposed.Storage() {
			t.Error("contiguous should create new storage for non-contiguous")
		}

		// Should be contiguous now
		if !cont.IsContiguous() {
			t.Error("result should be contiguous")
		}

		// Check values are in correct order
		// Original: [[1,2,3],[4,5,6]] -> transposed [[1,4],[2,5],[3,6]]
		// Contiguous should be [1,4,2,5,3,6]
		data := cont.MustData()
		expected := []float32{1, 4, 2, 5, 3, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Tensor with offset creates copy", func(t *testing.T) {
		original := MustFromSlice([]float32{0, 1, 2, 3, 4, 5}, 6)
		// Create sliced tensor with offset
		sliced := &Tensor{
			storage: original.storage,
			shape:   []int{3},
			stride:  []int{1},
			offset:  2,
		}

		cont := sliced.Contiguous()

		// Should create new storage
		if cont.Storage() == sliced.Storage() {
			t.Error("tensor with offset should create new storage")
		}

		// Values should be [2, 3, 4]
		data := cont.MustData()
		expected := []float32{2, 3, 4}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}
