package tendo

import (
	"errors"
	"testing"
)

func TestFromSlice(t *testing.T) {
	t.Run("1D tensor", func(t *testing.T) {
		data := []float32{1, 2, 3, 4, 5}
		tensor := MustFromSlice(data, 5)

		if tensor.Numel() != 5 {
			t.Errorf("expected 5 elements, got %d", tensor.Numel())
		}

		expectedShape := []int{5}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}

		result := tensor.MustData()
		for i, v := range data {
			if result[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, result[i])
			}
		}
	})

	t.Run("2D tensor", func(t *testing.T) {
		data := []float32{1, 2, 3, 4, 5, 6}
		tensor := MustFromSlice(data, 2, 3)

		expectedShape := []int{2, 3}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}
	})

	t.Run("3D tensor", func(t *testing.T) {
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i)
		}
		tensor := MustFromSlice(data, 2, 3, 4)

		expectedShape := []int{2, 3, 4}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}
	})

	t.Run("shape mismatch error", func(t *testing.T) {
		data := []float32{1, 2, 3}
		_, err := FromSlice(data, 2, 2) // 4 elements expected, 3 provided
		if !errors.Is(err, ErrShapeMismatch) {
			t.Errorf("expected ErrShapeMismatch, got %v", err)
		}
	})
}

func TestZeros(t *testing.T) {
	t.Run("1D zeros", func(t *testing.T) {
		tensor := MustZeros(5)

		if tensor.Numel() != 5 {
			t.Errorf("expected 5 elements, got %d", tensor.Numel())
		}

		data := tensor.MustData()
		for i, v := range data {
			if v != 0 {
				t.Errorf("at index %d: expected 0, got %v", i, v)
			}
		}
	})

	t.Run("2D zeros", func(t *testing.T) {
		tensor := MustZeros(3, 4)

		expectedShape := []int{3, 4}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}

		data := tensor.MustData()
		for i, v := range data {
			if v != 0 {
				t.Errorf("at index %d: expected 0, got %v", i, v)
			}
		}
	})
}

func TestOnes(t *testing.T) {
	t.Run("1D ones", func(t *testing.T) {
		tensor := MustOnes(5)

		data := tensor.MustData()
		for i, v := range data {
			if v != 1 {
				t.Errorf("at index %d: expected 1, got %v", i, v)
			}
		}
	})

	t.Run("2D ones", func(t *testing.T) {
		tensor := MustOnes(2, 3)

		expectedShape := []int{2, 3}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}

		data := tensor.MustData()
		for i, v := range data {
			if v != 1 {
				t.Errorf("at index %d: expected 1, got %v", i, v)
			}
		}
	})
}

func TestFull(t *testing.T) {
	t.Run("Full with value", func(t *testing.T) {
		tensor := MustFull(42.5, 3, 3)

		data := tensor.MustData()
		for i, v := range data {
			if v != 42.5 {
				t.Errorf("at index %d: expected 42.5, got %v", i, v)
			}
		}
	})

	t.Run("Full with negative value", func(t *testing.T) {
		tensor := MustFull(-3.14, 2, 2)

		data := tensor.MustData()
		for i, v := range data {
			if v != -3.14 {
				t.Errorf("at index %d: expected -3.14, got %v", i, v)
			}
		}
	})
}

func TestEmpty(t *testing.T) {
	t.Run("Empty tensor shape", func(t *testing.T) {
		tensor := MustEmpty(2, 3, 4)

		expectedShape := []int{2, 3, 4}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}

		if tensor.Numel() != 24 {
			t.Errorf("expected 24 elements, got %d", tensor.Numel())
		}
	})
}

func TestRand(t *testing.T) {
	t.Run("Rand values in range", func(t *testing.T) {
		tensor := MustRand(100)

		data := tensor.MustData()
		for i, v := range data {
			if v < 0 || v >= 1 {
				t.Errorf("at index %d: value %v not in [0, 1)", i, v)
			}
		}
	})

	t.Run("Rand shape", func(t *testing.T) {
		tensor := MustRand(3, 4, 5)

		expectedShape := []int{3, 4, 5}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}
	})
}

func TestRandN(t *testing.T) {
	t.Run("RandN shape", func(t *testing.T) {
		tensor := MustRandN(10, 10)

		expectedShape := []int{10, 10}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}
	})

	t.Run("RandN approximate mean", func(t *testing.T) {
		// Large sample should have mean near 0
		tensor := MustRandN(1000)
		data := tensor.MustData()

		sum := float32(0)
		for _, v := range data {
			sum += v
		}
		mean := sum / float32(len(data))

		// Mean should be close to 0 (within 0.2 for 1000 samples)
		if mean < -0.2 || mean > 0.2 {
			t.Errorf("expected mean near 0, got %v", mean)
		}
	})
}

func TestArange(t *testing.T) {
	t.Run("Arange basic", func(t *testing.T) {
		tensor := MustArange(0, 5, 1)

		expectedShape := []int{5}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}

		data := tensor.MustData()
		expected := []float32{0, 1, 2, 3, 4}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Arange with step", func(t *testing.T) {
		tensor := MustArange(0, 10, 2)

		data := tensor.MustData()
		expected := []float32{0, 2, 4, 6, 8}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Arange with float step", func(t *testing.T) {
		tensor := MustArange(0, 1, 0.25)

		data := tensor.MustData()
		expected := []float32{0, 0.25, 0.5, 0.75}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Arange negative step", func(t *testing.T) {
		tensor := MustArange(5, 0, -1)

		data := tensor.MustData()
		expected := []float32{5, 4, 3, 2, 1}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Arange zero step error", func(t *testing.T) {
		_, err := Arange(0, 10, 0)
		if !errors.Is(err, ErrZeroStep) {
			t.Errorf("expected ErrZeroStep, got %v", err)
		}
	})
}

func TestLinspace(t *testing.T) {
	t.Run("Linspace basic", func(t *testing.T) {
		tensor := MustLinspace(0, 1, 5)

		expectedShape := []int{5}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}

		data := tensor.MustData()
		expected := []float32{0, 0.25, 0.5, 0.75, 1}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Linspace single point", func(t *testing.T) {
		tensor := MustLinspace(5, 10, 1)

		data := tensor.MustData()
		if len(data) != 1 || data[0] != 5 {
			t.Errorf("expected [5], got %v", data)
		}
	})

	t.Run("Linspace two points", func(t *testing.T) {
		tensor := MustLinspace(0, 10, 2)

		data := tensor.MustData()
		expected := []float32{0, 10}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestEye(t *testing.T) {
	t.Run("Eye 3x3", func(t *testing.T) {
		tensor := MustEye(3)

		expectedShape := []int{3, 3}
		if !shapesEqual(tensor.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape())
		}

		data := tensor.MustData()
		expected := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Eye 1x1", func(t *testing.T) {
		tensor := MustEye(1)

		data := tensor.MustData()
		if len(data) != 1 || data[0] != 1 {
			t.Errorf("expected [1], got %v", data)
		}
	})

	t.Run("Eye 4x4", func(t *testing.T) {
		tensor := MustEye(4)

		data := tensor.MustData()
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				expected := float32(0)
				if i == j {
					expected = 1
				}
				if data[i*4+j] != expected {
					t.Errorf("at [%d,%d]: expected %v, got %v", i, j, expected, data[i*4+j])
				}
			}
		}
	})
}

func TestZerosLike(t *testing.T) {
	t.Run("ZerosLike", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		tensor := MustZerosLike(original)

		if !shapesEqual(tensor.Shape(), original.Shape()) {
			t.Errorf("expected shape %v, got %v", original.Shape(), tensor.Shape())
		}

		data := tensor.MustData()
		for i, v := range data {
			if v != 0 {
				t.Errorf("at index %d: expected 0, got %v", i, v)
			}
		}
	})
}

func TestOnesLike(t *testing.T) {
	t.Run("OnesLike", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		tensor := MustOnesLike(original)

		if !shapesEqual(tensor.Shape(), original.Shape()) {
			t.Errorf("expected shape %v, got %v", original.Shape(), tensor.Shape())
		}

		data := tensor.MustData()
		for i, v := range data {
			if v != 1 {
				t.Errorf("at index %d: expected 1, got %v", i, v)
			}
		}
	})
}

func TestEmptyLike(t *testing.T) {
	t.Run("EmptyLike", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		tensor := MustEmptyLike(original)

		if !shapesEqual(tensor.Shape(), original.Shape()) {
			t.Errorf("expected shape %v, got %v", original.Shape(), tensor.Shape())
		}

		if tensor.Numel() != original.Numel() {
			t.Errorf("expected %d elements, got %d", original.Numel(), tensor.Numel())
		}
	})
}

func TestRandLike(t *testing.T) {
	t.Run("RandLike", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		tensor := MustRandLike(original)

		if !shapesEqual(tensor.Shape(), original.Shape()) {
			t.Errorf("expected shape %v, got %v", original.Shape(), tensor.Shape())
		}

		data := tensor.MustData()
		for i, v := range data {
			if v < 0 || v >= 1 {
				t.Errorf("at index %d: value %v not in [0, 1)", i, v)
			}
		}
	})
}

func TestRandNLike(t *testing.T) {
	t.Run("RandNLike", func(t *testing.T) {
		original := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		tensor := MustRandNLike(original)

		if !shapesEqual(tensor.Shape(), original.Shape()) {
			t.Errorf("expected shape %v, got %v", original.Shape(), tensor.Shape())
		}

		if tensor.Numel() != original.Numel() {
			t.Errorf("expected %d elements, got %d", original.Numel(), tensor.Numel())
		}
	})
}

func TestDefaultDType(t *testing.T) {
	t.Run("DefaultDType is Float32", func(t *testing.T) {
		dtype := DefaultDType()
		if dtype != Float32 {
			t.Errorf("expected Float32, got %v", dtype)
		}
	})

	t.Run("SetDefaultDType", func(t *testing.T) {
		original := DefaultDType()
		defer SetDefaultDType(original)

		SetDefaultDType(Float16)

		if DefaultDType() != Float16 {
			t.Errorf("expected Float16, got %v", DefaultDType())
		}
	})
}
