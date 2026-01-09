package tendo

import (
	"testing"
)

func TestNumel(t *testing.T) {
	t.Run("Empty shape (scalar)", func(t *testing.T) {
		result := Numel([]int{})
		if result != 1 {
			t.Errorf("expected 1, got %d", result)
		}
	})

	t.Run("1D shape", func(t *testing.T) {
		result := Numel([]int{5})
		if result != 5 {
			t.Errorf("expected 5, got %d", result)
		}
	})

	t.Run("2D shape", func(t *testing.T) {
		result := Numel([]int{3, 4})
		if result != 12 {
			t.Errorf("expected 12, got %d", result)
		}
	})

	t.Run("3D shape", func(t *testing.T) {
		result := Numel([]int{2, 3, 4})
		if result != 24 {
			t.Errorf("expected 24, got %d", result)
		}
	})

	t.Run("Shape with 1s", func(t *testing.T) {
		result := Numel([]int{1, 5, 1, 3, 1})
		if result != 15 {
			t.Errorf("expected 15, got %d", result)
		}
	})
}

func TestComputeStrides(t *testing.T) {
	t.Run("Empty shape", func(t *testing.T) {
		result := ComputeStrides([]int{})
		if len(result) != 0 {
			t.Errorf("expected empty strides, got %v", result)
		}
	})

	t.Run("1D shape", func(t *testing.T) {
		result := ComputeStrides([]int{5})
		expected := []int{1}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("2D shape", func(t *testing.T) {
		result := ComputeStrides([]int{3, 4})
		expected := []int{4, 1}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("3D shape", func(t *testing.T) {
		result := ComputeStrides([]int{2, 3, 4})
		expected := []int{12, 4, 1}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("4D shape", func(t *testing.T) {
		result := ComputeStrides([]int{2, 3, 4, 5})
		expected := []int{60, 20, 5, 1}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})
}

func TestIsContiguous(t *testing.T) {
	t.Run("Contiguous 2D", func(t *testing.T) {
		shape := []int{3, 4}
		stride := []int{4, 1}
		if !IsContiguous(shape, stride) {
			t.Error("expected contiguous")
		}
	})

	t.Run("Non-contiguous (transposed)", func(t *testing.T) {
		shape := []int{4, 3}
		stride := []int{1, 4} // transposed strides
		if IsContiguous(shape, stride) {
			t.Error("expected non-contiguous")
		}
	})

	t.Run("Empty shape is contiguous", func(t *testing.T) {
		if !IsContiguous([]int{}, []int{}) {
			t.Error("expected empty to be contiguous")
		}
	})

	t.Run("Mismatched lengths", func(t *testing.T) {
		if IsContiguous([]int{3, 4}, []int{4}) {
			t.Error("expected false for mismatched lengths")
		}
	})
}

func TestInferShape(t *testing.T) {
	t.Run("No inference needed", func(t *testing.T) {
		result, err := InferShape(12, []int{3, 4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Infer first dimension", func(t *testing.T) {
		result, err := InferShape(12, []int{-1, 4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Infer last dimension", func(t *testing.T) {
		result, err := InferShape(12, []int{3, -1})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Infer middle dimension", func(t *testing.T) {
		result, err := InferShape(24, []int{2, -1, 4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{2, 3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Multiple -1 error", func(t *testing.T) {
		_, err := InferShape(12, []int{-1, -1})
		if err == nil {
			t.Error("expected error for multiple -1")
		}
	})

	t.Run("Invalid dimension error", func(t *testing.T) {
		_, err := InferShape(12, []int{3, 0})
		if err == nil {
			t.Error("expected error for zero dimension")
		}
	})

	t.Run("Non-divisible error", func(t *testing.T) {
		_, err := InferShape(13, []int{-1, 4})
		if err == nil {
			t.Error("expected error for non-divisible")
		}
	})
}

func TestBroadcastShapes(t *testing.T) {
	t.Run("Same shape", func(t *testing.T) {
		result, err := BroadcastShapes([]int{3, 4}, []int{3, 4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Scalar broadcast", func(t *testing.T) {
		result, err := BroadcastShapes([]int{3, 4}, []int{1})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Row broadcast", func(t *testing.T) {
		result, err := BroadcastShapes([]int{3, 4}, []int{1, 4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Column broadcast", func(t *testing.T) {
		result, err := BroadcastShapes([]int{3, 4}, []int{3, 1})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Different dimensions", func(t *testing.T) {
		result, err := BroadcastShapes([]int{3, 4}, []int{4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("3D broadcast", func(t *testing.T) {
		result, err := BroadcastShapes([]int{2, 3, 4}, []int{1, 3, 1})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{2, 3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Incompatible error", func(t *testing.T) {
		_, err := BroadcastShapes([]int{3, 4}, []int{3, 5})
		if err == nil {
			t.Error("expected error for incompatible shapes")
		}
	})
}

func TestCanBroadcast(t *testing.T) {
	t.Run("Same shape", func(t *testing.T) {
		if !CanBroadcast([]int{3, 4}, []int{3, 4}) {
			t.Error("expected broadcastable")
		}
	})

	t.Run("With ones", func(t *testing.T) {
		if !CanBroadcast([]int{3, 4}, []int{1, 4}) {
			t.Error("expected broadcastable")
		}
	})

	t.Run("Different dims", func(t *testing.T) {
		if !CanBroadcast([]int{3, 4}, []int{4}) {
			t.Error("expected broadcastable")
		}
	})

	t.Run("Incompatible", func(t *testing.T) {
		if CanBroadcast([]int{3, 4}, []int{3, 5}) {
			t.Error("expected not broadcastable")
		}
	})
}

func TestValidateMatMul(t *testing.T) {
	t.Run("Basic 2D matmul", func(t *testing.T) {
		result, err := ValidateMatMul([]int{2, 3}, []int{3, 4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{2, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Batched matmul same batch", func(t *testing.T) {
		result, err := ValidateMatMul([]int{5, 2, 3}, []int{5, 3, 4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{5, 2, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Batched matmul broadcast", func(t *testing.T) {
		result, err := ValidateMatMul([]int{5, 2, 3}, []int{1, 3, 4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{5, 2, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Inner dimension mismatch", func(t *testing.T) {
		_, err := ValidateMatMul([]int{2, 3}, []int{4, 5})
		if err == nil {
			t.Error("expected error for inner dimension mismatch")
		}
	})

	t.Run("1D tensor error", func(t *testing.T) {
		_, err := ValidateMatMul([]int{3}, []int{3, 4})
		if err == nil {
			t.Error("expected error for 1D tensor")
		}
	})
}

func TestValidateElementwise(t *testing.T) {
	t.Run("Same shape", func(t *testing.T) {
		result, err := ValidateElementwise([]int{3, 4}, []int{3, 4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Broadcast", func(t *testing.T) {
		result, err := ValidateElementwise([]int{3, 4}, []int{1})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})
}

func TestTransposeShape(t *testing.T) {
	t.Run("Transpose 2D", func(t *testing.T) {
		result, err := TransposeShape([]int{3, 4}, 0, 1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{4, 3}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Transpose 3D", func(t *testing.T) {
		result, err := TransposeShape([]int{2, 3, 4}, 0, 2)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{4, 3, 2}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Negative dims", func(t *testing.T) {
		result, err := TransposeShape([]int{2, 3, 4}, -2, -1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{2, 4, 3}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Out of range error", func(t *testing.T) {
		_, err := TransposeShape([]int{3, 4}, 0, 5)
		if err == nil {
			t.Error("expected error for out of range dimension")
		}
	})
}

func TestTransposeStride(t *testing.T) {
	t.Run("Transpose stride 2D", func(t *testing.T) {
		result, err := TransposeStride([]int{4, 1}, 0, 1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{1, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})
}

func TestSqueezeShape(t *testing.T) {
	t.Run("Squeeze dim 0", func(t *testing.T) {
		result, err := SqueezeShape([]int{1, 3, 4}, 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Squeeze dim -1", func(t *testing.T) {
		result, err := SqueezeShape([]int{3, 4, 1}, -1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Cannot squeeze non-1 dim", func(t *testing.T) {
		_, err := SqueezeShape([]int{3, 4}, 0)
		if err == nil {
			t.Error("expected error for non-1 dimension")
		}
	})
}

func TestUnsqueezeShape(t *testing.T) {
	t.Run("Unsqueeze at 0", func(t *testing.T) {
		result, err := UnsqueezeShape([]int{3, 4}, 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{1, 3, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Unsqueeze at end", func(t *testing.T) {
		result, err := UnsqueezeShape([]int{3, 4}, 2)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4, 1}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Unsqueeze middle", func(t *testing.T) {
		result, err := UnsqueezeShape([]int{3, 4}, 1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 1, 4}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})

	t.Run("Unsqueeze negative", func(t *testing.T) {
		result, err := UnsqueezeShape([]int{3, 4}, -1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := []int{3, 4, 1}
		if !shapesEqual(result, expected) {
			t.Errorf("expected %v, got %v", expected, result)
		}
	})
}
