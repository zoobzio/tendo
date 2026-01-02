package tendo_test

import (
	"context"
	"testing"

	"github.com/zoobzio/tendo"
)

func TestReshape(t *testing.T) {
	ctx := context.Background()

	t.Run("Reshape 1D to 2D", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)

		result, err := tendo.NewReshape(2, 3).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Data should be unchanged
		data := result.MustData()
		expected := []float32{1, 2, 3, 4, 5, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Reshape with -1 inference", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)

		result, err := tendo.NewReshape(2, -1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("Reshape to scalar-like", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{42}, 1)

		result, err := tendo.NewReshape().Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Empty shape = scalar
		if len(result.Shape()) != 0 {
			t.Errorf("expected scalar shape [], got %v", result.Shape())
		}
	})
}

func TestView(t *testing.T) {
	ctx := context.Background()

	t.Run("View creates shared storage", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)

		result, err := tendo.NewView(2, 3).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should share storage
		if result.Storage() != a.Storage() {
			t.Error("view should share storage with original")
		}
	})
}

func TestSqueeze(t *testing.T) {
	ctx := context.Background()

	t.Run("Squeeze all", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 1, 3, 1)

		result, err := tendo.NewSqueeze().Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("Squeeze specific dim", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 1, 3, 1)

		result, err := tendo.NewSqueezeDim(0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 1}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("Squeeze non-1 dim does nothing", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 1, 3, 1)

		result, err := tendo.NewSqueezeDim(1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{1, 3, 1}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})
}

func TestUnsqueeze(t *testing.T) {
	ctx := context.Background()

	t.Run("Unsqueeze at 0", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)

		result, err := tendo.NewUnsqueeze(0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{1, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("Unsqueeze at end", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)

		result, err := tendo.NewUnsqueeze(1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 1}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("Unsqueeze negative dim", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)

		result, err := tendo.NewUnsqueeze(-1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 1}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})
}

func TestFlatten(t *testing.T) {
	ctx := context.Background()

	t.Run("Flatten all", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

		result, err := tendo.NewFlatten(0, 1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{6}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("Flatten partial", func(t *testing.T) {
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i)
		}
		a := tendo.MustFromSlice(data, 2, 3, 4)

		result, err := tendo.NewFlatten(1, 2).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 12}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})
}

func TestSlice(t *testing.T) {
	ctx := context.Background()

	t.Run("Slice 1D", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{0, 1, 2, 3, 4, 5}, 6)

		result, err := tendo.NewSlice(0, 2, 5).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Make contiguous to get values
		cont := result.Contiguous()
		data := cont.MustData()
		expected := []float32{2, 3, 4}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Slice 2D rows", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)

		result, err := tendo.NewSlice(0, 1, 3).Process(ctx, a) // rows 1 and 2
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		cont := result.Contiguous()
		data := cont.MustData()
		expected := []float32{4, 5, 6, 7, 8, 9}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Slice 2D cols", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)

		result, err := tendo.NewSlice(1, 0, 2).Process(ctx, a) // cols 0 and 1
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		cont := result.Contiguous()
		data := cont.MustData()
		expected := []float32{1, 2, 4, 5, 7, 8}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestNarrow(t *testing.T) {
	ctx := context.Background()

	t.Run("Narrow 1D", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{0, 1, 2, 3, 4, 5}, 6)

		result, err := tendo.NewNarrow(0, 1, 3).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		cont := result.Contiguous()
		data := cont.MustData()
		expected := []float32{1, 2, 3}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Narrow 2D", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)

		result, err := tendo.NewNarrow(0, 0, 2).Process(ctx, a) // first 2 rows
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		cont := result.Contiguous()
		data := cont.MustData()
		expected := []float32{1, 2, 3, 4, 5, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Narrow equivalent to Slice", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{0, 1, 2, 3, 4, 5}, 6)

		narrow, err := tendo.NewNarrow(0, 2, 3).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		slice, err := tendo.NewSlice(0, 2, 5).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !shapesEqual(narrow.Shape(), slice.Shape()) {
			t.Errorf("Narrow and Slice shapes differ: %v vs %v", narrow.Shape(), slice.Shape())
		}

		narrowData := narrow.Contiguous().MustData()
		sliceData := slice.Contiguous().MustData()
		for i := range narrowData {
			if narrowData[i] != sliceData[i] {
				t.Errorf("at index %d: Narrow=%v, Slice=%v", i, narrowData[i], sliceData[i])
			}
		}
	})
}

func TestExpand(t *testing.T) {
	ctx := context.Background()

	t.Run("Expand singleton", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 1, 3)

		result, err := tendo.NewExpand(4, 3).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{4, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Should share storage (stride 0 for expanded dim)
		if result.Storage() != a.Storage() {
			t.Error("expand should share storage")
		}

		// Make contiguous to verify values
		cont := result.Contiguous()
		data := cont.MustData()
		expected := []float32{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestPermute(t *testing.T) {
	ctx := context.Background()

	t.Run("Permute 3D", func(t *testing.T) {
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i)
		}
		a := tendo.MustFromSlice(data, 2, 3, 4)

		// Permute to [4, 2, 3]
		result, err := tendo.NewPermute(2, 0, 1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{4, 2, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("Permute is inverse of itself", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

		// Permute [0,1] -> [1,0] then back
		p1, err := tendo.NewPermute(1, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		p2, err := tendo.NewPermute(1, 0).Process(ctx, p1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !shapesEqual(p2.Shape(), a.Shape()) {
			t.Errorf("double permute should restore shape: expected %v, got %v",
				a.Shape(), p2.Shape())
		}
	})
}

func TestCat(t *testing.T) {
	ctx := context.Background()

	t.Run("Cat 1D", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)
		b := tendo.MustFromSlice([]float32{4, 5}, 2)

		result, err := tendo.NewCat(nil, []*tendo.Tensor{b}, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{5}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4, 5}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Cat 2D along dim 0", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		b := tendo.MustFromSlice([]float32{5, 6}, 1, 2)

		result, err := tendo.NewCat(nil, []*tendo.Tensor{b}, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4, 5, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Cat 2D along dim 1", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		b := tendo.MustFromSlice([]float32{5, 6}, 2, 1)

		result, err := tendo.NewCat(nil, []*tendo.Tensor{b}, 1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 2, 5, 3, 4, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Cat multiple tensors", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2}, 2)
		b := tendo.MustFromSlice([]float32{3}, 1)
		c := tendo.MustFromSlice([]float32{4, 5, 6}, 3)

		result, err := tendo.NewCat(nil, []*tendo.Tensor{b, c}, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{6}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4, 5, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Cat shape mismatch error", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		b := tendo.MustFromSlice([]float32{5, 6, 7}, 1, 3) // wrong cols

		_, err := tendo.NewCat(nil, []*tendo.Tensor{b}, 0).Process(ctx, a)
		if err == nil {
			t.Error("expected error for shape mismatch")
		}
	})
}

func TestStack(t *testing.T) {
	ctx := context.Background()

	t.Run("Stack 1D tensors at dim 0", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)
		b := tendo.MustFromSlice([]float32{4, 5, 6}, 3)

		result, err := tendo.NewStack(nil, []*tendo.Tensor{b}, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4, 5, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Stack 1D tensors at dim 1", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)
		b := tendo.MustFromSlice([]float32{4, 5, 6}, 3)

		result, err := tendo.NewStack(nil, []*tendo.Tensor{b}, 1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 4, 2, 5, 3, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Stack 2D tensors at dim 0", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		b := tendo.MustFromSlice([]float32{5, 6, 7, 8}, 2, 2)

		result, err := tendo.NewStack(nil, []*tendo.Tensor{b}, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Stack multiple tensors", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2}, 2)
		b := tendo.MustFromSlice([]float32{3, 4}, 2)
		c := tendo.MustFromSlice([]float32{5, 6}, 2)

		result, err := tendo.NewStack(nil, []*tendo.Tensor{b, c}, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4, 5, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Stack shape mismatch error", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)
		b := tendo.MustFromSlice([]float32{4, 5}, 2) // different size

		_, err := tendo.NewStack(nil, []*tendo.Tensor{b}, 0).Process(ctx, a)
		if err == nil {
			t.Error("expected error for shape mismatch")
		}
	})
}
