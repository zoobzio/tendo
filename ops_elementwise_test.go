package tendo_test

import (
	"context"
	"math"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/pkg/cpu"
)

func TestAdd(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Add same shape", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		b := tendo.MustFromSlice([]float32{5, 6, 7, 8}, 2, 2)

		result, err := tendo.NewAdd(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{6, 8, 10, 12}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Add with broadcasting scalar", func(t *testing.T) {
		// [2, 3] + [1] (scalar broadcast)
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := tendo.MustFromSlice([]float32{10}, 1)

		result, err := tendo.NewAdd(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{11, 12, 13, 14, 15, 16}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Add with broadcasting row", func(t *testing.T) {
		// [2, 3] + [1, 3] (broadcast along dim 0)
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := tendo.MustFromSlice([]float32{10, 20, 30}, 1, 3)

		result, err := tendo.NewAdd(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{11, 22, 33, 14, 25, 36}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Add with broadcasting column", func(t *testing.T) {
		// [2, 3] + [2, 1] (broadcast along dim 1)
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := tendo.MustFromSlice([]float32{10, 20}, 2, 1)

		result, err := tendo.NewAdd(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{11, 12, 13, 24, 25, 26}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestMul(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Mul same shape", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		b := tendo.MustFromSlice([]float32{2, 3, 4, 5}, 2, 2)

		result, err := tendo.NewMul(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{2, 6, 12, 20}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Mul with broadcasting", func(t *testing.T) {
		// [2, 3] * [3] (broadcast)
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := tendo.MustFromSlice([]float32{2, 3, 4}, 3)

		result, err := tendo.NewMul(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{2, 6, 12, 8, 15, 24}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestSub(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Sub same shape", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{5, 6, 7, 8}, 2, 2)
		b := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)

		result, err := tendo.NewSub(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{4, 4, 4, 4}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestDiv(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Div same shape", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{10, 20, 30, 40}, 2, 2)
		b := tendo.MustFromSlice([]float32{2, 4, 5, 8}, 2, 2)

		result, err := tendo.NewDiv(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{5, 5, 6, 5}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestNeg(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Neg", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, -2, 3, -4}, 2, 2)

		result, err := tendo.NewNeg(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{-1, 2, -3, 4}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestExp(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Exp", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{0, 1, 2}, 3)

		result, err := tendo.NewExp(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, float32(math.E), float32(math.E * math.E)}
		for i, v := range expected {
			if math.Abs(float64(data[i]-v)) > 1e-5 {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestLog(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Log", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, float32(math.E), float32(math.E * math.E)}, 3)

		result, err := tendo.NewLog(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{0, 1, 2}
		for i, v := range expected {
			if math.Abs(float64(data[i]-v)) > 1e-5 {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestSqrt(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Sqrt", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 4, 9, 16}, 4)

		result, err := tendo.NewSqrt(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestSquare(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Square", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 4)

		result, err := tendo.NewSquare(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 4, 9, 16}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestPow(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Pow", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 4)

		result, err := tendo.NewPow(backend, 3).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 8, 27, 64}
		for i, v := range expected {
			if math.Abs(float64(data[i]-v)) > 1e-5 {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestAbs(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Abs", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{-1, 2, -3, 4}, 4)

		result, err := tendo.NewAbs(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestBroadcasting3D(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("3D broadcasting", func(t *testing.T) {
		// [2, 3, 4] + [1, 3, 1] should broadcast
		a := tendo.MustOnes(2, 3, 4)
		b := tendo.MustFromSlice([]float32{1, 2, 3}, 1, 3, 1)

		result, err := tendo.NewAdd(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3, 4}
		if !shapesEqualTest(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Row 0 should all be 2 (1+1), row 1 should all be 3 (1+2), row 2 should all be 4 (1+3)
		data := result.MustData()
		for i := 0; i < 2; i++ { // batch
			for j := 0; j < 3; j++ { // row
				for k := 0; k < 4; k++ { // col
					idx := i*12 + j*4 + k
					expected := float32(j + 2)
					if data[idx] != expected {
						t.Errorf("at [%d,%d,%d]: expected %v, got %v", i, j, k, expected, data[idx])
					}
				}
			}
		}
	})
}

func TestNonContiguousUnary(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Neg on transposed tensor", func(t *testing.T) {
		// Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
		// Storage order: [1, 2, 3, 4, 5, 6]
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

		// Transpose to 3x2: [[1, 4], [2, 5], [3, 6]]
		// Storage still: [1, 2, 3, 4, 5, 6] but strides are [1, 3]
		transposed, err := tendo.NewPermute(1, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error during transpose: %v", err)
		}

		// Verify it's non-contiguous
		if transposed.IsContiguous() {
			t.Fatal("expected transposed tensor to be non-contiguous")
		}

		// Apply Neg: should produce [[-1, -4], [-2, -5], [-3, -6]]
		result, err := tendo.NewNeg(backend).Process(ctx, transposed)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Result should be contiguous with correct values
		expectedShape := []int{3, 2}
		if !shapesEqualTest(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		// Expected in row-major order: [-1, -4, -2, -5, -3, -6]
		expected := []float32{-1, -4, -2, -5, -3, -6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Square on sliced tensor", func(t *testing.T) {
		// Create a 4x3 tensor
		a := tendo.MustFromSlice([]float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		}, 4, 3)

		// Slice rows 1-3 (indices 1, 2): [[4, 5, 6], [7, 8, 9]]
		sliced, err := tendo.NewSlice(0, 1, 3).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error during slice: %v", err)
		}

		// Verify offset is set
		if sliced.Offset() != 3 {
			t.Errorf("expected offset 3, got %d", sliced.Offset())
		}

		// Apply Square
		result, err := tendo.NewSquare(backend).Process(ctx, sliced)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{16, 25, 36, 49, 64, 81}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestNonContiguousBinary(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Add with transposed tensor", func(t *testing.T) {
		// a: 2x3 [[1, 2, 3], [4, 5, 6]]
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

		// b: 3x2 [[10, 20], [30, 40], [50, 60]]
		b := tendo.MustFromSlice([]float32{10, 20, 30, 40, 50, 60}, 3, 2)

		// Transpose a to 3x2: [[1, 4], [2, 5], [3, 6]]
		aT, err := tendo.NewPermute(1, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error during transpose: %v", err)
		}

		// aT + b: [[11, 24], [32, 45], [53, 66]]
		result, err := tendo.NewAdd(backend, b).Process(ctx, aT)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{11, 24, 32, 45, 53, 66}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Mul with both tensors transposed", func(t *testing.T) {
		// a: 2x3, transposed to 3x2
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		aT, err := tendo.NewPermute(1, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// b: 2x3, transposed to 3x2
		b := tendo.MustFromSlice([]float32{2, 3, 4, 5, 6, 7}, 2, 3)
		bT, err := tendo.NewPermute(1, 0).Process(ctx, b)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// aT * bT: [[1*2, 4*5], [2*3, 5*6], [3*4, 6*7]] = [[2, 20], [6, 30], [12, 42]]
		result, err := tendo.NewMul(backend, bT).Process(ctx, aT)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{2, 20, 6, 30, 12, 42}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Add with sliced tensors", func(t *testing.T) {
		// a: slice of larger tensor
		full := tendo.MustFromSlice([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8}, 3, 3)
		a, err := tendo.NewSlice(0, 1, 3).Process(ctx, full) // rows 1-2: [[3, 4, 5], [6, 7, 8]]
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// b: another tensor
		b := tendo.MustFromSlice([]float32{10, 20, 30, 40, 50, 60}, 2, 3)

		result, err := tendo.NewAdd(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{13, 24, 35, 46, 57, 68}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Add transposed with broadcasting", func(t *testing.T) {
		// a: 2x3 transposed to 3x2
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		aT, err := tendo.NewPermute(1, 0).Process(ctx, a) // 3x2: [[1, 4], [2, 5], [3, 6]]
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// b: [2] broadcasts to 3x2
		b := tendo.MustFromSlice([]float32{10, 100}, 2)

		// Result: [[11, 104], [12, 105], [13, 106]]
		result, err := tendo.NewAdd(backend, b).Process(ctx, aT)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{11, 104, 12, 105, 13, 106}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestClamp(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Clamp basic", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{-5, -1, 0, 1, 5, 10}, 6)

		result, err := tendo.NewClamp(backend, -2, 3).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{-2, -1, 0, 1, 3, 3}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestSign(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Sign", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{-5, -0.1, 0, 0.1, 5}, 5)

		result, err := tendo.NewSign(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{-1, -1, 0, 1, 1}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestWhere(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Where basic", func(t *testing.T) {
		// condition > 0: select from input, else from other
		input := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 4)
		condition := tendo.MustFromSlice([]float32{1, 0, 1, 0}, 4)
		other := tendo.MustFromSlice([]float32{10, 20, 30, 40}, 4)

		result, err := tendo.NewWhere(backend, condition, other).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 20, 3, 40}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Where with broadcasting", func(t *testing.T) {
		// input: [2, 2]
		input := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		// condition: [2] - broadcasts to [2, 2]
		condition := tendo.MustFromSlice([]float32{1, 0}, 2)
		// other: [2, 2]
		other := tendo.MustFromSlice([]float32{10, 20, 30, 40}, 2, 2)

		result, err := tendo.NewWhere(backend, condition, other).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// condition broadcasts: [[1, 0], [1, 0]]
		// Result: [[1, 20], [3, 40]]
		data := result.MustData()
		expected := []float32{1, 20, 3, 40}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

// shapesEqualTest compares two shapes for equality.
func shapesEqualTest(a, b []int) bool {
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
