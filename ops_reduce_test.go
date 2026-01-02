package tendo_test

import (
	"context"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/pkg/cpu"
)

func TestSum(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Sum all elements", func(t *testing.T) {
		// [1, 2, 3, 4, 5, 6] -> 21
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)
		result, err := tendo.NewSum(backend, false).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Numel() != 1 {
			t.Errorf("expected scalar, got shape %v", result.Shape())
		}

		data := result.MustData()
		if data[0] != 21 {
			t.Errorf("expected 21, got %v", data[0])
		}
	})

	t.Run("Sum 2D along dim 0", func(t *testing.T) {
		// [[1, 2, 3],
		//  [4, 5, 6]] -> [5, 7, 9]
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewSum(backend, false, 0).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{5, 7, 9}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Sum 2D along dim 1", func(t *testing.T) {
		// [[1, 2, 3],
		//  [4, 5, 6]] -> [6, 15]
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewSum(backend, false, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{6, 15}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Sum 3D along dim 1", func(t *testing.T) {
		// Shape [2, 3, 4], sum along dim 1 -> [2, 4]
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i)
		}
		tensor := tendo.MustFromSlice(data, 2, 3, 4)
		result, err := tendo.NewSum(backend, false, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 4}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// First batch: sum of [0,1,2,3], [4,5,6,7], [8,9,10,11] along dim 1
		// = [0+4+8, 1+5+9, 2+6+10, 3+7+11] = [12, 15, 18, 21]
		// Second batch: sum of [12,13,14,15], [16,17,18,19], [20,21,22,23]
		// = [12+16+20, 13+17+21, 14+18+22, 15+19+23] = [48, 51, 54, 57]
		expected := []float32{12, 15, 18, 21, 48, 51, 54, 57}
		resultData := result.MustData()
		for i, v := range expected {
			if resultData[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, resultData[i])
			}
		}
	})

	t.Run("Sum with negative dimension", func(t *testing.T) {
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewSum(backend, false, -1).Process(ctx, tensor) // dim -1 = dim 1
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{6, 15}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestMean(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Mean all elements", func(t *testing.T) {
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)
		result, err := tendo.NewMean(backend, false).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := float32(3.5) // (1+2+3+4+5+6)/6
		if data[0] != expected {
			t.Errorf("expected %v, got %v", expected, data[0])
		}
	})

	t.Run("Mean 2D along dim 0", func(t *testing.T) {
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewMean(backend, false, 0).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{2.5, 3.5, 4.5} // (1+4)/2, (2+5)/2, (3+6)/2
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestMax(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Max along dim 0", func(t *testing.T) {
		// [[1, 5, 3],
		//  [4, 2, 6]] -> [4, 5, 6]
		tensor := tendo.MustFromSlice([]float32{1, 5, 3, 4, 2, 6}, 2, 3)
		result, err := tendo.NewMax(backend, false, 0).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{4, 5, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Max along dim 1", func(t *testing.T) {
		tensor := tendo.MustFromSlice([]float32{1, 5, 3, 4, 2, 6}, 2, 3)
		result, err := tendo.NewMax(backend, false, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{5, 6} // max of [1,5,3], max of [4,2,6]
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestMin(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Min along dim 1", func(t *testing.T) {
		tensor := tendo.MustFromSlice([]float32{1, 5, 3, 4, 2, 6}, 2, 3)
		result, err := tendo.NewMin(backend, false, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 2} // min of [1,5,3], min of [4,2,6]
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestArgMax(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("ArgMax along dim 0", func(t *testing.T) {
		// [[1, 5, 3],
		//  [4, 2, 6]]
		// ArgMax dim 0: [1, 0, 1] (indices of max in each column)
		tensor := tendo.MustFromSlice([]float32{1, 5, 3, 4, 2, 6}, 2, 3)
		result, err := tendo.NewArgMax(backend, 0, false).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 0, 1}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("ArgMax along dim 1", func(t *testing.T) {
		// [[1, 5, 3],
		//  [4, 2, 6]]
		// ArgMax dim 1: [1, 2] (indices of max in each row)
		tensor := tendo.MustFromSlice([]float32{1, 5, 3, 4, 2, 6}, 2, 3)
		result, err := tendo.NewArgMax(backend, 1, false).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{1, 2}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("ArgMax 3D along dim 1", func(t *testing.T) {
		// Shape [2, 3, 2]
		// Batch 0: [[0, 1], [2, 3], [4, 5]] -> argmax along dim 1 = [2, 2]
		// Batch 1: [[6, 7], [8, 9], [10, 11]] -> argmax along dim 1 = [2, 2]
		data := make([]float32, 12)
		for i := range data {
			data[i] = float32(i)
		}
		tensor := tendo.MustFromSlice(data, 2, 3, 2)
		result, err := tendo.NewArgMax(backend, 1, false).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		resultData := result.MustData()
		expected := []float32{2, 2, 2, 2}
		for i, v := range expected {
			if resultData[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, resultData[i])
			}
		}
	})
}

func TestArgMin(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("ArgMin along dim 1", func(t *testing.T) {
		// [[1, 5, 3],
		//  [4, 2, 6]]
		// ArgMin dim 1: [0, 1] (indices of min in each row)
		tensor := tendo.MustFromSlice([]float32{1, 5, 3, 4, 2, 6}, 2, 3)
		result, err := tendo.NewArgMin(backend, 1, false).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{0, 1}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestVar(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Var all elements", func(t *testing.T) {
		// [1, 2, 3, 4, 5]
		// mean = 3, var = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 4 = (4+1+0+1+4)/4 = 2.5
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5}, 5)
		result, err := tendo.NewVar(backend, false, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := float32(2.5) // Bessel's correction (N-1)
		if abs(data[0]-expected) > 0.001 {
			t.Errorf("expected %v, got %v", expected, data[0])
		}
	})

	t.Run("Var 2D along dim 0", func(t *testing.T) {
		// [[1, 2],
		//  [3, 4]]
		// Var along dim 0: [(1-2)^2+(3-2)^2]/(2-1), [(2-3)^2+(4-3)^2]/(2-1) = [2, 2]
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		result, err := tendo.NewVar(backend, false, 1, 0).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{2, 2}
		for i, v := range expected {
			if abs(data[i]-v) > 0.001 {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Var 2D along dim 1", func(t *testing.T) {
		// [[1, 2, 3],
		//  [4, 5, 6]]
		// Row 0: mean=2, var=[(1-2)^2+(2-2)^2+(3-2)^2]/2 = 2/2 = 1
		// Row 1: mean=5, var=[(4-5)^2+(5-5)^2+(6-5)^2]/2 = 2/2 = 1
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewVar(backend, false, 1, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 1}
		for i, v := range expected {
			if abs(data[i]-v) > 0.001 {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestStd(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Std all elements", func(t *testing.T) {
		// [1, 2, 3, 4, 5] -> var = 2.5 -> std = sqrt(2.5) ≈ 1.581
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5}, 5)
		result, err := tendo.NewStd(backend, false, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := float32(1.5811388) // sqrt(2.5)
		if abs(data[0]-expected) > 0.001 {
			t.Errorf("expected %v, got %v", expected, data[0])
		}
	})

	t.Run("Std 2D along dim 1", func(t *testing.T) {
		// [[1, 2, 3],
		//  [4, 5, 6]]
		// var = 1, std = 1
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewStd(backend, false, 1, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 1}
		for i, v := range expected {
			if abs(data[i]-v) > 0.001 {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestProd(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Prod all elements", func(t *testing.T) {
		// [1, 2, 3, 4] -> 24
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 4)
		result, err := tendo.NewProd(backend, false).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Numel() != 1 {
			t.Errorf("expected scalar, got shape %v", result.Shape())
		}

		data := result.MustData()
		if data[0] != 24 {
			t.Errorf("expected 24, got %v", data[0])
		}
	})

	t.Run("Prod 2D along dim 0", func(t *testing.T) {
		// [[1, 2, 3],
		//  [4, 5, 6]] -> [4, 10, 18]
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewProd(backend, false, 0).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{4, 10, 18}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Prod 2D along dim 1", func(t *testing.T) {
		// [[1, 2, 3],
		//  [4, 5, 6]] -> [6, 120]
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewProd(backend, false, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{6, 120}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Prod with zeros", func(t *testing.T) {
		// [1, 0, 3] -> 0
		tensor := tendo.MustFromSlice([]float32{1, 0, 3}, 3)
		result, err := tendo.NewProd(backend, false).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		if data[0] != 0 {
			t.Errorf("expected 0, got %v", data[0])
		}
	})
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
