package tendo_test

import (
	"context"
	"math"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/pkg/cpu"
	"github.com/zoobzio/tendo/pkg/cuda"
)

func TestMatMul(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("2x3 @ 3x2 = 2x2", func(t *testing.T) {
		// A = [[1, 2, 3],
		//      [4, 5, 6]]
		// B = [[7, 8],
		//      [9, 10],
		//      [11, 12]]
		// C = A @ B = [[1*7+2*9+3*11, 1*8+2*10+3*12],
		//              [4*7+5*9+6*11, 4*8+5*10+6*12]]
		//           = [[58, 64],
		//              [139, 154]]
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := tendo.MustFromSlice([]float32{7, 8, 9, 10, 11, 12}, 3, 2)

		result, err := tendo.NewMatMul(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{58, 64, 139, 154}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("1x3 @ 3x1 = 1x1 (dot product)", func(t *testing.T) {
		// [1, 2, 3] @ [4, 5, 6]^T = 1*4 + 2*5 + 3*6 = 32
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 1, 3)
		b := tendo.MustFromSlice([]float32{4, 5, 6}, 3, 1)

		result, err := tendo.NewMatMul(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		if data[0] != 32 {
			t.Errorf("expected 32, got %v", data[0])
		}
	})

	t.Run("Identity matrix", func(t *testing.T) {
		// A @ I = A
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
		identity := tendo.MustEye(3)

		result, err := tendo.NewMatMul(backend, identity).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Batched matmul", func(t *testing.T) {
		// Shape [2, 2, 3] @ [2, 3, 2] = [2, 2, 2]
		// Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
		// Batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]] = [[184,202],[247,274]]
		a := tendo.MustFromSlice([]float32{
			1, 2, 3, 4, 5, 6, // batch 0
			7, 8, 9, 10, 11, 12, // batch 1
		}, 2, 2, 3)

		b := tendo.MustFromSlice([]float32{
			1, 2, 3, 4, 5, 6, // batch 0
			7, 8, 9, 10, 11, 12, // batch 1
		}, 2, 3, 2)

		result, err := tendo.NewMatMul(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		// Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
		// Batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]] = [[220,244],[301,334]]
		expected := []float32{22, 28, 49, 64, 220, 244, 301, 334}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Shape mismatch error", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)
		b := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 3, 2) // incompatible: 2 != 3

		_, err := tendo.NewMatMul(backend, b).Process(ctx, a)
		if err == nil {
			t.Fatal("expected shape mismatch error")
		}
	})
}

func TestTranspose(t *testing.T) {
	ctx := context.Background()

	t.Run("Transpose 2D", func(t *testing.T) {
		// [[1, 2, 3],
		//  [4, 5, 6]] -> [[1, 4],
		//                 [2, 5],
		//                 [3, 6]]
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewTranspose(0, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Verify values at specific positions
		// Original [0,0]=1, [0,1]=2, [0,2]=3, [1,0]=4, [1,1]=5, [1,2]=6
		// Transposed [0,0]=1, [1,0]=2, [2,0]=3, [0,1]=4, [1,1]=5, [2,1]=6

		// Make contiguous to get values in row-major order
		cont := result.Contiguous()
		data := cont.MustData()
		expected := []float32{1, 4, 2, 5, 3, 6}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Transpose is view (shares storage)", func(t *testing.T) {
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewTranspose(0, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Transposed tensor should share storage
		if result.Storage() != tensor.Storage() {
			t.Error("transpose should share storage with original")
		}
	})

	t.Run("Double transpose is identity", func(t *testing.T) {
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewTranspose(0, 1).Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		result, err = tendo.NewTranspose(0, 1).Process(ctx, result)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !shapesEqual(result.Shape(), tensor.Shape()) {
			t.Errorf("double transpose shape mismatch: expected %v, got %v",
				tensor.Shape(), result.Shape())
		}
	})
}

func TestT(t *testing.T) {
	ctx := context.Background()

	t.Run("T transposes last two dims", func(t *testing.T) {
		// Shape [2, 3] -> [3, 2]
		tensor := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		result, err := tendo.NewT().Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("T on 3D tensor", func(t *testing.T) {
		// Shape [2, 3, 4] -> [2, 4, 3]
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i)
		}
		tensor := tendo.MustFromSlice(data, 2, 3, 4)
		result, err := tendo.NewT().Process(ctx, tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 4, 3}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})
}

func TestBatchedMatMulBroadcast(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Broadcast batch dim: [2,2,3] @ [3,2] = [2,2,2]", func(t *testing.T) {
		// A has batch dim 2, B has no batch dim
		a := tendo.MustFromSlice([]float32{
			1, 2, 3, 4, 5, 6, // batch 0: [[1,2,3],[4,5,6]]
			7, 8, 9, 10, 11, 12, // batch 1: [[7,8,9],[10,11,12]]
		}, 2, 2, 3)

		b := tendo.MustFromSlice([]float32{
			1, 2, 3, 4, 5, 6, // [[1,2],[3,4],[5,6]]
		}, 3, 2)

		result, err := tendo.NewMatMul(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		// Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
		// Batch 1: [[7,8,9],[10,11,12]] @ [[1,2],[3,4],[5,6]] = [[76,100],[103,136]]
		expected := []float32{22, 28, 49, 64, 76, 100, 103, 136}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("4D batched matmul: [2,3,2,3] @ [2,3,3,2] = [2,3,2,2]", func(t *testing.T) {
		// Create test data
		aData := make([]float32, 2*3*2*3)
		bData := make([]float32, 2*3*3*2)
		for i := range aData {
			aData[i] = float32(i + 1)
		}
		for i := range bData {
			bData[i] = float32(i + 1)
		}

		a := tendo.MustFromSlice(aData, 2, 3, 2, 3)
		b := tendo.MustFromSlice(bData, 2, 3, 3, 2)

		result, err := tendo.NewMatMul(backend, b).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Just verify shape and no errors for 4D case
		if result.Numel() != 2*3*2*2 {
			t.Errorf("expected %d elements, got %d", 2*3*2*2, result.Numel())
		}
	})
}

func TestGPUMatMul(t *testing.T) {
	if !cuda.IsCUDAAvailable() {
		t.Skip("CUDA not available")
	}

	ctx := context.Background()
	gpuBackend := cuda.NewBackend()

	t.Run("GPU 2D matmul", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := tendo.MustFromSlice([]float32{7, 8, 9, 10, 11, 12}, 3, 2)

		// Move to GPU
		aGPU, err := tendo.ToGPU(0).Process(ctx, a)
		if err != nil {
			t.Fatalf("failed to move a to GPU: %v", err)
		}
		bGPU, err := tendo.ToGPU(0).Process(ctx, b)
		if err != nil {
			t.Fatalf("failed to move b to GPU: %v", err)
		}

		// Perform matmul on GPU
		resultGPU, err := tendo.NewMatMul(gpuBackend, bGPU).Process(ctx, aGPU)
		if err != nil {
			t.Fatalf("GPU matmul error: %v", err)
		}

		// Move back to CPU for verification
		result, err := tendo.ToCPU().Process(ctx, resultGPU)
		if err != nil {
			t.Fatalf("failed to move result to CPU: %v", err)
		}

		expectedShape := []int{2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{58, 64, 139, 154}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("GPU batched matmul", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{
			1, 2, 3, 4, 5, 6,
			7, 8, 9, 10, 11, 12,
		}, 2, 2, 3)

		b := tendo.MustFromSlice([]float32{
			1, 2, 3, 4, 5, 6,
			7, 8, 9, 10, 11, 12,
		}, 2, 3, 2)

		// Move to GPU
		aGPU, err := tendo.ToGPU(0).Process(ctx, a)
		if err != nil {
			t.Fatalf("failed to move a to GPU: %v", err)
		}
		bGPU, err := tendo.ToGPU(0).Process(ctx, b)
		if err != nil {
			t.Fatalf("failed to move b to GPU: %v", err)
		}

		// Perform batched matmul on GPU
		resultGPU, err := tendo.NewMatMul(gpuBackend, bGPU).Process(ctx, aGPU)
		if err != nil {
			t.Fatalf("GPU batched matmul error: %v", err)
		}

		// Move back to CPU for verification
		result, err := tendo.ToCPU().Process(ctx, resultGPU)
		if err != nil {
			t.Fatalf("failed to move result to CPU: %v", err)
		}

		expectedShape := []int{2, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		data := result.MustData()
		expected := []float32{22, 28, 49, 64, 220, 244, 301, 334}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestMatMulTranspose(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("A @ A^T is symmetric", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)

		// Transpose A
		aT, err := tendo.NewT().Process(ctx, a)
		if err != nil {
			t.Fatalf("transpose error: %v", err)
		}

		// Make contiguous before matmul (transpose creates non-contiguous view)
		aTCont := aT.Contiguous()

		// A @ A^T
		result, err := tendo.NewMatMul(backend, aTCont).Process(ctx, a)
		if err != nil {
			t.Fatalf("matmul error: %v", err)
		}

		// Result should be symmetric: result[0,1] == result[1,0]
		data := result.MustData()
		// [[1,2],[3,4]] @ [[1,3],[2,4]] = [[1+4, 3+8],[3+8, 9+16]] = [[5,11],[11,25]]
		expected := []float32{5, 11, 11, 25}
		for i, v := range expected {
			if math.Abs(float64(data[i]-v)) > 1e-6 {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}
