package tendo_test

import (
	"context"
	"math"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/pkg/cpu"
)

func TestReLU(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("ReLU", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{-2, -1, 0, 1, 2}, 5)

		result, err := tendo.NewReLU(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{0, 0, 0, 1, 2}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestSigmoid(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Sigmoid", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{0}, 1)

		result, err := tendo.NewSigmoid(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		if math.Abs(float64(data[0]-0.5)) > 1e-6 {
			t.Errorf("sigmoid(0) should be 0.5, got %v", data[0])
		}
	})

	t.Run("Sigmoid range", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{-10, 0, 10}, 3)

		result, err := tendo.NewSigmoid(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		// sigmoid(-10) ≈ 0, sigmoid(0) = 0.5, sigmoid(10) ≈ 1
		if data[0] > 0.001 {
			t.Errorf("sigmoid(-10) should be near 0, got %v", data[0])
		}
		if math.Abs(float64(data[1]-0.5)) > 1e-6 {
			t.Errorf("sigmoid(0) should be 0.5, got %v", data[1])
		}
		if data[2] < 0.999 {
			t.Errorf("sigmoid(10) should be near 1, got %v", data[2])
		}
	})
}

func TestTanh(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Tanh", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{0}, 1)

		result, err := tendo.NewTanh(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		if math.Abs(float64(data[0])) > 1e-6 {
			t.Errorf("tanh(0) should be 0, got %v", data[0])
		}
	})

	t.Run("Tanh range", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{-10, 0, 10}, 3)

		result, err := tendo.NewTanh(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		// tanh(-10) ≈ -1, tanh(0) = 0, tanh(10) ≈ 1
		if data[0] > -0.999 {
			t.Errorf("tanh(-10) should be near -1, got %v", data[0])
		}
		if math.Abs(float64(data[1])) > 1e-6 {
			t.Errorf("tanh(0) should be 0, got %v", data[1])
		}
		if data[2] < 0.999 {
			t.Errorf("tanh(10) should be near 1, got %v", data[2])
		}
	})
}

func TestSoftmax(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("Softmax 1D", func(t *testing.T) {
		// softmax([1, 2, 3]) = [e^1, e^2, e^3] / sum
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)

		result, err := tendo.NewSoftmax(backend, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()

		// Check that values sum to 1
		sum := float32(0)
		for _, v := range data {
			sum += v
		}
		if math.Abs(float64(sum-1)) > 1e-5 {
			t.Errorf("softmax should sum to 1, got %v", sum)
		}

		// Check ordering: softmax preserves order
		if data[0] >= data[1] || data[1] >= data[2] {
			t.Errorf("softmax should preserve order: got %v", data)
		}
	})

	t.Run("Softmax 2D dim=-1", func(t *testing.T) {
		// [[1, 2, 3],
		//  [1, 1, 1]]
		a := tendo.MustFromSlice([]float32{1, 2, 3, 1, 1, 1}, 2, 3)

		result, err := tendo.NewSoftmax(backend, -1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()

		// Check row sums
		row0Sum := data[0] + data[1] + data[2]
		row1Sum := data[3] + data[4] + data[5]
		if math.Abs(float64(row0Sum-1)) > 1e-5 {
			t.Errorf("row 0 should sum to 1, got %v", row0Sum)
		}
		if math.Abs(float64(row1Sum-1)) > 1e-5 {
			t.Errorf("row 1 should sum to 1, got %v", row1Sum)
		}

		// Row 1 should be uniform (1/3 each)
		for i := 3; i < 6; i++ {
			if math.Abs(float64(data[i]-1.0/3.0)) > 1e-5 {
				t.Errorf("uniform input should give uniform softmax, got %v", data[i])
			}
		}
	})

	t.Run("Softmax numerical stability", func(t *testing.T) {
		// Large values should not overflow
		a := tendo.MustFromSlice([]float32{1000, 1001, 1002}, 3)

		result, err := tendo.NewSoftmax(backend, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()

		// Should not have NaN or Inf
		for i, v := range data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("softmax produced NaN/Inf at index %d", i)
			}
		}

		// Should still sum to 1
		sum := data[0] + data[1] + data[2]
		if math.Abs(float64(sum-1)) > 1e-5 {
			t.Errorf("softmax should sum to 1, got %v", sum)
		}
	})

	t.Run("Softmax 3D", func(t *testing.T) {
		// Shape [2, 2, 3], softmax along dim 2
		data := make([]float32, 12)
		for i := range data {
			data[i] = float32(i)
		}
		a := tendo.MustFromSlice(data, 2, 2, 3)

		result, err := tendo.NewSoftmax(backend, 2).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		resultData := result.MustData()

		// Check each slice along dim 2 sums to 1
		for batch := 0; batch < 2; batch++ {
			for row := 0; row < 2; row++ {
				sum := float32(0)
				for col := 0; col < 3; col++ {
					idx := batch*6 + row*3 + col
					sum += resultData[idx]
				}
				if math.Abs(float64(sum-1)) > 1e-5 {
					t.Errorf("batch %d, row %d should sum to 1, got %v", batch, row, sum)
				}
			}
		}
	})
}

func TestLogSoftmax(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("LogSoftmax", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)

		result, err := tendo.NewLogSoftmax(backend, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()

		// All values should be negative (log of probability < 1)
		for i, v := range data {
			if v > 0 {
				t.Errorf("log softmax should be <= 0, got %v at index %d", v, i)
			}
		}

		// exp(logsoftmax) should sum to 1
		sum := float32(0)
		for _, v := range data {
			sum += float32(math.Exp(float64(v)))
		}
		if math.Abs(float64(sum-1)) > 1e-5 {
			t.Errorf("exp(logsoftmax) should sum to 1, got %v", sum)
		}
	})
}

func TestGELU(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("GELU", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{-2, -1, 0, 1, 2}, 5)

		result, err := tendo.NewGELU(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()

		// GELU(0) = 0
		if math.Abs(float64(data[2])) > 1e-6 {
			t.Errorf("GELU(0) should be 0, got %v", data[2])
		}

		// GELU is approximately x for large positive x
		if math.Abs(float64(data[4]-2)) > 0.1 {
			t.Errorf("GELU(2) should be near 2, got %v", data[4])
		}

		// GELU is near 0 for large negative x
		if data[0] > 0.1 || data[0] < -0.1 {
			t.Errorf("GELU(-2) should be near 0, got %v", data[0])
		}
	})
}

func TestLeakyReLU(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("LeakyReLU", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{-2, -1, 0, 1, 2}, 5)

		result, err := tendo.NewLeakyReLU(backend, 0.1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{-0.2, -0.1, 0, 1, 2}
		for i, v := range expected {
			if math.Abs(float64(data[i]-v)) > 1e-6 {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestSiLU(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("SiLU", func(t *testing.T) {
		a := tendo.MustFromSlice([]float32{0}, 1)

		result, err := tendo.NewSiLU(backend).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		// SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
		if math.Abs(float64(data[0])) > 1e-6 {
			t.Errorf("SiLU(0) should be 0, got %v", data[0])
		}
	})
}

func TestDropout(t *testing.T) {
	backend := cpu.NewBackend()

	t.Run("Dropout inference mode (no-op)", func(t *testing.T) {
		ctx := context.Background() // default is inference mode
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5}, 5)

		result, err := tendo.NewDropout(backend, 0.5).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// In inference mode, should return same tensor
		if result != a {
			t.Error("dropout in inference mode should return same tensor")
		}
	})

	t.Run("Dropout training mode applies mask", func(t *testing.T) {
		ctx := tendo.WithTraining(context.Background())
		a := tendo.MustOnes(1000) // large tensor for statistical test

		result, err := tendo.NewDropout(backend, 0.5).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should be different tensor
		if result == a {
			t.Error("dropout in training mode should return new tensor")
		}

		// Count zeros and non-zeros
		data := result.MustData()
		zeros := 0
		nonZeros := 0
		for _, v := range data {
			if v == 0 {
				zeros++
			} else {
				nonZeros++
			}
		}

		// With p=0.5, expect roughly 50% zeros (allow 40-60% range)
		zeroRatio := float64(zeros) / float64(len(data))
		if zeroRatio < 0.35 || zeroRatio > 0.65 {
			t.Errorf("expected ~50%% zeros, got %.1f%%", zeroRatio*100)
		}

		// Non-zero values should be scaled by 1/(1-p) = 2
		for _, v := range data {
			if v != 0 && math.Abs(float64(v-2.0)) > 1e-5 {
				t.Errorf("non-zero values should be scaled to 2, got %v", v)
				break
			}
		}
	})

	t.Run("Dropout p=0 keeps all", func(t *testing.T) {
		ctx := tendo.WithTraining(context.Background())
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5}, 5)

		result, err := tendo.NewDropout(backend, 0).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		expected := []float32{1, 2, 3, 4, 5}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})

	t.Run("Dropout p=1 drops all", func(t *testing.T) {
		ctx := tendo.WithTraining(context.Background())
		a := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5}, 5)

		result, err := tendo.NewDropout(backend, 1).Process(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := result.MustData()
		for i, v := range data {
			if v != 0 {
				t.Errorf("at index %d: expected 0, got %v", i, v)
			}
		}
	})
}
