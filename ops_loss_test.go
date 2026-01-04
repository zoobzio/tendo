package tendo_test

import (
	"context"
	"math"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cpu"
)

func TestMSELoss(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("MSELoss mean reduction", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 4)
		target := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 4)

		result, err := tendo.NewMSELoss(backend, target, tendo.ReductionMean).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Perfect match should give 0 loss
		data := result.MustData()
		if data[0] != 0 {
			t.Errorf("expected 0, got %v", data[0])
		}
	})

	t.Run("MSELoss with difference", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{0, 0, 0, 0}, 4)
		target := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 4)

		result, err := tendo.NewMSELoss(backend, target, tendo.ReductionMean).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// MSE = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
		data := result.MustData()
		if data[0] != 7.5 {
			t.Errorf("expected 7.5, got %v", data[0])
		}
	})

	t.Run("MSELoss sum reduction", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{0, 0}, 2)
		target := tendo.MustFromSlice([]float32{1, 2}, 2)

		result, err := tendo.NewMSELoss(backend, target, tendo.ReductionSum).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Sum = 1 + 4 = 5
		data := result.MustData()
		if data[0] != 5 {
			t.Errorf("expected 5, got %v", data[0])
		}
	})

	t.Run("MSELoss none reduction", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{0, 0}, 2)
		target := tendo.MustFromSlice([]float32{1, 2}, 2)

		result, err := tendo.NewMSELoss(backend, target, tendo.ReductionNone).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Element-wise: [1, 4]
		data := result.MustData()
		expected := []float32{1, 4}
		for i, v := range expected {
			if data[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, data[i])
			}
		}
	})
}

func TestL1Loss(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("L1Loss mean reduction", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{0, 0, 0, 0}, 4)
		target := tendo.MustFromSlice([]float32{1, -2, 3, -4}, 4)

		result, err := tendo.NewL1Loss(backend, target, tendo.ReductionMean).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// L1 = (1 + 2 + 3 + 4) / 4 = 2.5
		data := result.MustData()
		if data[0] != 2.5 {
			t.Errorf("expected 2.5, got %v", data[0])
		}
	})

	t.Run("L1Loss sum reduction", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{0, 0}, 2)
		target := tendo.MustFromSlice([]float32{1, -2}, 2)

		result, err := tendo.NewL1Loss(backend, target, tendo.ReductionSum).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Sum = 1 + 2 = 3
		data := result.MustData()
		if data[0] != 3 {
			t.Errorf("expected 3, got %v", data[0])
		}
	})
}

func TestCrossEntropyLoss(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("CrossEntropyLoss basic", func(t *testing.T) {
		// 2 samples, 3 classes
		// Sample 0: logits [2, 1, 0.1], target class 0
		// Sample 1: logits [0, 2, 1], target class 1
		input := tendo.MustFromSlice([]float32{
			2, 1, 0.1,
			0, 2, 1,
		}, 2, 3)
		target := tendo.MustFromSlice([]float32{0, 1}, 2)

		result, err := tendo.NewCrossEntropyLoss(backend, target, tendo.ReductionMean).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Loss should be positive and reasonable
		data := result.MustData()
		if data[0] <= 0 || data[0] > 10 {
			t.Errorf("expected loss in reasonable range, got %v", data[0])
		}
	})

	t.Run("CrossEntropyLoss confident prediction", func(t *testing.T) {
		// Very confident prediction: softmax will be nearly 1 for correct class
		input := tendo.MustFromSlice([]float32{100, 0, 0}, 1, 3)
		target := tendo.MustFromSlice([]float32{0}, 1)

		result, err := tendo.NewCrossEntropyLoss(backend, target, tendo.ReductionMean).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Loss should be very small (near 0)
		data := result.MustData()
		if data[0] > 0.01 {
			t.Errorf("expected very small loss, got %v", data[0])
		}
	})

	t.Run("CrossEntropyLoss wrong prediction", func(t *testing.T) {
		// Very wrong prediction
		input := tendo.MustFromSlice([]float32{0, 0, 100}, 1, 3)
		target := tendo.MustFromSlice([]float32{0}, 1) // but we want class 0

		result, err := tendo.NewCrossEntropyLoss(backend, target, tendo.ReductionMean).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Loss should be large (around 100)
		data := result.MustData()
		if data[0] < 90 {
			t.Errorf("expected large loss, got %v", data[0])
		}
	})
}

func TestNLLLoss(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("NLLLoss basic", func(t *testing.T) {
		// Log probabilities (already log-softmaxed)
		// For testing, use log([0.5, 0.3, 0.2])
		input := tendo.MustFromSlice([]float32{
			float32(math.Log(0.5)), float32(math.Log(0.3)), float32(math.Log(0.2)),
		}, 1, 3)
		target := tendo.MustFromSlice([]float32{0}, 1)

		result, err := tendo.NewNLLLoss(backend, target, tendo.ReductionMean).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Loss = -log(0.5) ≈ 0.693
		data := result.MustData()
		expected := float32(-math.Log(0.5))
		if math.Abs(float64(data[0]-expected)) > 0.001 {
			t.Errorf("expected %v, got %v", expected, data[0])
		}
	})

	t.Run("NLLLoss multiple samples", func(t *testing.T) {
		// 2 samples
		input := tendo.MustFromSlice([]float32{
			float32(math.Log(0.8)), float32(math.Log(0.1)), float32(math.Log(0.1)),
			float32(math.Log(0.1)), float32(math.Log(0.8)), float32(math.Log(0.1)),
		}, 2, 3)
		target := tendo.MustFromSlice([]float32{0, 1}, 2)

		result, err := tendo.NewNLLLoss(backend, target, tendo.ReductionMean).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Loss = (-log(0.8) + -log(0.8)) / 2 ≈ 0.223
		data := result.MustData()
		expected := float32(-math.Log(0.8))
		if math.Abs(float64(data[0]-expected)) > 0.001 {
			t.Errorf("expected %v, got %v", expected, data[0])
		}
	})
}
