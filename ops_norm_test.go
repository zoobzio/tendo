package tendo_test

import (
	"context"
	"math"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cpu"
)

func TestBatchNorm2d(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("BatchNorm2d inference mode", func(t *testing.T) {
		// Input: [2, 2, 2, 2] (batch=2, channels=2, height=2, width=2)
		data := make([]float32, 16)
		for i := range data {
			data[i] = float32(i)
		}
		input := tendo.MustFromSlice(data, 2, 2, 2, 2)

		// Simple parameters
		weight := tendo.MustFromSlice([]float32{1, 1}, 2)
		bias := tendo.MustFromSlice([]float32{0, 0}, 2)
		runningMean := tendo.MustFromSlice([]float32{0, 0}, 2)
		runningVar := tendo.MustFromSlice([]float32{1, 1}, 2)

		result, err := tendo.NewBatchNorm2d(backend, weight, bias, runningMean, runningVar, 1e-5, 0.1).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Shape should be unchanged
		expectedShape := []int{2, 2, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// With running_mean=0, running_var=1, weight=1, bias=0, output ≈ input
		resultData := result.MustData()
		for i, v := range data {
			if math.Abs(float64(resultData[i]-v)) > 1e-4 {
				t.Errorf("at index %d: expected ≈%v, got %v", i, v, resultData[i])
			}
		}
	})

	t.Run("BatchNorm2d training mode", func(t *testing.T) {
		// Input: [2, 1, 2, 2] (batch=2, channels=1, height=2, width=2)
		// Values for channel 0: [0,1,2,3, 4,5,6,7] across both batches
		data := []float32{0, 1, 2, 3, 4, 5, 6, 7}
		input := tendo.MustFromSlice(data, 2, 1, 2, 2)

		weight := tendo.MustFromSlice([]float32{1}, 1)
		bias := tendo.MustFromSlice([]float32{0}, 1)
		runningMean := tendo.MustFromSlice([]float32{0}, 1)
		runningVar := tendo.MustFromSlice([]float32{1}, 1)

		// Use training context
		trainCtx := tendo.WithTraining(ctx)

		result, err := tendo.NewBatchNorm2d(backend, weight, bias, runningMean, runningVar, 1e-5, 0.1).Process(trainCtx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// In training mode, output should be normalized to mean=0, var≈1
		resultData := result.MustData()

		// Compute mean of output
		sum := float32(0)
		for _, v := range resultData {
			sum += v
		}
		mean := sum / float32(len(resultData))

		// Mean should be close to 0
		if math.Abs(float64(mean)) > 1e-5 {
			t.Errorf("expected mean ≈ 0, got %v", mean)
		}

		// Compute variance of output
		sumSq := float32(0)
		for _, v := range resultData {
			sumSq += v * v
		}
		variance := sumSq / float32(len(resultData))

		// Variance should be close to 1
		if math.Abs(float64(variance-1)) > 1e-4 {
			t.Errorf("expected variance ≈ 1, got %v", variance)
		}
	})

	t.Run("BatchNorm2d with scale and shift", func(t *testing.T) {
		// Simple case: single value per position
		data := []float32{0, 0, 0, 0, 0, 0, 0, 0}
		input := tendo.MustFromSlice(data, 2, 1, 2, 2)

		weight := tendo.MustFromSlice([]float32{2}, 1)      // scale by 2
		bias := tendo.MustFromSlice([]float32{5}, 1)        // shift by 5
		runningMean := tendo.MustFromSlice([]float32{0}, 1) // mean = 0
		runningVar := tendo.MustFromSlice([]float32{1}, 1)  // var = 1

		result, err := tendo.NewBatchNorm2d(backend, weight, bias, runningMean, runningVar, 0, 0.1).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// With input=0, mean=0, var=1: normalized=0, output = 2*0 + 5 = 5
		resultData := result.MustData()
		for i, v := range resultData {
			if v != 5 {
				t.Errorf("at index %d: expected 5, got %v", i, v)
			}
		}
	})
}

func TestLayerNorm(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("LayerNorm basic", func(t *testing.T) {
		// Input: [2, 4] - normalize over last dimension
		data := []float32{0, 1, 2, 3, 4, 5, 6, 7}
		input := tendo.MustFromSlice(data, 2, 4)

		result, err := tendo.NewLayerNorm(backend, []int{4}, nil, nil, 1e-5).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Shape should be unchanged
		expectedShape := []int{2, 4}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		resultData := result.MustData()

		// Check each row is normalized (mean≈0, var≈1)
		for row := 0; row < 2; row++ {
			sum := float32(0)
			for col := 0; col < 4; col++ {
				sum += resultData[row*4+col]
			}
			mean := sum / 4

			if math.Abs(float64(mean)) > 1e-5 {
				t.Errorf("row %d: expected mean ≈ 0, got %v", row, mean)
			}

			sumSq := float32(0)
			for col := 0; col < 4; col++ {
				v := resultData[row*4+col]
				sumSq += v * v
			}
			variance := sumSq / 4

			if math.Abs(float64(variance-1)) > 1e-4 {
				t.Errorf("row %d: expected variance ≈ 1, got %v", row, variance)
			}
		}
	})

	t.Run("LayerNorm with weight and bias", func(t *testing.T) {
		// All zeros input
		data := []float32{0, 0, 0, 0}
		input := tendo.MustFromSlice(data, 1, 4)

		weight := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 4)
		bias := tendo.MustFromSlice([]float32{10, 20, 30, 40}, 4)

		result, err := tendo.NewLayerNorm(backend, []int{4}, weight, bias, 1e-5).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// With all zeros, normalized is all zeros (0-0)/sqrt(0+eps) ≈ 0
		// Output = weight * 0 + bias = bias
		resultData := result.MustData()
		expected := []float32{10, 20, 30, 40}
		for i, v := range expected {
			if math.Abs(float64(resultData[i]-v)) > 1e-3 {
				t.Errorf("at index %d: expected %v, got %v", i, v, resultData[i])
			}
		}
	})

	t.Run("LayerNorm 3D", func(t *testing.T) {
		// Input: [2, 3, 4] - normalize over last 2 dimensions
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i)
		}
		input := tendo.MustFromSlice(data, 2, 3, 4)

		result, err := tendo.NewLayerNorm(backend, []int{3, 4}, nil, nil, 1e-5).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 3, 4}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		resultData := result.MustData()

		// Check each batch is normalized
		for batch := 0; batch < 2; batch++ {
			offset := batch * 12
			sum := float32(0)
			for i := 0; i < 12; i++ {
				sum += resultData[offset+i]
			}
			mean := sum / 12

			if math.Abs(float64(mean)) > 1e-5 {
				t.Errorf("batch %d: expected mean ≈ 0, got %v", batch, mean)
			}
		}
	})

	t.Run("LayerNorm shape mismatch error", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2)

		// normalizedShape doesn't match input's trailing dimensions
		_, err := tendo.NewLayerNorm(backend, []int{3}, nil, nil, 1e-5).Process(ctx, input)
		if err == nil {
			t.Error("expected error for shape mismatch")
		}
	})
}

// shapesEqual compares two shapes for equality.
func shapesEqual(a, b []int) bool {
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
