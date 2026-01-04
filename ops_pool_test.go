package tendo_test

import (
	"context"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cpu"
)

func TestMaxPool2d(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("MaxPool2d basic", func(t *testing.T) {
		// Input: [1, 1, 4, 4]
		data := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}
		input := tendo.MustFromSlice(data, 1, 1, 4, 4)

		config := tendo.Pool2dConfig{
			KernelSize: [2]int{2, 2},
			Stride:     [2]int{2, 2},
		}

		result, err := tendo.NewMaxPool2d(backend, config).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Expected output shape: [1, 1, 2, 2]
		expectedShape := []int{1, 1, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Expected values: max of each 2x2 region
		// Top-left 2x2: max(1,2,5,6) = 6
		// Top-right 2x2: max(3,4,7,8) = 8
		// Bottom-left 2x2: max(9,10,13,14) = 14
		// Bottom-right 2x2: max(11,12,15,16) = 16
		expected := []float32{6, 8, 14, 16}
		resultData := result.MustData()
		for i, v := range expected {
			if resultData[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, resultData[i])
			}
		}
	})

	t.Run("MaxPool2d with stride 1", func(t *testing.T) {
		// Input: [1, 1, 3, 3]
		data := []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		}
		input := tendo.MustFromSlice(data, 1, 1, 3, 3)

		config := tendo.Pool2dConfig{
			KernelSize: [2]int{2, 2},
			Stride:     [2]int{1, 1},
		}

		result, err := tendo.NewMaxPool2d(backend, config).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Expected output shape: [1, 1, 2, 2]
		expectedShape := []int{1, 1, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Expected values:
		// Position (0,0): max(1,2,4,5) = 5
		// Position (0,1): max(2,3,5,6) = 6
		// Position (1,0): max(4,5,7,8) = 8
		// Position (1,1): max(5,6,8,9) = 9
		expected := []float32{5, 6, 8, 9}
		resultData := result.MustData()
		for i, v := range expected {
			if resultData[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, resultData[i])
			}
		}
	})

	t.Run("MaxPool2d 3D input", func(t *testing.T) {
		// Input: [1, 4, 4] (no batch dimension)
		data := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}
		input := tendo.MustFromSlice(data, 1, 4, 4)

		config := tendo.Pool2dConfig{
			KernelSize: [2]int{2, 2},
		}

		result, err := tendo.NewMaxPool2d(backend, config).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Expected output shape: [1, 2, 2] (no batch)
		expectedShape := []int{1, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("MaxPool2d with padding", func(t *testing.T) {
		// Input: [1, 1, 2, 2]
		data := []float32{1, 2, 3, 4}
		input := tendo.MustFromSlice(data, 1, 1, 2, 2)

		config := tendo.Pool2dConfig{
			KernelSize: [2]int{2, 2},
			Stride:     [2]int{2, 2},
			Padding:    [2]int{1, 1},
		}

		result, err := tendo.NewMaxPool2d(backend, config).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// With padding=1, kernel=2, stride=2 on 2x2 input:
		// padded input is 4x4 with zeros around
		// Output shape: ((2+2-2)/2+1) x ((2+2-2)/2+1) = 2x2
		expectedShape := []int{1, 1, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})

	t.Run("MaxPool2d multi-channel", func(t *testing.T) {
		// Input: [1, 2, 4, 4] - 2 channels
		data := make([]float32, 32)
		for i := range data {
			data[i] = float32(i + 1)
		}
		input := tendo.MustFromSlice(data, 1, 2, 4, 4)

		config := tendo.Pool2dConfig{
			KernelSize: [2]int{2, 2},
			Stride:     [2]int{2, 2},
		}

		result, err := tendo.NewMaxPool2d(backend, config).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{1, 2, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})
}

func TestAvgPool2d(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("AvgPool2d basic", func(t *testing.T) {
		// Input: [1, 1, 4, 4]
		data := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}
		input := tendo.MustFromSlice(data, 1, 1, 4, 4)

		config := tendo.Pool2dConfig{
			KernelSize: [2]int{2, 2},
			Stride:     [2]int{2, 2},
		}

		result, err := tendo.NewAvgPool2d(backend, config).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Expected output shape: [1, 1, 2, 2]
		expectedShape := []int{1, 1, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Expected values: avg of each 2x2 region
		// Top-left 2x2: avg(1,2,5,6) = 3.5
		// Top-right 2x2: avg(3,4,7,8) = 5.5
		// Bottom-left 2x2: avg(9,10,13,14) = 11.5
		// Bottom-right 2x2: avg(11,12,15,16) = 13.5
		expected := []float32{3.5, 5.5, 11.5, 13.5}
		resultData := result.MustData()
		for i, v := range expected {
			if resultData[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, resultData[i])
			}
		}
	})

	t.Run("AvgPool2d with stride 1", func(t *testing.T) {
		// Input: [1, 1, 3, 3]
		data := []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		}
		input := tendo.MustFromSlice(data, 1, 1, 3, 3)

		config := tendo.Pool2dConfig{
			KernelSize: [2]int{2, 2},
			Stride:     [2]int{1, 1},
		}

		result, err := tendo.NewAvgPool2d(backend, config).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Expected output shape: [1, 1, 2, 2]
		expectedShape := []int{1, 1, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		// Expected values:
		// Position (0,0): avg(1,2,4,5) = 3
		// Position (0,1): avg(2,3,5,6) = 4
		// Position (1,0): avg(4,5,7,8) = 6
		// Position (1,1): avg(5,6,8,9) = 7
		expected := []float32{3, 4, 6, 7}
		resultData := result.MustData()
		for i, v := range expected {
			if resultData[i] != v {
				t.Errorf("at index %d: expected %v, got %v", i, v, resultData[i])
			}
		}
	})

	t.Run("AvgPool2d 3D input", func(t *testing.T) {
		// Input: [1, 4, 4] (no batch dimension)
		data := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}
		input := tendo.MustFromSlice(data, 1, 4, 4)

		config := tendo.Pool2dConfig{
			KernelSize: [2]int{2, 2},
		}

		result, err := tendo.NewAvgPool2d(backend, config).Process(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Expected output shape: [1, 2, 2] (no batch)
		expectedShape := []int{1, 2, 2}
		if !shapesEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
	})
}
