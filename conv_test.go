package tendo_test

import (
	"context"
	"math"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cpu"
)

func TestConv2dBasic(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	// Simple 3x3 convolution on 1x1x4x4 input
	// Input: 1 batch, 1 channel, 4x4 spatial
	input := tendo.MustFromSlice([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, 1, 1, 4, 4)

	// Weight: 1 output channel, 1 input channel, 3x3 kernel (all ones)
	weight := tendo.MustFromSlice([]float32{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}, 1, 1, 3, 3)

	config := tendo.DefaultConv2dConfig()

	result, err := tendo.NewConv2d(backend, weight, config).Process(ctx, input)
	if err != nil {
		t.Fatalf("Conv2d failed: %v", err)
	}

	// Expected output shape: [1, 1, 2, 2] (no padding, stride 1)
	expectedShape := []int{1, 1, 2, 2}
	if !shapesEqual(result.Shape(), expectedShape) {
		t.Errorf("Shape mismatch: got %v, expected %v", result.Shape(), expectedShape)
	}

	// Expected values (sum of 3x3 regions)
	// Top-left 3x3: 1+2+3+5+6+7+9+10+11 = 54
	// Top-right 3x3: 2+3+4+6+7+8+10+11+12 = 63
	// Bottom-left 3x3: 5+6+7+9+10+11+13+14+15 = 90
	// Bottom-right 3x3: 6+7+8+10+11+12+14+15+16 = 99
	data := result.MustData()
	expected := []float32{54, 63, 90, 99}

	for i, v := range expected {
		if math.Abs(float64(data[i]-v)) > 0.001 {
			t.Errorf("Value mismatch at index %d: got %v, expected %v", i, data[i], v)
		}
	}
}

func TestConv2dWithPadding(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	// 3x3 convolution with padding=1 (same padding)
	input := tendo.MustFromSlice([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, 1, 1, 3, 3)

	// Identity-like kernel (just center = 1)
	weight := tendo.MustFromSlice([]float32{
		0, 0, 0,
		0, 1, 0,
		0, 0, 0,
	}, 1, 1, 3, 3)

	config := tendo.Conv2dConfig{
		Padding:  [2]int{1, 1},
		Stride:   [2]int{1, 1},
		Dilation: [2]int{1, 1},
		Groups:   1,
	}

	result, err := tendo.NewConv2d(backend, weight, config).Process(ctx, input)
	if err != nil {
		t.Fatalf("Conv2d failed: %v", err)
	}

	// With same padding, output should be same size as input
	expectedShape := []int{1, 1, 3, 3}
	if !shapesEqual(result.Shape(), expectedShape) {
		t.Errorf("Shape mismatch: got %v, expected %v", result.Shape(), expectedShape)
	}

	// With identity kernel, output should equal input
	data := result.MustData()
	inputData := input.MustData()

	for i, v := range inputData {
		if math.Abs(float64(data[i]-v)) > 0.001 {
			t.Errorf("Value mismatch at index %d: got %v, expected %v", i, data[i], v)
		}
	}
}

func TestConv2dWithStride(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	// 1x1 convolution with stride=2
	input := tendo.MustFromSlice([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, 1, 1, 4, 4)

	// 1x1 kernel (scaling by 2)
	weight := tendo.MustFromSlice([]float32{2}, 1, 1, 1, 1)

	config := tendo.Conv2dConfig{
		Padding:  [2]int{0, 0},
		Stride:   [2]int{2, 2},
		Dilation: [2]int{1, 1},
		Groups:   1,
	}

	result, err := tendo.NewConv2d(backend, weight, config).Process(ctx, input)
	if err != nil {
		t.Fatalf("Conv2d failed: %v", err)
	}

	// Output shape: [1, 1, 2, 2]
	expectedShape := []int{1, 1, 2, 2}
	if !shapesEqual(result.Shape(), expectedShape) {
		t.Errorf("Shape mismatch: got %v, expected %v", result.Shape(), expectedShape)
	}

	// Expected values (scaled by 2): corners at positions (0,0), (0,2), (2,0), (2,2)
	data := result.MustData()
	expected := []float32{2, 6, 18, 22}

	for i, v := range expected {
		if math.Abs(float64(data[i]-v)) > 0.001 {
			t.Errorf("Value mismatch at index %d: got %v, expected %v", i, data[i], v)
		}
	}
}

func TestConv2dMultipleChannels(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	// 2 input channels, 3 output channels
	// Input: [1, 2, 2, 2]
	input := tendo.MustFromSlice([]float32{
		// Channel 0
		1, 2,
		3, 4,
		// Channel 1
		5, 6,
		7, 8,
	}, 1, 2, 2, 2)

	// Weight: [3, 2, 1, 1] - 3 output channels, 2 input channels, 1x1 kernel
	weight := tendo.MustFromSlice([]float32{
		1, 1, // out_channel 0: sum of inputs
		1, -1, // out_channel 1: diff of inputs
		2, 0, // out_channel 2: 2x first input
	}, 3, 2, 1, 1)

	config := tendo.DefaultConv2dConfig()

	result, err := tendo.NewConv2d(backend, weight, config).Process(ctx, input)
	if err != nil {
		t.Fatalf("Conv2d failed: %v", err)
	}

	// Output shape: [1, 3, 2, 2]
	expectedShape := []int{1, 3, 2, 2}
	if !shapesEqual(result.Shape(), expectedShape) {
		t.Errorf("Shape mismatch: got %v, expected %v", result.Shape(), expectedShape)
	}

	data := result.MustData()

	// Channel 0 (sum): 1+5=6, 2+6=8, 3+7=10, 4+8=12
	// Channel 1 (diff): 1-5=-4, 2-6=-4, 3-7=-4, 4-8=-4
	// Channel 2 (2x first): 2, 4, 6, 8
	expected := []float32{6, 8, 10, 12, -4, -4, -4, -4, 2, 4, 6, 8}

	for i, v := range expected {
		if math.Abs(float64(data[i]-v)) > 0.001 {
			t.Errorf("Value mismatch at index %d: got %v, expected %v", i, data[i], v)
		}
	}
}

func TestConv2dBatched(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	// 2 batches, 1 channel, 2x2 input
	input := tendo.MustFromSlice([]float32{
		// Batch 0
		1, 2,
		3, 4,
		// Batch 1
		5, 6,
		7, 8,
	}, 2, 1, 2, 2)

	// 1x1 identity kernel
	weight := tendo.MustFromSlice([]float32{1}, 1, 1, 1, 1)

	config := tendo.DefaultConv2dConfig()

	result, err := tendo.NewConv2d(backend, weight, config).Process(ctx, input)
	if err != nil {
		t.Fatalf("Conv2d failed: %v", err)
	}

	// Output shape: [2, 1, 2, 2]
	expectedShape := []int{2, 1, 2, 2}
	if !shapesEqual(result.Shape(), expectedShape) {
		t.Errorf("Shape mismatch: got %v, expected %v", result.Shape(), expectedShape)
	}

	// Output should equal input
	data := result.MustData()
	inputData := input.MustData()

	for i, v := range inputData {
		if math.Abs(float64(data[i]-v)) > 0.001 {
			t.Errorf("Value mismatch at index %d: got %v, expected %v", i, data[i], v)
		}
	}
}

func TestConv2dOutputShape(t *testing.T) {
	tests := []struct {
		name        string
		inputShape  []int
		weightShape []int
		config      tendo.Conv2dConfig
		expected    []int
	}{
		{
			name:        "no padding stride 1",
			inputShape:  []int{1, 1, 5, 5},
			weightShape: []int{1, 1, 3, 3},
			config:      tendo.DefaultConv2dConfig(),
			expected:    []int{1, 1, 3, 3},
		},
		{
			name:        "same padding stride 1",
			inputShape:  []int{1, 1, 5, 5},
			weightShape: []int{1, 1, 3, 3},
			config: tendo.Conv2dConfig{
				Padding:  [2]int{1, 1},
				Stride:   [2]int{1, 1},
				Dilation: [2]int{1, 1},
				Groups:   1,
			},
			expected: []int{1, 1, 5, 5},
		},
		{
			name:        "stride 2",
			inputShape:  []int{1, 1, 8, 8},
			weightShape: []int{1, 1, 3, 3},
			config: tendo.Conv2dConfig{
				Padding:  [2]int{1, 1},
				Stride:   [2]int{2, 2},
				Dilation: [2]int{1, 1},
				Groups:   1,
			},
			expected: []int{1, 1, 4, 4},
		},
		{
			name:        "multiple output channels",
			inputShape:  []int{4, 3, 32, 32},
			weightShape: []int{64, 3, 3, 3},
			config: tendo.Conv2dConfig{
				Padding:  [2]int{1, 1},
				Stride:   [2]int{1, 1},
				Dilation: [2]int{1, 1},
				Groups:   1,
			},
			expected: []int{4, 64, 32, 32},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tendo.Conv2dOutputShape(tt.inputShape, tt.weightShape, tt.config)
			if !shapesEqual(result, tt.expected) {
				t.Errorf("Shape mismatch: got %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestConv2dShapeErrors(t *testing.T) {
	ctx := context.Background()
	backend := cpu.NewBackend()

	t.Run("input not 4D", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{1, 2, 3}, 3)
		weight := tendo.MustFromSlice([]float32{1}, 1, 1, 1, 1)

		_, err := tendo.NewConv2d(backend, weight, tendo.DefaultConv2dConfig()).Process(ctx, input)
		if err == nil {
			t.Error("Expected error for non-4D input")
		}
	})

	t.Run("weight not 4D", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
		weight := tendo.MustFromSlice([]float32{1}, 1)

		_, err := tendo.NewConv2d(backend, weight, tendo.DefaultConv2dConfig()).Process(ctx, input)
		if err == nil {
			t.Error("Expected error for non-4D weight")
		}
	})

	t.Run("channel mismatch", func(t *testing.T) {
		input := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 2, 2, 2) // 2 channels
		weight := tendo.MustFromSlice([]float32{1, 1, 1}, 1, 3, 1, 1)               // expects 3 channels

		_, err := tendo.NewConv2d(backend, weight, tendo.DefaultConv2dConfig()).Process(ctx, input)
		if err == nil {
			t.Error("Expected error for channel mismatch")
		}
	})
}
