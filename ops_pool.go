package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// Pool2dConfig configures 2D pooling operations.
type Pool2dConfig struct {
	KernelSize [2]int // [height, width]
	Stride     [2]int // [height, width] - defaults to KernelSize if zero
	Padding    [2]int // [height, width]
}

// MaxPool2d is a chainable operator that applies 2D max pooling.
// Input shape: [N, C, H, W] or [C, H, W]
// Output shape: [N, C, H', W'] or [C, H', W'] where:
//
//	H' = (H + 2*padding[0] - kernel[0]) / stride[0] + 1
//	W' = (W + 2*padding[1] - kernel[1]) / stride[1] + 1
type MaxPool2d struct {
	backend PoolOps
	config  Pool2dConfig
}

// NewMaxPool2d creates a MaxPool2d operator.
func NewMaxPool2d(backend PoolOps, config Pool2dConfig) *MaxPool2d {
	return &MaxPool2d{backend: backend, config: config}
}

// Process applies 2D max pooling.
func (m *MaxPool2d) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	// Default stride to kernel size
	stride := m.config.Stride
	if stride[0] == 0 {
		stride[0] = m.config.KernelSize[0]
	}
	if stride[1] == 0 {
		stride[1] = m.config.KernelSize[1]
	}

	result, indices, err := m.backend.MaxPool2d(ctx, t, m.config.KernelSize, stride, m.config.Padding)
	if err != nil {
		return nil, fmt.Errorf("maxpool2d: %w", err)
	}

	// Convert indices to tensor if provided
	var indicesTensor *Tensor
	if len(indices) > 0 {
		indicesStorage := NewCPUStorageFromSlice(make([]float32, len(indices)), Float32)
		for i, idx := range indices {
			indicesStorage.Data()[i] = float32(idx)
		}
		indicesTensor = NewTensor(indicesStorage, result.Shape(), nil)
	}

	emitWithTrace(ctx, OpMaxPool2d,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyKernelSize.Field(m.config.KernelSize),
		KeyPoolStride.Field(stride),
		KeyPadding.Field(m.config.Padding),
		KeyIndices.Field(indicesTensor),
	)

	propagateTape(t, result, "maxpool2d", map[string]*Tensor{"input": t, "indices": indicesTensor})

	return result, nil
}

// Name returns the operator name.
func (m *MaxPool2d) Name() pipz.Name { return "maxpool2d" }

// Close releases any resources held by this operator.
func (m *MaxPool2d) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*MaxPool2d)(nil)

// AvgPool2d is a chainable operator that applies 2D average pooling.
// Input shape: [N, C, H, W] or [C, H, W]
// Output shape: [N, C, H', W'] or [C, H', W'].
type AvgPool2d struct {
	backend PoolOps
	config  Pool2dConfig
}

// NewAvgPool2d creates an AvgPool2d operator.
func NewAvgPool2d(backend PoolOps, config Pool2dConfig) *AvgPool2d {
	return &AvgPool2d{backend: backend, config: config}
}

// Process applies 2D average pooling.
func (a *AvgPool2d) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	// Default stride to kernel size
	stride := a.config.Stride
	if stride[0] == 0 {
		stride[0] = a.config.KernelSize[0]
	}
	if stride[1] == 0 {
		stride[1] = a.config.KernelSize[1]
	}

	result, err := a.backend.AvgPool2d(ctx, t, a.config.KernelSize, stride, a.config.Padding)
	if err != nil {
		return nil, fmt.Errorf("avgpool2d: %w", err)
	}

	emitWithTrace(ctx, OpAvgPool2d,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyKernelSize.Field(a.config.KernelSize),
		KeyPoolStride.Field(stride),
		KeyPadding.Field(a.config.Padding),
	)

	propagateTape(t, result, "avgpool2d", map[string]*Tensor{"input": t})

	return result, nil
}

// Name returns the operator name.
func (a *AvgPool2d) Name() pipz.Name { return "avgpool2d" }

// Close releases any resources held by this operator.
func (a *AvgPool2d) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*AvgPool2d)(nil)

// AdaptiveAvgPool2d is a chainable operator that applies adaptive 2D average pooling.
// The output has a fixed spatial size regardless of input size.
// Input shape: [N, C, H, W] or [C, H, W].
type AdaptiveAvgPool2d struct {
	backend    PoolOps
	outputSize [2]int
}

// NewAdaptiveAvgPool2d creates an AdaptiveAvgPool2d operator.
func NewAdaptiveAvgPool2d(backend PoolOps, outputSize [2]int) *AdaptiveAvgPool2d {
	return &AdaptiveAvgPool2d{backend: backend, outputSize: outputSize}
}

// Process applies adaptive average pooling.
func (a *AdaptiveAvgPool2d) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	result, err := a.backend.AdaptiveAvgPool2d(ctx, t, a.outputSize)
	if err != nil {
		return nil, fmt.Errorf("adaptiveavgpool2d: %w", err)
	}

	emitWithTrace(ctx, OpAdaptiveAvgPool2d,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyOutputSize.Field(a.outputSize),
	)

	propagateTape(t, result, "adaptiveavgpool2d", map[string]*Tensor{"input": t})

	return result, nil
}

// Name returns the operator name.
func (a *AdaptiveAvgPool2d) Name() pipz.Name { return "adaptiveavgpool2d" }

// Close releases any resources held by this operator.
func (a *AdaptiveAvgPool2d) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*AdaptiveAvgPool2d)(nil)

// AdaptiveMaxPool2d is a chainable operator that applies adaptive 2D max pooling.
// The output has a fixed spatial size regardless of input size.
// Input shape: [N, C, H, W] or [C, H, W].
type AdaptiveMaxPool2d struct {
	backend    PoolOps
	outputSize [2]int
}

// NewAdaptiveMaxPool2d creates an AdaptiveMaxPool2d operator.
func NewAdaptiveMaxPool2d(backend PoolOps, outputSize [2]int) *AdaptiveMaxPool2d {
	return &AdaptiveMaxPool2d{backend: backend, outputSize: outputSize}
}

// Process applies adaptive max pooling.
func (a *AdaptiveMaxPool2d) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	result, indices, err := a.backend.AdaptiveMaxPool2d(ctx, t, a.outputSize)
	if err != nil {
		return nil, fmt.Errorf("adaptivemaxpool2d: %w", err)
	}

	// Convert indices to tensor if provided
	var indicesTensor *Tensor
	if len(indices) > 0 {
		indicesStorage := NewCPUStorageFromSlice(make([]float32, len(indices)), Float32)
		for i, idx := range indices {
			indicesStorage.Data()[i] = float32(idx)
		}
		indicesTensor = NewTensor(indicesStorage, result.Shape(), nil)
	}

	emitWithTrace(ctx, OpAdaptiveMaxPool2d,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyOutputSize.Field(a.outputSize),
	)

	propagateTape(t, result, "adaptivemaxpool2d", map[string]*Tensor{"input": t, "indices": indicesTensor})

	return result, nil
}

// Name returns the operator name.
func (a *AdaptiveMaxPool2d) Name() pipz.Name { return "adaptivemaxpool2d" }

// Close releases any resources held by this operator.
func (a *AdaptiveMaxPool2d) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*AdaptiveMaxPool2d)(nil)
