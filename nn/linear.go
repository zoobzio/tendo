package nn

import (
	"context"
	"fmt"

	"github.com/zoobzio/tendo"
)

// LinearBackend defines the operations needed for a Linear layer.
type LinearBackend interface {
	MatMul(ctx context.Context, a, b *tendo.Tensor) (*tendo.Tensor, error)
	Add(ctx context.Context, a, b *tendo.Tensor) (*tendo.Tensor, error)
}

// Linear represents a fully connected layer: y = xW^T + b
type Linear struct {
	Weight *tendo.Tensor // [out_features, in_features]
	Bias   *tendo.Tensor // [out_features], optional
}

// NewLinear creates a Linear layer from weight and optional bias tensors.
// Weight shape: [out_features, in_features]
// Bias shape: [out_features] or nil
func NewLinear(weight, bias *tendo.Tensor) (*Linear, error) {
	if weight == nil {
		return nil, fmt.Errorf("nn.Linear: weight cannot be nil")
	}
	if weight.Dim() != 2 {
		return nil, fmt.Errorf("nn.Linear: weight must be 2D, got %dD", weight.Dim())
	}
	if bias != nil {
		if bias.Dim() != 1 {
			return nil, fmt.Errorf("nn.Linear: bias must be 1D, got %dD", bias.Dim())
		}
		if bias.Size(0) != weight.Size(0) {
			return nil, fmt.Errorf("nn.Linear: bias size %d != out_features %d", bias.Size(0), weight.Size(0))
		}
	}
	return &Linear{Weight: weight, Bias: bias}, nil
}

// InFeatures returns the input dimension.
func (l *Linear) InFeatures() int {
	return l.Weight.Size(1)
}

// OutFeatures returns the output dimension.
func (l *Linear) OutFeatures() int {
	return l.Weight.Size(0)
}

// Forward computes y = xW^T + b
// Input shape: [..., in_features]
// Output shape: [..., out_features]
func (l *Linear) Forward(ctx context.Context, x *tendo.Tensor, backend LinearBackend) (*tendo.Tensor, error) {
	// x: [..., in_features]
	// weight: [out_features, in_features]
	// weight.T: [in_features, out_features]
	// result: [..., out_features]

	// Transpose weight for matmul: [out, in] -> [in, out]
	weightT, err := tendo.NewT().Process(ctx, l.Weight)
	if err != nil {
		return nil, fmt.Errorf("nn.Linear: transpose weight: %w", err)
	}

	// MatMul: [..., in] @ [in, out] -> [..., out]
	out, err := backend.MatMul(ctx, x, weightT)
	if err != nil {
		return nil, fmt.Errorf("nn.Linear: matmul: %w", err)
	}

	// Add bias if present
	if l.Bias != nil {
		out, err = backend.Add(ctx, out, l.Bias)
		if err != nil {
			return nil, fmt.Errorf("nn.Linear: add bias: %w", err)
		}
	}

	return out, nil
}
