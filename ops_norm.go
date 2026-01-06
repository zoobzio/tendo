package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// BatchNorm2d is a chainable operator that applies batch normalization over 4D input.
// Input shape: [N, C, H, W]
// Normalizes over N, H, W dimensions per channel.
type BatchNorm2d struct { //nolint:govet // field alignment is less important than readability
	backend     NormOps
	weight      *Tensor
	bias        *Tensor
	runningMean *Tensor
	runningVar  *Tensor
	epsilon     float32
	momentum    float32
	identity    pipz.Identity
}

// NewBatchNorm2d creates a BatchNorm2d operator.
func NewBatchNorm2d(backend NormOps, weight, bias, runningMean, runningVar *Tensor, epsilon, momentum float32) *BatchNorm2d {
	return &BatchNorm2d{
		identity:    IdentityBatchNorm2d,
		backend:     backend,
		weight:      weight,
		bias:        bias,
		runningMean: runningMean,
		runningVar:  runningVar,
		epsilon:     epsilon,
		momentum:    momentum,
	}
}

// Process applies batch normalization.
func (b *BatchNorm2d) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	training := IsTraining(ctx)
	result, err := b.backend.BatchNorm2d(ctx, t, b.weight, b.bias, b.runningMean, b.runningVar, b.epsilon, b.momentum, training)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: %w", err)
	}

	emitWithTrace(ctx, OpBatchNorm,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyWeight.Field(b.weight),
		KeyBias.Field(b.bias),
		KeyRunningMean.Field(b.runningMean),
		KeyRunningVar.Field(b.runningVar),
		KeyEpsilon.Field(b.epsilon),
		KeyMomentum.Field(b.momentum),
	)


	return result, nil
}

// Identity returns the operator identity.
func (b *BatchNorm2d) Identity() pipz.Identity { return b.identity }

// Schema returns the operator schema node.
func (b *BatchNorm2d) Schema() pipz.Node { return pipz.Node{Identity: b.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (b *BatchNorm2d) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*BatchNorm2d)(nil)

// LayerNorm is a chainable operator that applies layer normalization.
// Normalizes over the last len(normalizedShape) dimensions.
type LayerNorm struct {
	identity        pipz.Identity
	backend         NormOps
	weight          *Tensor
	bias            *Tensor
	normalizedShape []int
	epsilon         float32
}

// NewLayerNorm creates a LayerNorm operator.
func NewLayerNorm(backend NormOps, normalizedShape []int, weight, bias *Tensor, epsilon float32) *LayerNorm {
	return &LayerNorm{
		identity:        IdentityLayerNorm,
		backend:         backend,
		normalizedShape: normalizedShape,
		weight:          weight,
		bias:            bias,
		epsilon:         epsilon,
	}
}

// Process applies layer normalization.
func (l *LayerNorm) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	result, err := l.backend.LayerNorm(ctx, t, l.normalizedShape, l.weight, l.bias, l.epsilon)
	if err != nil {
		return nil, fmt.Errorf("layernorm: %w", err)
	}

	emitWithTrace(ctx, OpLayerNorm,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyNormalizedShape.Field(l.normalizedShape),
		KeyWeight.Field(l.weight),
		KeyBias.Field(l.bias),
		KeyEpsilon.Field(l.epsilon),
	)


	return result, nil
}

// Identity returns the operator identity.
func (l *LayerNorm) Identity() pipz.Identity { return l.identity }

// Schema returns the operator schema node.
func (l *LayerNorm) Schema() pipz.Node { return pipz.Node{Identity: l.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (l *LayerNorm) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*LayerNorm)(nil)

// RMSNorm is a chainable operator that applies RMS (Root Mean Square) normalization.
// Normalizes over the last len(normalizedShape) dimensions without mean subtraction.
type RMSNorm struct {
	identity        pipz.Identity
	backend         NormOps
	weight          *Tensor
	normalizedShape []int
	epsilon         float32
}

// NewRMSNorm creates an RMSNorm operator.
func NewRMSNorm(backend NormOps, normalizedShape []int, weight *Tensor, epsilon float32) *RMSNorm {
	return &RMSNorm{
		identity:        IdentityRMSNorm,
		backend:         backend,
		normalizedShape: normalizedShape,
		weight:          weight,
		epsilon:         epsilon,
	}
}

// Process applies RMS normalization.
func (r *RMSNorm) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	result, err := r.backend.RMSNorm(ctx, t, r.normalizedShape, r.weight, r.epsilon)
	if err != nil {
		return nil, fmt.Errorf("rmsnorm: %w", err)
	}

	emitWithTrace(ctx, OpRMSNorm,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyNormalizedShape.Field(r.normalizedShape),
		KeyWeight.Field(r.weight),
		KeyEpsilon.Field(r.epsilon),
	)


	return result, nil
}

// Identity returns the operator identity.
func (r *RMSNorm) Identity() pipz.Identity { return r.identity }

// Schema returns the operator schema node.
func (r *RMSNorm) Schema() pipz.Node { return pipz.Node{Identity: r.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (r *RMSNorm) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*RMSNorm)(nil)

// GroupNorm is a chainable operator that applies group normalization.
// Divides channels into groups and normalizes within each group.
// Input shape: [N, C, ...] where C must be divisible by numGroups.
type GroupNorm struct { //nolint:govet // field alignment is less important than readability
	backend   NormOps
	weight    *Tensor
	bias      *Tensor
	numGroups int
	epsilon   float32
	identity  pipz.Identity
}

// NewGroupNorm creates a GroupNorm operator.
func NewGroupNorm(backend NormOps, numGroups int, weight, bias *Tensor, epsilon float32) *GroupNorm {
	return &GroupNorm{
		identity:  IdentityGroupNorm,
		backend:   backend,
		numGroups: numGroups,
		weight:    weight,
		bias:      bias,
		epsilon:   epsilon,
	}
}

// Process applies group normalization.
func (g *GroupNorm) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	result, err := g.backend.GroupNorm(ctx, t, g.numGroups, g.weight, g.bias, g.epsilon)
	if err != nil {
		return nil, fmt.Errorf("groupnorm: %w", err)
	}

	emitWithTrace(ctx, OpGroupNorm,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyNumGroups.Field(g.numGroups),
		KeyWeight.Field(g.weight),
		KeyBias.Field(g.bias),
		KeyEpsilon.Field(g.epsilon),
	)


	return result, nil
}

// Identity returns the operator identity.
func (g *GroupNorm) Identity() pipz.Identity { return g.identity }

// Schema returns the operator schema node.
func (g *GroupNorm) Schema() pipz.Node { return pipz.Node{Identity: g.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (g *GroupNorm) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*GroupNorm)(nil)

// InstanceNorm2d is a chainable operator that applies instance normalization over 4D input.
// Input shape: [N, C, H, W]
// Normalizes over H, W dimensions independently for each channel and sample.
type InstanceNorm2d struct { //nolint:govet // field alignment is less important than readability
	backend  NormOps
	weight   *Tensor
	bias     *Tensor
	epsilon  float32
	identity pipz.Identity
}

// NewInstanceNorm2d creates an InstanceNorm2d operator.
func NewInstanceNorm2d(backend NormOps, weight, bias *Tensor, epsilon float32) *InstanceNorm2d {
	return &InstanceNorm2d{
		identity: IdentityInstanceNorm2d,
		backend:  backend,
		weight:   weight,
		bias:     bias,
		epsilon:  epsilon,
	}
}

// Process applies instance normalization.
func (i *InstanceNorm2d) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	result, err := i.backend.InstanceNorm2d(ctx, t, i.weight, i.bias, i.epsilon)
	if err != nil {
		return nil, fmt.Errorf("instancenorm2d: %w", err)
	}

	emitWithTrace(ctx, OpInstanceNorm,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyWeight.Field(i.weight),
		KeyBias.Field(i.bias),
		KeyEpsilon.Field(i.epsilon),
	)


	return result, nil
}

// Identity returns the operator identity.
func (i *InstanceNorm2d) Identity() pipz.Identity { return i.identity }

// Schema returns the operator schema node.
func (i *InstanceNorm2d) Schema() pipz.Node { return pipz.Node{Identity: i.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (i *InstanceNorm2d) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*InstanceNorm2d)(nil)
