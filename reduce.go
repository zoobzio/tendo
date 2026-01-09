package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// Sum is a chainable operator that sums elements along dimensions.
// If no dimensions are specified, sums all elements.
type Sum struct {
	identity pipz.Identity
	backend  ReduceOps
	dims     []int
	keepdim  bool
}

// NewSum creates a Sum operator.
func NewSum(backend ReduceOps, keepdim bool, dims ...int) *Sum {
	return &Sum{
		identity: IdentitySum,
		backend:  backend,
		dims:     dims,
		keepdim:  keepdim,
	}
}

// Process sums elements along the specified dimensions.
func (s *Sum) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.Sum(ctx, t, s.dims, s.keepdim)
	if err != nil {
		return nil, fmt.Errorf("sum: %w", err)
	}

	emitWithTrace(ctx, OpSum,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDims.Field(s.dims),
	)


	return out, nil
}

// Identity returns the operator identity.
func (s *Sum) Identity() pipz.Identity { return s.identity }

// Schema returns the operator schema.
func (s *Sum) Schema() pipz.Node { return pipz.Node{Identity: s.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (s *Sum) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Sum)(nil)

// Mean is a chainable operator that computes the mean along dimensions.
// If no dimensions are specified, computes mean of all elements.
type Mean struct {
	identity pipz.Identity
	backend  ReduceOps
	dims     []int
	keepdim  bool
}

// NewMean creates a Mean operator.
func NewMean(backend ReduceOps, keepdim bool, dims ...int) *Mean {
	return &Mean{
		identity: IdentityMean,
		backend:  backend,
		dims:     dims,
		keepdim:  keepdim,
	}
}

// Process computes the mean along the specified dimensions.
func (m *Mean) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := m.backend.Mean(ctx, t, m.dims, m.keepdim)
	if err != nil {
		return nil, fmt.Errorf("mean: %w", err)
	}

	emitWithTrace(ctx, OpMean,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDims.Field(m.dims),
	)


	return out, nil
}

// Identity returns the operator identity.
func (m *Mean) Identity() pipz.Identity { return m.identity }

// Schema returns the operator schema.
func (m *Mean) Schema() pipz.Node { return pipz.Node{Identity: m.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (m *Mean) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Mean)(nil)

// Max is a chainable operator that computes the maximum along dimensions.
// If no dimensions are specified, computes max of all elements.
type Max struct {
	identity pipz.Identity
	backend  ReduceOps
	dims     []int
	keepdim  bool
}

// NewMax creates a Max operator.
func NewMax(backend ReduceOps, keepdim bool, dims ...int) *Max {
	return &Max{
		identity: IdentityMax,
		backend:  backend,
		dims:     dims,
		keepdim:  keepdim,
	}
}

// Process computes the maximum along the specified dimensions.
func (m *Max) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := m.backend.Max(ctx, t, m.dims, m.keepdim)
	if err != nil {
		return nil, fmt.Errorf("max: %w", err)
	}

	emitWithTrace(ctx, OpMax,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDims.Field(m.dims),
	)


	return out, nil
}

// Identity returns the operator identity.
func (m *Max) Identity() pipz.Identity { return m.identity }

// Schema returns the operator schema.
func (m *Max) Schema() pipz.Node { return pipz.Node{Identity: m.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (m *Max) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Max)(nil)

// Min is a chainable operator that computes the minimum along dimensions.
// If no dimensions are specified, computes min of all elements.
type Min struct {
	identity pipz.Identity
	backend  ReduceOps
	dims     []int
	keepdim  bool
}

// NewMin creates a Min operator.
func NewMin(backend ReduceOps, keepdim bool, dims ...int) *Min {
	return &Min{
		identity: IdentityMin,
		backend:  backend,
		dims:     dims,
		keepdim:  keepdim,
	}
}

// Process computes the minimum along the specified dimensions.
func (m *Min) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := m.backend.Min(ctx, t, m.dims, m.keepdim)
	if err != nil {
		return nil, fmt.Errorf("min: %w", err)
	}

	emitWithTrace(ctx, OpMin,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDims.Field(m.dims),
	)


	return out, nil
}

// Identity returns the operator identity.
func (m *Min) Identity() pipz.Identity { return m.identity }

// Schema returns the operator schema.
func (m *Min) Schema() pipz.Node { return pipz.Node{Identity: m.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (m *Min) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Min)(nil)

// ArgMax is a chainable operator that returns indices of max values along a dimension.
type ArgMax struct { //nolint:govet // field alignment is less important than readability
	backend  ReduceOps
	dim      int
	keepdim  bool
	identity pipz.Identity
}

// NewArgMax creates an ArgMax operator.
func NewArgMax(backend ReduceOps, dim int, keepdim bool) *ArgMax {
	return &ArgMax{
		identity: IdentityArgMax,
		backend:  backend,
		dim:      dim,
		keepdim:  keepdim,
	}
}

// Process computes indices of max values along the dimension.
func (a *ArgMax) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := a.backend.ArgMax(ctx, t, a.dim, a.keepdim)
	if err != nil {
		return nil, fmt.Errorf("argmax: %w", err)
	}

	emitWithTrace(ctx, OpArgMax,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDim.Field(a.dim),
	)

	return out, nil
}

// Identity returns the operator identity.
func (a *ArgMax) Identity() pipz.Identity { return a.identity }

// Schema returns the operator schema.
func (a *ArgMax) Schema() pipz.Node { return pipz.Node{Identity: a.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (a *ArgMax) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*ArgMax)(nil)

// ArgMin is a chainable operator that returns indices of min values along a dimension.
type ArgMin struct { //nolint:govet // field alignment is less important than readability
	backend  ReduceOps
	dim      int
	keepdim  bool
	identity pipz.Identity
}

// NewArgMin creates an ArgMin operator.
func NewArgMin(backend ReduceOps, dim int, keepdim bool) *ArgMin {
	return &ArgMin{
		identity: IdentityArgMin,
		backend:  backend,
		dim:      dim,
		keepdim:  keepdim,
	}
}

// Process computes indices of min values along the dimension.
func (a *ArgMin) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := a.backend.ArgMin(ctx, t, a.dim, a.keepdim)
	if err != nil {
		return nil, fmt.Errorf("argmin: %w", err)
	}

	emitWithTrace(ctx, OpArgMin,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDim.Field(a.dim),
	)

	return out, nil
}

// Identity returns the operator identity.
func (a *ArgMin) Identity() pipz.Identity { return a.identity }

// Schema returns the operator schema.
func (a *ArgMin) Schema() pipz.Node { return pipz.Node{Identity: a.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (a *ArgMin) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*ArgMin)(nil)

// Var is a chainable operator that computes variance along dimensions.
// Uses Bessel's correction (N-1) by default for unbiased estimation.
type Var struct {
	identity   pipz.Identity
	backend    ReduceOps
	dims       []int
	keepdim    bool
	correction int
}

// NewVar creates a Var operator.
func NewVar(backend ReduceOps, keepdim bool, correction int, dims ...int) *Var {
	return &Var{
		identity:   IdentityVar,
		backend:    backend,
		dims:       dims,
		keepdim:    keepdim,
		correction: correction,
	}
}

// Process computes variance along the specified dimensions.
func (v *Var) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := v.backend.Var(ctx, t, v.dims, v.keepdim, v.correction)
	if err != nil {
		return nil, fmt.Errorf("var: %w", err)
	}

	emitWithTrace(ctx, OpVar,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDims.Field(v.dims),
	)


	return out, nil
}

// Identity returns the operator identity.
func (v *Var) Identity() pipz.Identity { return v.identity }

// Schema returns the operator schema.
func (v *Var) Schema() pipz.Node { return pipz.Node{Identity: v.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (v *Var) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Var)(nil)

// Std is a chainable operator that computes standard deviation along dimensions.
// Uses Bessel's correction (N-1) by default for unbiased estimation.
type Std struct {
	identity   pipz.Identity
	backend    ReduceOps
	dims       []int
	keepdim    bool
	correction int
}

// NewStd creates a Std operator.
func NewStd(backend ReduceOps, keepdim bool, correction int, dims ...int) *Std {
	return &Std{
		identity:   IdentityStd,
		backend:    backend,
		dims:       dims,
		keepdim:    keepdim,
		correction: correction,
	}
}

// Process computes standard deviation along the specified dimensions.
func (s *Std) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.Std(ctx, t, s.dims, s.keepdim, s.correction)
	if err != nil {
		return nil, fmt.Errorf("std: %w", err)
	}

	emitWithTrace(ctx, OpStd,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDims.Field(s.dims),
	)


	return out, nil
}

// Identity returns the operator identity.
func (s *Std) Identity() pipz.Identity { return s.identity }

// Schema returns the operator schema.
func (s *Std) Schema() pipz.Node { return pipz.Node{Identity: s.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (s *Std) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Std)(nil)

// Prod is a chainable operator that computes the product along dimensions.
// If no dimensions are specified, computes product of all elements.
type Prod struct {
	identity pipz.Identity
	backend  ReduceOps
	dims     []int
	keepdim  bool
}

// NewProd creates a Prod operator.
func NewProd(backend ReduceOps, keepdim bool, dims ...int) *Prod {
	return &Prod{
		identity: IdentityProd,
		backend:  backend,
		dims:     dims,
		keepdim:  keepdim,
	}
}

// Process computes the product along the specified dimensions.
func (p *Prod) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := p.backend.Prod(ctx, t, p.dims, p.keepdim)
	if err != nil {
		return nil, fmt.Errorf("prod: %w", err)
	}

	emitWithTrace(ctx, OpProd,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDims.Field(p.dims),
	)


	return out, nil
}

// Identity returns the operator identity.
func (p *Prod) Identity() pipz.Identity { return p.identity }

// Schema returns the operator schema.
func (p *Prod) Schema() pipz.Node { return pipz.Node{Identity: p.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (p *Prod) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Prod)(nil)
