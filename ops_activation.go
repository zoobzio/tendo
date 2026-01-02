package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// ReLU is a chainable operator that applies the ReLU activation function.
// ReLU(x) = max(0, x).
type ReLU struct {
	backend UnaryOps
}

// NewReLU creates a ReLU operator.
func NewReLU(backend UnaryOps) *ReLU {
	return &ReLU{backend: backend}
}

// Process applies ReLU to the input tensor.
func (r *ReLU) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := r.backend.ReLU(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("relu: %w", err)
	}

	emitWithTrace(ctx, OpReLU,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "relu", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (r *ReLU) Name() pipz.Name { return "relu" }

// Close releases any resources held by this operator.
func (r *ReLU) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*ReLU)(nil)

// Sigmoid is a chainable operator that applies the sigmoid activation function.
// Sigmoid(x) = 1 / (1 + exp(-x)).
type Sigmoid struct {
	backend UnaryOps
}

// NewSigmoid creates a Sigmoid operator.
func NewSigmoid(backend UnaryOps) *Sigmoid {
	return &Sigmoid{backend: backend}
}

// Process applies sigmoid to the input tensor.
func (s *Sigmoid) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.Sigmoid(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("sigmoid: %w", err)
	}

	emitWithTrace(ctx, OpSigmoid,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "sigmoid", map[string]*Tensor{"output": out})

	return out, nil
}

// Name returns the operator name.
func (s *Sigmoid) Name() pipz.Name { return "sigmoid" }

// Close releases any resources held by this operator.
func (s *Sigmoid) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Sigmoid)(nil)

// Tanh is a chainable operator that applies the tanh activation function.
type Tanh struct {
	backend UnaryOps
}

// NewTanh creates a Tanh operator.
func NewTanh(backend UnaryOps) *Tanh {
	return &Tanh{backend: backend}
}

// Process applies tanh to the input tensor.
func (t *Tanh) Process(ctx context.Context, in *Tensor) (*Tensor, error) {
	out, err := t.backend.Tanh(ctx, in)
	if err != nil {
		return nil, fmt.Errorf("tanh: %w", err)
	}

	emitWithTrace(ctx, OpTanh,
		KeyInput.Field(in),
		KeyOutput.Field(out),
	)

	propagateTape(in, out, "tanh", map[string]*Tensor{"output": out})

	return out, nil
}

// Name returns the operator name.
func (t *Tanh) Name() pipz.Name { return "tanh" }

// Close releases any resources held by this operator.
func (t *Tanh) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Tanh)(nil)

// GELU is a chainable operator that applies the GELU activation function.
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
type GELU struct {
	backend UnaryOps
}

// NewGELU creates a GELU operator.
func NewGELU(backend UnaryOps) *GELU {
	return &GELU{backend: backend}
}

// Process applies GELU to the input tensor.
func (g *GELU) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := g.backend.GELU(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("gelu: %w", err)
	}

	emitWithTrace(ctx, OpGELU,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "gelu", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (g *GELU) Name() pipz.Name { return "gelu" }

// Close releases any resources held by this operator.
func (g *GELU) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*GELU)(nil)

// SiLU is a chainable operator that applies the SiLU (Swish) activation.
// SiLU(x) = x * sigmoid(x).
type SiLU struct {
	backend UnaryOps
}

// NewSiLU creates a SiLU operator.
func NewSiLU(backend UnaryOps) *SiLU {
	return &SiLU{backend: backend}
}

// Process applies SiLU to the input tensor.
func (s *SiLU) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.SiLU(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("silu: %w", err)
	}

	emitWithTrace(ctx, OpSiLU,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "silu", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (s *SiLU) Name() pipz.Name { return "silu" }

// Close releases any resources held by this operator.
func (s *SiLU) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*SiLU)(nil)

// Softmax is a chainable operator that applies softmax along a dimension.
// Softmax(x)_i = exp(x_i) / sum(exp(x_j)).
type Softmax struct {
	backend ActivationOps
	dim     int
}

// NewSoftmax creates a Softmax operator.
func NewSoftmax(backend ActivationOps, dim int) *Softmax {
	return &Softmax{backend: backend, dim: dim}
}

// Process applies softmax to the input tensor.
func (s *Softmax) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.Softmax(ctx, t, s.dim)
	if err != nil {
		return nil, fmt.Errorf("softmax: %w", err)
	}

	emitWithTrace(ctx, OpSoftmax,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDim.Field(s.dim),
	)

	propagateTape(t, out, "softmax", map[string]*Tensor{"output": out})

	return out, nil
}

// Name returns the operator name.
func (s *Softmax) Name() pipz.Name { return "softmax" }

// Close releases any resources held by this operator.
func (s *Softmax) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Softmax)(nil)

// LogSoftmax is a chainable operator that applies log softmax along a dimension.
// LogSoftmax(x)_i = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j))).
type LogSoftmax struct {
	backend ActivationOps
	dim     int
}

// NewLogSoftmax creates a LogSoftmax operator.
func NewLogSoftmax(backend ActivationOps, dim int) *LogSoftmax {
	return &LogSoftmax{backend: backend, dim: dim}
}

// Process applies log softmax to the input tensor.
func (l *LogSoftmax) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := l.backend.LogSoftmax(ctx, t, l.dim)
	if err != nil {
		return nil, fmt.Errorf("logsoftmax: %w", err)
	}

	emitWithTrace(ctx, OpLogSoftmax,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyDim.Field(l.dim),
	)

	propagateTape(t, out, "logsoftmax", map[string]*Tensor{"output": out})

	return out, nil
}

// Name returns the operator name.
func (l *LogSoftmax) Name() pipz.Name { return "logsoftmax" }

// Close releases any resources held by this operator.
func (l *LogSoftmax) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*LogSoftmax)(nil)

// LeakyReLU is a chainable operator that applies leaky ReLU.
// LeakyReLU(x) = x if x > 0, else negative_slope * x.
type LeakyReLU struct {
	backend       ActivationOps
	negativeSlope float32
}

// NewLeakyReLU creates a LeakyReLU operator.
func NewLeakyReLU(backend ActivationOps, negativeSlope float32) *LeakyReLU {
	return &LeakyReLU{backend: backend, negativeSlope: negativeSlope}
}

// Process applies leaky ReLU to the input tensor.
func (l *LeakyReLU) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := l.backend.LeakyReLU(ctx, t, l.negativeSlope)
	if err != nil {
		return nil, fmt.Errorf("leakyrelu: %w", err)
	}

	emitWithTrace(ctx, OpLeakyReLU,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyScalar.Field(l.negativeSlope),
	)

	propagateTape(t, out, "leaky_relu", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (l *LeakyReLU) Name() pipz.Name { return "leaky_relu" }

// Close releases any resources held by this operator.
func (l *LeakyReLU) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*LeakyReLU)(nil)

// Dropout is a chainable operator that applies dropout during training.
// During evaluation (inference mode), this is a no-op.
// p is the probability of dropping a value (0 to 1).
// Use WithTraining(ctx) to enable dropout during training.
type Dropout struct {
	backend ActivationOps
	p       float32
}

// NewDropout creates a Dropout operator.
func NewDropout(backend ActivationOps, p float32) *Dropout {
	return &Dropout{backend: backend, p: p}
}

// Process applies dropout to the input tensor.
func (d *Dropout) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	training := IsTraining(ctx)
	out, mask, err := d.backend.Dropout(ctx, t, d.p, training)
	if err != nil {
		return nil, fmt.Errorf("dropout: %w", err)
	}

	emitWithTrace(ctx, OpDropout,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyScalar.Field(d.p),
	)

	propagateTape(t, out, "dropout", map[string]*Tensor{"mask": mask})

	return out, nil
}

// Name returns the operator name.
func (d *Dropout) Name() pipz.Name { return "dropout" }

// Close releases any resources held by this operator.
func (d *Dropout) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Dropout)(nil)
