package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// ReLU is a chainable operator that applies the ReLU activation function.
// ReLU(x) = max(0, x).
type ReLU struct {
	backend  UnaryOps
	identity pipz.Identity
}

// NewReLU creates a ReLU operator.
func NewReLU(backend UnaryOps) *ReLU {
	return &ReLU{
		backend:  backend,
		identity: IdentityReLU,
	}
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

// Identity returns the operator identity.
func (r *ReLU) Identity() pipz.Identity { return r.identity }

// Schema returns the operator schema.
func (r *ReLU) Schema() pipz.Node {
	return pipz.Node{Identity: r.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (r *ReLU) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*ReLU)(nil)

// Sigmoid is a chainable operator that applies the sigmoid activation function.
// Sigmoid(x) = 1 / (1 + exp(-x)).
type Sigmoid struct {
	backend  UnaryOps
	identity pipz.Identity
}

// NewSigmoid creates a Sigmoid operator.
func NewSigmoid(backend UnaryOps) *Sigmoid {
	return &Sigmoid{
		backend:  backend,
		identity: IdentitySigmoid,
	}
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

// Identity returns the operator identity.
func (s *Sigmoid) Identity() pipz.Identity { return s.identity }

// Schema returns the operator schema.
func (s *Sigmoid) Schema() pipz.Node {
	return pipz.Node{Identity: s.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (s *Sigmoid) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Sigmoid)(nil)

// Tanh is a chainable operator that applies the tanh activation function.
type Tanh struct {
	backend  UnaryOps
	identity pipz.Identity
}

// NewTanh creates a Tanh operator.
func NewTanh(backend UnaryOps) *Tanh {
	return &Tanh{
		backend:  backend,
		identity: IdentityTanh,
	}
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

// Identity returns the operator identity.
func (t *Tanh) Identity() pipz.Identity { return t.identity }

// Schema returns the operator schema.
func (t *Tanh) Schema() pipz.Node {
	return pipz.Node{Identity: t.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (t *Tanh) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Tanh)(nil)

// GELU is a chainable operator that applies the GELU activation function.
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
type GELU struct {
	backend  UnaryOps
	identity pipz.Identity
}

// NewGELU creates a GELU operator.
func NewGELU(backend UnaryOps) *GELU {
	return &GELU{
		backend:  backend,
		identity: IdentityGELU,
	}
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

// Identity returns the operator identity.
func (g *GELU) Identity() pipz.Identity { return g.identity }

// Schema returns the operator schema.
func (g *GELU) Schema() pipz.Node {
	return pipz.Node{Identity: g.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (g *GELU) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*GELU)(nil)

// SiLU is a chainable operator that applies the SiLU (Swish) activation.
// SiLU(x) = x * sigmoid(x).
type SiLU struct {
	backend  UnaryOps
	identity pipz.Identity
}

// NewSiLU creates a SiLU operator.
func NewSiLU(backend UnaryOps) *SiLU {
	return &SiLU{
		backend:  backend,
		identity: IdentitySiLU,
	}
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

// Identity returns the operator identity.
func (s *SiLU) Identity() pipz.Identity { return s.identity }

// Schema returns the operator schema.
func (s *SiLU) Schema() pipz.Node {
	return pipz.Node{Identity: s.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (s *SiLU) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*SiLU)(nil)

// Softmax is a chainable operator that applies softmax along a dimension.
// Softmax(x)_i = exp(x_i) / sum(exp(x_j)).
type Softmax struct {
	backend  ActivationOps
	identity pipz.Identity
	dim      int
}

// NewSoftmax creates a Softmax operator.
func NewSoftmax(backend ActivationOps, dim int) *Softmax {
	return &Softmax{
		backend:  backend,
		identity: IdentitySoftmax,
		dim:      dim,
	}
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

// Identity returns the operator identity.
func (s *Softmax) Identity() pipz.Identity { return s.identity }

// Schema returns the operator schema.
func (s *Softmax) Schema() pipz.Node {
	return pipz.Node{Identity: s.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (s *Softmax) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Softmax)(nil)

// LogSoftmax is a chainable operator that applies log softmax along a dimension.
// LogSoftmax(x)_i = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j))).
type LogSoftmax struct {
	backend  ActivationOps
	identity pipz.Identity
	dim      int
}

// NewLogSoftmax creates a LogSoftmax operator.
func NewLogSoftmax(backend ActivationOps, dim int) *LogSoftmax {
	return &LogSoftmax{
		backend:  backend,
		identity: IdentityLogSoftmax,
		dim:      dim,
	}
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

// Identity returns the operator identity.
func (l *LogSoftmax) Identity() pipz.Identity { return l.identity }

// Schema returns the operator schema.
func (l *LogSoftmax) Schema() pipz.Node {
	return pipz.Node{Identity: l.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (l *LogSoftmax) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*LogSoftmax)(nil)

// LeakyReLU is a chainable operator that applies leaky ReLU.
// LeakyReLU(x) = x if x > 0, else negative_slope * x.
type LeakyReLU struct {
	backend       ActivationOps
	identity      pipz.Identity
	negativeSlope float32
}

// NewLeakyReLU creates a LeakyReLU operator.
func NewLeakyReLU(backend ActivationOps, negativeSlope float32) *LeakyReLU {
	return &LeakyReLU{
		backend:       backend,
		identity:      IdentityLeakyReLU,
		negativeSlope: negativeSlope,
	}
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

// Identity returns the operator identity.
func (l *LeakyReLU) Identity() pipz.Identity { return l.identity }

// Schema returns the operator schema.
func (l *LeakyReLU) Schema() pipz.Node {
	return pipz.Node{Identity: l.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (l *LeakyReLU) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*LeakyReLU)(nil)

// Dropout is a chainable operator that applies dropout during training.
// During evaluation (inference mode), this is a no-op.
// p is the probability of dropping a value (0 to 1).
// Use WithTraining(ctx) to enable dropout during training.
type Dropout struct {
	backend  ActivationOps
	identity pipz.Identity
	p        float32
}

// NewDropout creates a Dropout operator.
func NewDropout(backend ActivationOps, p float32) *Dropout {
	return &Dropout{
		backend:  backend,
		identity: IdentityDropout,
		p:        p,
	}
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

// Identity returns the operator identity.
func (d *Dropout) Identity() pipz.Identity { return d.identity }

// Schema returns the operator schema.
func (d *Dropout) Schema() pipz.Node {
	return pipz.Node{Identity: d.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (d *Dropout) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Dropout)(nil)
