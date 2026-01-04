package nn

import (
	"context"
	"fmt"

	"github.com/zoobzio/tendo"
)

// Activation represents an activation function type.
type Activation int

// Supported activation functions.
const (
	GELU Activation = iota
	SiLU
	ReLU
)

// MLPBackend defines the operations needed for an MLP.
type MLPBackend interface {
	LinearBackend
	GELU(ctx context.Context, t *tendo.Tensor) (*tendo.Tensor, error)
	SiLU(ctx context.Context, t *tendo.Tensor) (*tendo.Tensor, error)
	ReLU(ctx context.Context, t *tendo.Tensor) (*tendo.Tensor, error)
	Mul(ctx context.Context, a, b *tendo.Tensor) (*tendo.Tensor, error)
}

// MLP represents a feed-forward network.
// Standard: up -> activation -> down
// Gated (SwiGLU style): (gate * up) -> down.
type MLP struct {
	UpProj   *Linear // [dim, hidden_dim]
	DownProj *Linear // [hidden_dim, dim]
	GateProj *Linear // [dim, hidden_dim], optional for gated MLPs

	Activation Activation
	Gated      bool
}

// MLPConfig configures an MLP layer.
type MLPConfig struct {
	Dim        int        // model dimension
	HiddenDim  int        // intermediate dimension (usually 4x dim)
	Activation Activation // activation function
	Gated      bool       // use gated activation (SwiGLU style)
	Bias       bool       // whether projections have bias
}

// NewMLP creates an MLP from weight tensors.
// For standard MLP: upWeight [hidden_dim, dim], downWeight [dim, hidden_dim]
// For gated MLP: additionally gateWeight [hidden_dim, dim].
func NewMLP(cfg MLPConfig, upWeight, downWeight, gateWeight *tendo.Tensor, upBias, downBias, gateBias *tendo.Tensor) (*MLP, error) {
	upProj, err := NewLinear(upWeight, upBias)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: up projection: %w", err)
	}

	downProj, err := NewLinear(downWeight, downBias)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: down projection: %w", err)
	}

	var gateProj *Linear
	if cfg.Gated {
		if gateWeight == nil {
			return nil, fmt.Errorf("nn.MLP: gated MLP requires gate weight")
		}
		gateProj, err = NewLinear(gateWeight, gateBias)
		if err != nil {
			return nil, fmt.Errorf("nn.MLP: gate projection: %w", err)
		}
	}

	return &MLP{
		UpProj:     upProj,
		DownProj:   downProj,
		GateProj:   gateProj,
		Activation: cfg.Activation,
		Gated:      cfg.Gated,
	}, nil
}

// Forward computes the MLP output.
// Standard: down(activation(up(x)))
// Gated: down(activation(gate(x)) * up(x)).
func (m *MLP) Forward(ctx context.Context, x *tendo.Tensor, backend MLPBackend) (*tendo.Tensor, error) {
	if m.Gated {
		return m.forwardGated(ctx, x, backend)
	}
	return m.forwardStandard(ctx, x, backend)
}

func (m *MLP) forwardStandard(ctx context.Context, x *tendo.Tensor, backend MLPBackend) (*tendo.Tensor, error) {
	// up projection
	h, err := m.UpProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: up projection: %w", err)
	}

	// activation
	h, err = m.applyActivation(ctx, h, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: activation: %w", err)
	}

	// down projection
	out, err := m.DownProj.Forward(ctx, h, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: down projection: %w", err)
	}

	return out, nil
}

func (m *MLP) forwardGated(ctx context.Context, x *tendo.Tensor, backend MLPBackend) (*tendo.Tensor, error) {
	// gate projection with activation
	gate, err := m.GateProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: gate projection: %w", err)
	}
	gate, err = m.applyActivation(ctx, gate, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: gate activation: %w", err)
	}

	// up projection (no activation)
	up, err := m.UpProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: up projection: %w", err)
	}

	// element-wise multiply: gate * up
	h, err := backend.Mul(ctx, gate, up)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: gate multiply: %w", err)
	}

	// down projection
	out, err := m.DownProj.Forward(ctx, h, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.MLP: down projection: %w", err)
	}

	return out, nil
}

func (m *MLP) applyActivation(ctx context.Context, t *tendo.Tensor, backend MLPBackend) (*tendo.Tensor, error) {
	switch m.Activation {
	case GELU:
		return backend.GELU(ctx, t)
	case SiLU:
		return backend.SiLU(ctx, t)
	case ReLU:
		return backend.ReLU(ctx, t)
	default:
		return nil, fmt.Errorf("unknown activation: %d", m.Activation)
	}
}
