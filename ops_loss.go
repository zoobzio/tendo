package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// Reduction modes for loss functions.
const (
	ReductionNone = "none" // No reduction, return element-wise loss
	ReductionMean = "mean" // Return mean of losses
	ReductionSum  = "sum"  // Return sum of losses
)

// MSELoss is a chainable operator that computes mean squared error loss.
// MSE = (1/n) * sum((input - target)^2) for reduction="mean".
type MSELoss struct {
	backend   LossOps
	target    *Tensor
	reduction string
}

// NewMSELoss creates an MSELoss operator.
func NewMSELoss(backend LossOps, target *Tensor, reduction string) *MSELoss {
	return &MSELoss{backend: backend, target: target, reduction: reduction}
}

// Process computes MSE loss.
func (m *MSELoss) Process(ctx context.Context, input *Tensor) (*Tensor, error) {
	result, err := m.backend.MSELoss(ctx, input, m.target, m.reduction)
	if err != nil {
		return nil, fmt.Errorf("mseloss: %w", err)
	}

	emitWithTrace(ctx, OpMSELoss,
		KeyInput.Field(input),
		KeyTarget.Field(m.target),
		KeyOutput.Field(result),
		KeyReduction.Field(m.reduction),
	)

	propagateTape(input, result, "mseloss", map[string]*Tensor{"input": input, "target": m.target})

	return result, nil
}

// Name returns the operator name.
func (m *MSELoss) Name() pipz.Name { return "mse_loss" }

// Close releases any resources held by this operator.
func (m *MSELoss) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*MSELoss)(nil)

// L1Loss is a chainable operator that computes L1 (mean absolute error) loss.
// L1 = (1/n) * sum(|input - target|) for reduction="mean".
type L1Loss struct {
	backend   LossOps
	target    *Tensor
	reduction string
}

// NewL1Loss creates an L1Loss operator.
func NewL1Loss(backend LossOps, target *Tensor, reduction string) *L1Loss {
	return &L1Loss{backend: backend, target: target, reduction: reduction}
}

// Process computes L1 loss.
func (l *L1Loss) Process(ctx context.Context, input *Tensor) (*Tensor, error) {
	result, err := l.backend.L1Loss(ctx, input, l.target, l.reduction)
	if err != nil {
		return nil, fmt.Errorf("l1loss: %w", err)
	}

	emitWithTrace(ctx, OpL1Loss,
		KeyInput.Field(input),
		KeyTarget.Field(l.target),
		KeyOutput.Field(result),
		KeyReduction.Field(l.reduction),
	)

	propagateTape(input, result, "l1loss", map[string]*Tensor{"input": input, "target": l.target})

	return result, nil
}

// Name returns the operator name.
func (l *L1Loss) Name() pipz.Name { return "l1_loss" }

// Close releases any resources held by this operator.
func (l *L1Loss) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*L1Loss)(nil)

// CrossEntropyLoss is a chainable operator that computes cross-entropy loss.
// Input: [N, C] logits (not softmaxed)
// Target: [N] class indices (integers as float32)
// Loss = -log(softmax(input)[target]).
type CrossEntropyLoss struct {
	backend   LossOps
	target    *Tensor
	reduction string
}

// NewCrossEntropyLoss creates a CrossEntropyLoss operator.
func NewCrossEntropyLoss(backend LossOps, target *Tensor, reduction string) *CrossEntropyLoss {
	return &CrossEntropyLoss{backend: backend, target: target, reduction: reduction}
}

// Process computes cross-entropy loss.
func (c *CrossEntropyLoss) Process(ctx context.Context, input *Tensor) (*Tensor, error) {
	result, err := c.backend.CrossEntropyLoss(ctx, input, c.target, c.reduction)
	if err != nil {
		return nil, fmt.Errorf("crossentropyloss: %w", err)
	}

	emitWithTrace(ctx, OpCrossEntropyLoss,
		KeyInput.Field(input),
		KeyTarget.Field(c.target),
		KeyOutput.Field(result),
		KeyReduction.Field(c.reduction),
	)

	propagateTape(input, result, "crossentropyloss", map[string]*Tensor{"input": input, "target": c.target})

	return result, nil
}

// Name returns the operator name.
func (c *CrossEntropyLoss) Name() pipz.Name { return "cross_entropy_loss" }

// Close releases any resources held by this operator.
func (c *CrossEntropyLoss) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*CrossEntropyLoss)(nil)

// NLLLoss is a chainable operator that computes negative log likelihood loss.
// Input: [N, C] log-probabilities (already log-softmaxed)
// Target: [N] class indices (integers as float32)
// Loss = -input[target].
type NLLLoss struct {
	backend   LossOps
	target    *Tensor
	reduction string
}

// NewNLLLoss creates an NLLLoss operator.
func NewNLLLoss(backend LossOps, target *Tensor, reduction string) *NLLLoss {
	return &NLLLoss{backend: backend, target: target, reduction: reduction}
}

// Process computes NLL loss.
func (n *NLLLoss) Process(ctx context.Context, input *Tensor) (*Tensor, error) {
	result, err := n.backend.NLLLoss(ctx, input, n.target, n.reduction)
	if err != nil {
		return nil, fmt.Errorf("nllloss: %w", err)
	}

	emitWithTrace(ctx, OpNLLLoss,
		KeyInput.Field(input),
		KeyTarget.Field(n.target),
		KeyOutput.Field(result),
		KeyReduction.Field(n.reduction),
	)

	propagateTape(input, result, "nllloss", map[string]*Tensor{"input": input, "target": n.target})

	return result, nil
}

// Name returns the operator name.
func (n *NLLLoss) Name() pipz.Name { return "nll_loss" }

// Close releases any resources held by this operator.
func (n *NLLLoss) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*NLLLoss)(nil)
