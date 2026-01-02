package tendo

import (
	"context"
	"fmt"
	"math"
)

// AdamWConfig contains hyperparameters for the AdamW optimizer.
type AdamWConfig struct {
	LR          float32 // Learning rate (default: 1e-3)
	Beta1       float32 // First moment decay (default: 0.9)
	Beta2       float32 // Second moment decay (default: 0.999)
	Epsilon     float32 // Numerical stability (default: 1e-8)
	WeightDecay float32 // Decoupled weight decay (default: 0.01)
}

// DefaultAdamWConfig returns default AdamW hyperparameters.
func DefaultAdamWConfig() AdamWConfig {
	return AdamWConfig{
		LR:          1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Epsilon:     1e-8,
		WeightDecay: 0.01,
	}
}

// paramState holds the optimizer state for a single parameter tensor.
type paramState struct {
	m *Tensor // First moment (mean of gradients)
	v *Tensor // Second moment (mean of squared gradients)
}

// AdamW implements the AdamW optimizer with decoupled weight decay.
type AdamW struct { //nolint:govet // field alignment is less important than readability
	backend OptimizerOps
	config  AdamWConfig
	step    int
	states  map[*Tensor]*paramState
}

// NewAdamW creates a new AdamW optimizer.
// If backend is nil, falls back to a composed CPU implementation.
func NewAdamW(backend OptimizerOps, config AdamWConfig) *AdamW {
	return &AdamW{
		backend: backend,
		config:  config,
		step:    0,
		states:  make(map[*Tensor]*paramState),
	}
}

// Step performs one optimization step.
// params and grads must have the same length and corresponding shapes.
func (o *AdamW) Step(ctx context.Context, params, grads []*Tensor) error {
	if len(params) != len(grads) {
		return fmt.Errorf("adamw: params and grads length mismatch: %d vs %d", len(params), len(grads))
	}

	o.step++

	// Compute bias corrections
	biasCorrection1 := 1.0 - float32(math.Pow(float64(o.config.Beta1), float64(o.step)))
	biasCorrection2 := 1.0 - float32(math.Pow(float64(o.config.Beta2), float64(o.step)))

	for i, param := range params {
		grad := grads[i]

		// Get or create state for this parameter
		state, ok := o.states[param]
		if !ok {
			// Initialize state tensors (zeros with same shape as param)
			m, err := o.zerosLike(param)
			if err != nil {
				return fmt.Errorf("adamw: create m state: %w", err)
			}
			v, err := o.zerosLike(param)
			if err != nil {
				m.Free()
				return fmt.Errorf("adamw: create v state: %w", err)
			}
			state = &paramState{m: m, v: v}
			o.states[param] = state
		}

		// Perform the update
		if o.backend != nil {
			// Use fused kernel
			if err := o.backend.AdamW(ctx, param, grad, state.m, state.v,
				o.config.LR, o.config.Beta1, o.config.Beta2,
				o.config.Epsilon, o.config.WeightDecay,
				biasCorrection1, biasCorrection2); err != nil {
				return fmt.Errorf("adamw: step %d param %d: %w", o.step, i, err)
			}
		} else {
			// CPU fallback - composed operations
			if err := o.cpuStep(param, grad, state, biasCorrection1, biasCorrection2); err != nil {
				return fmt.Errorf("adamw: cpu step %d param %d: %w", o.step, i, err)
			}
		}
	}

	return nil
}

// ZeroGrad is a placeholder for zeroing gradients.
// In tendo, gradients are typically computed fresh each forward pass,
// but this method is provided for API compatibility.
func (o *AdamW) ZeroGrad() {
	// No-op in current design
}

// StepCount returns the current optimization step number.
func (o *AdamW) StepCount() int {
	return o.step
}

// Config returns the optimizer configuration.
func (o *AdamW) Config() AdamWConfig {
	return o.config
}

// SetLR updates the learning rate.
func (o *AdamW) SetLR(lr float32) {
	o.config.LR = lr
}

// Free releases optimizer state tensors.
func (o *AdamW) Free() {
	for _, state := range o.states {
		if state.m != nil {
			state.m.Free()
		}
		if state.v != nil {
			state.v.Free()
		}
	}
	o.states = make(map[*Tensor]*paramState)
}

// zerosLike creates a zero-filled tensor with the same shape and device as t.
func (o *AdamW) zerosLike(t *Tensor) (*Tensor, error) {
	numel := t.Numel()
	storage := NewCPUStorage(numel, t.DType())
	storage.Fill(0)
	return NewTensor(storage, t.Shape(), nil), nil
}

// cpuStep performs an AdamW step using composed CPU operations.
func (o *AdamW) cpuStep(param, grad *Tensor, state *paramState, bc1, bc2 float32) error {
	// Get data accessors
	paramData, ok := param.Storage().(CPUDataAccessor)
	if !ok {
		return fmt.Errorf("param is not CPU storage")
	}
	gradData, ok := grad.Storage().(CPUDataAccessor)
	if !ok {
		return fmt.Errorf("grad is not CPU storage")
	}
	mData, ok := state.m.Storage().(CPUDataAccessor)
	if !ok {
		return fmt.Errorf("m is not CPU storage")
	}
	vData, ok := state.v.Storage().(CPUDataAccessor)
	if !ok {
		return fmt.Errorf("v is not CPU storage")
	}

	pSlice := paramData.Data()
	gSlice := gradData.Data()
	mSlice := mData.Data()
	vSlice := vData.Data()

	beta1 := o.config.Beta1
	beta2 := o.config.Beta2
	lr := o.config.LR
	eps := o.config.Epsilon
	wd := o.config.WeightDecay

	for i := range pSlice {
		g := gSlice[i]
		p := pSlice[i]

		// Update moments
		mSlice[i] = beta1*mSlice[i] + (1-beta1)*g
		vSlice[i] = beta2*vSlice[i] + (1-beta2)*g*g

		// Bias-corrected estimates
		mHat := mSlice[i] / bc1
		vHat := vSlice[i] / bc2

		// Update parameter with decoupled weight decay
		pSlice[i] = p - lr*(mHat/(float32(math.Sqrt(float64(vHat)))+eps)+wd*p)
	}

	return nil
}
