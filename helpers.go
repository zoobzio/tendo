package tendo

import (
	"context"
	"time"

	"github.com/zoobzio/pipz"
)

// -----------------------------------------------------------------------------
// Pipeline Composition
// -----------------------------------------------------------------------------

// Sequence creates a sequential pipeline of tensor processors.
// Each processor receives the output of the previous one.
//
// Example:
//
//	pipeline := tendo.Sequence("preprocess",
//	    tendo.NewReshape(-1, 224, 224, 3),
//	    tendo.NewPermute(0, 3, 1, 2),  // NHWC -> NCHW
//	    normalize,
//	)
func Sequence(name string, processors ...pipz.Chainable[*Tensor]) *pipz.Sequence[*Tensor] {
	return pipz.NewSequence(pipz.Name(name), processors...)
}

// -----------------------------------------------------------------------------
// Control Flow
// -----------------------------------------------------------------------------

// Filter creates a conditional processor that either processes or passes through.
// When the predicate returns true, the processor is executed.
// When false, the tensor passes through unchanged.
//
// Example:
//
//	gpuOnly := tendo.Filter("gpu-only",
//	    func(ctx context.Context, t *tendo.Tensor) bool {
//	        return t.Device().Type == tendo.CUDA
//	    },
//	    gpuProcessor,
//	)
func Filter(name string, predicate func(context.Context, *Tensor) bool, processor pipz.Chainable[*Tensor]) *pipz.Filter[*Tensor] {
	return pipz.NewFilter(pipz.Name(name), predicate, processor)
}

// Switch creates a router that directs tensors to different processors.
// The condition function returns a route key that determines which processor handles the tensor.
//
// Example:
//
//	router := tendo.Switch("device-router", func(ctx context.Context, t *tendo.Tensor) tendo.DeviceType {
//	    return t.Device().Type
//	})
//	router.AddRoute(tendo.CPU, cpuProcessor)
//	router.AddRoute(tendo.CUDA, cudaProcessor)
func Switch[K comparable](name string, condition func(context.Context, *Tensor) K) *pipz.Switch[*Tensor, K] {
	return pipz.NewSwitch(pipz.Name(name), condition)
}

// -----------------------------------------------------------------------------
// Error Recovery
// -----------------------------------------------------------------------------

// Fallback creates a processor that tries alternatives on failure.
// Each processor is tried in order until one succeeds.
// Useful for GPU to CPU fallback.
//
// Example:
//
//	resilient := tendo.Fallback("matmul",
//	    cudaMatMul,   // Try GPU first
//	    cpuMatMul,    // Fall back to CPU
//	)
func Fallback(name string, processors ...pipz.Chainable[*Tensor]) *pipz.Fallback[*Tensor] {
	return pipz.NewFallback(pipz.Name(name), processors...)
}

// Timeout creates a processor that enforces a time limit on execution.
// If the timeout expires, the operation is canceled and an error is returned.
// Useful for SLA enforcement in inference services.
//
// Example:
//
//	bounded := tendo.Timeout("inference", model, 100*time.Millisecond)
func Timeout(name string, processor pipz.Chainable[*Tensor], duration time.Duration) *pipz.Timeout[*Tensor] {
	return pipz.NewTimeout(pipz.Name(name), processor, duration)
}
