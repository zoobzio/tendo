package tendo

import (
	"context"

	"github.com/zoobzio/capitan"
	"github.com/zoobzio/pipz"
)

// Training mode context key and helpers

type trainingModeKey struct{}

// WithTraining returns a context with training mode enabled.
// Operations like Dropout will apply their training behavior.
func WithTraining(ctx context.Context) context.Context {
	return context.WithValue(ctx, trainingModeKey{}, true)
}

// WithInference returns a context with training mode disabled (inference mode).
// This is the default behavior.
func WithInference(ctx context.Context) context.Context {
	return context.WithValue(ctx, trainingModeKey{}, false)
}

// IsTraining returns true if the context has training mode enabled.
func IsTraining(ctx context.Context) bool {
	if v, ok := ctx.Value(trainingModeKey{}).(bool); ok {
		return v
	}
	return false // default to inference mode
}

// Memory pool context key and helpers

type poolKey struct{}

// WithPool returns a context with the given memory pool.
// Operations using this context will allocate from the pool.
func WithPool(ctx context.Context, p *Pool) context.Context {
	return context.WithValue(ctx, poolKey{}, p)
}

// PoolFromContext returns the memory pool from the context.
// Returns the default pool if no pool is set in the context.
func PoolFromContext(ctx context.Context) *Pool {
	if p, ok := ctx.Value(poolKey{}).(*Pool); ok && p != nil {
		return p
	}
	return DefaultPool()
}

// AllocCPUFromContext allocates CPU storage from the pool in the context.
func AllocCPUFromContext(ctx context.Context, numel int, dtype DType) *CPUStorage {
	return PoolFromContext(ctx).AllocCPU(numel, dtype)
}

// NewTensorFromPool creates a tensor using storage from the specified pool.
func NewTensorFromPool(pool *Pool, shape []int, dtype DType) *Tensor {
	numel := Numel(shape)
	storage := pool.AllocCPU(numel, dtype)
	return &Tensor{
		storage: storage,
		shape:   shape,
		stride:  ComputeStrides(shape),
		offset:  0,
		pool:    pool,
	}
}

// NewTensorFromContext creates a tensor using storage from the context's pool.
func NewTensorFromContext(ctx context.Context, shape []int, dtype DType) *Tensor {
	return NewTensorFromPool(PoolFromContext(ctx), shape, dtype)
}

// Trace ID context key and helpers for autograd graph correlation

type traceIDKey struct{}

// WithTraceID returns a context with the given trace ID.
// All operations executed with this context will include the trace ID
// in their emitted events, enabling autograd to correlate operations
// into a single computational graph.
func WithTraceID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, traceIDKey{}, id)
}

// TraceID returns the trace ID from the context, or empty string if not set.
func TraceID(ctx context.Context) string {
	if v, ok := ctx.Value(traceIDKey{}).(string); ok {
		return v
	}
	return ""
}

// emitWithTrace wraps capitan.Emit to automatically include the trace ID
// when present in the context. This ensures all operation events can be
// correlated into a single computational graph for autograd.
func emitWithTrace(ctx context.Context, signal capitan.Signal, fields ...capitan.Field) {
	if id := TraceID(ctx); id != "" {
		fields = append(fields, KeyTraceID.Field(id))
	}
	capitan.Emit(ctx, signal, fields...)
}

// propagateTape copies tape from input to output and records an operation if tape exists.
// This should be called at the end of each op's Process method.
// The saved map contains tensors needed for the backward pass (e.g., "input", "output").
func propagateTape(input, output *Tensor, opName string, saved map[string]*Tensor) {
	// Propagate requiresGrad from input to output
	if input.requiresGrad {
		output.requiresGrad = true
	}

	// Only record to tape if input requires grad and has a tape
	if input.requiresGrad && input.tape != nil {
		output.tape = input.tape
		output.tape.Record(opName, saved)
	}
}

// Op creates a named Chainable operation that can fail.
// This is a convenience wrapper around pipz.Apply.
func Op(name pipz.Name, fn func(context.Context, *Tensor) (*Tensor, error)) pipz.Chainable[*Tensor] {
	return pipz.Apply(name, fn)
}

// Transform creates a named Chainable operation that cannot fail.
// This is a convenience wrapper around pipz.Transform.
func Transform(name pipz.Name, fn func(context.Context, *Tensor) *Tensor) pipz.Chainable[*Tensor] {
	return pipz.Transform(name, fn)
}

// UnaryOp creates a unary tensor operation with the given name, signal, and compute function.
// The compute function operates on CPU storage directly.
func UnaryOp(name pipz.Name, fn func(*CPUStorage) *CPUStorage) pipz.Chainable[*Tensor] {
	return pipz.Apply(name, func(ctx context.Context, t *Tensor) (*Tensor, error) {
		// Get CPU storage (for now, only CPU is supported)
		cpu, ok := t.storage.(*CPUStorage)
		if !ok {
			return nil, &DeviceError{Expected: CPU, Got: t.Device().Type}
		}

		// Apply the operation
		result := fn(cpu)

		// Create output tensor with same shape
		out := NewTensor(result, t.Shape(), nil)

		return out, nil
	})
}

// UnaryOpGeneric creates a unary tensor operation that works with any CPU storage.
func UnaryOpGeneric(name pipz.Name, fn func(CPUDataAccessor) *CPUStorage) pipz.Chainable[*Tensor] {
	return pipz.Apply(name, func(ctx context.Context, t *Tensor) (*Tensor, error) {
		cpu, ok := t.storage.(CPUDataAccessor)
		if !ok {
			return nil, &DeviceError{Expected: CPU, Got: t.Device().Type}
		}

		result := fn(cpu)
		out := NewTensor(result, t.Shape(), nil)

		return out, nil
	})
}

// UnaryOpInPlace creates an in-place unary tensor operation.
// The input tensor is modified and returned.
func UnaryOpInPlace(name pipz.Name, fn func(*CPUStorage)) pipz.Chainable[*Tensor] {
	return pipz.Transform(name, func(_ context.Context, t *Tensor) *Tensor {
		if cpu, ok := t.storage.(*CPUStorage); ok {
			fn(cpu)
		}
		return t
	})
}

// BinaryOp creates a binary tensor operation with the given name.
// The second operand is captured in the closure.
func BinaryOp(name pipz.Name, other *Tensor, fn func(*CPUStorage, *CPUStorage, []int) *CPUStorage) pipz.Chainable[*Tensor] {
	return pipz.Apply(name, func(ctx context.Context, t *Tensor) (*Tensor, error) {
		// Validate devices match
		if t.Device() != other.Device() {
			return nil, &DeviceMismatchError{A: t.Device(), B: other.Device()}
		}

		// Get CPU storage
		cpuA, ok := t.storage.(*CPUStorage)
		if !ok {
			return nil, &DeviceError{Expected: CPU, Got: t.Device().Type}
		}
		cpuB, ok := other.storage.(*CPUStorage)
		if !ok {
			return nil, &DeviceError{Expected: CPU, Got: other.Device().Type}
		}

		// Compute broadcast shape
		outShape, err := BroadcastShapes(t.Shape(), other.Shape())
		if err != nil {
			return nil, err
		}

		// Apply the operation
		result := fn(cpuA, cpuB, outShape)

		// Create output tensor
		out := NewTensor(result, outShape, nil)

		return out, nil
	})
}

// DeviceError indicates an operation was attempted on an unsupported device.
type DeviceError struct {
	Expected DeviceType
	Got      DeviceType
}

func (e *DeviceError) Error() string {
	return "expected device " + e.Expected.String() + ", got " + e.Got.String()
}

// Is reports whether target is a DeviceError.
func (e *DeviceError) Is(target error) bool {
	_, ok := target.(*DeviceError)
	return ok
}

// DeviceMismatchError indicates two tensors are on different devices.
type DeviceMismatchError struct {
	A Device
	B Device
}

func (e *DeviceMismatchError) Error() string {
	return "device mismatch: " + e.A.String() + " vs " + e.B.String()
}

// Is reports whether target is a DeviceMismatchError.
func (e *DeviceMismatchError) Is(target error) bool {
	_, ok := target.(*DeviceMismatchError)
	return ok
}

// ShapeError indicates a shape-related error.
type ShapeError struct {
	Op      string
	Message string
	ShapeA  []int
	ShapeB  []int
}

func (e *ShapeError) Error() string {
	return e.Op + ": " + e.Message
}

// Is reports whether target is a ShapeError.
func (e *ShapeError) Is(target error) bool {
	_, ok := target.(*ShapeError)
	return ok
}
