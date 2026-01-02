package tendo

import (
	"fmt"
)

// Tensor represents a multi-dimensional array of numeric data.
// It wraps a Storage backend and maintains shape/stride metadata
// for interpreting the underlying data.
type Tensor struct {
	storage      Storage
	pool         *Pool
	tape         *Tape    // execution history for autograd, nil during inference
	grad         *Tensor  // accumulated gradient, nil if not computed
	shape        []int
	stride       []int
	offset       int
	requiresGrad bool // whether this tensor participates in autograd
}

// NewTensor creates a tensor with the given storage, shape, and stride.
// If stride is nil, a contiguous stride is computed from shape.
func NewTensor(storage Storage, shape []int, stride []int) *Tensor {
	if stride == nil {
		stride = ComputeStrides(shape)
	}
	return &Tensor{
		storage: storage,
		shape:   shape,
		stride:  stride,
		offset:  0,
	}
}

// Shape returns the dimensions of the tensor.
func (t *Tensor) Shape() []int {
	result := make([]int, len(t.shape))
	copy(result, t.shape)
	return result
}

// Stride returns the strides of the tensor.
func (t *Tensor) Stride() []int {
	result := make([]int, len(t.stride))
	copy(result, t.stride)
	return result
}

// Dim returns the number of dimensions.
func (t *Tensor) Dim() int {
	return len(t.shape)
}

// Size returns the size of the given dimension.
func (t *Tensor) Size(dim int) int {
	if dim < 0 {
		dim = len(t.shape) + dim
	}
	if dim < 0 || dim >= len(t.shape) {
		return 0
	}
	return t.shape[dim]
}

// Numel returns the total number of elements in the tensor.
func (t *Tensor) Numel() int {
	return Numel(t.shape)
}

// Device returns the device where this tensor resides.
func (t *Tensor) Device() Device {
	return t.storage.Device()
}

// DType returns the data type of the tensor elements.
func (t *Tensor) DType() DType {
	return t.storage.DType()
}

// Storage returns the underlying storage.
func (t *Tensor) Storage() Storage {
	return t.storage
}

// Offset returns the offset into storage.
func (t *Tensor) Offset() int {
	return t.offset
}

// IsContiguous returns true if the tensor memory is contiguous.
func (t *Tensor) IsContiguous() bool {
	return IsContiguous(t.shape, t.stride)
}

// Clone creates a deep copy of the tensor.
// For non-contiguous or offset tensors, clones only the logical data
// and produces a contiguous output with offset 0.
// Implements pipz.Cloner[*Tensor].
func (t *Tensor) Clone() *Tensor {
	// For non-contiguous or offset tensors, clone only the logical data
	if !t.IsContiguous() || t.offset != 0 {
		contig := t.Contiguous()
		return contig.Clone()
	}

	newStorage := t.storage.Clone()
	newShape := make([]int, len(t.shape))
	copy(newShape, t.shape)
	newStride := make([]int, len(t.stride))
	copy(newStride, t.stride)

	return &Tensor{
		storage:      newStorage,
		shape:        newShape,
		stride:       newStride,
		offset:       0,
		requiresGrad: t.requiresGrad,
	}
}

// Free releases the underlying storage.
// If the tensor was allocated from a pool, storage is returned to that pool.
func (t *Tensor) Free() {
	if t.storage == nil {
		return
	}

	// Return to pool if one is set
	if t.pool != nil {
		switch s := t.storage.(type) {
		case *CPUStorage:
			t.pool.FreeCPU(s)
			t.storage = nil
			return
		case PoolableStorage:
			if s.Device().IsCUDA() {
				numel, dtype, idx := s.PoolKey()
				t.pool.FreeCUDA(s.Ptr(), numel, dtype, idx)
				t.storage = nil
				return
			}
		}
	}

	// Otherwise just free directly
	t.storage.Free()
	t.storage = nil
}

// Pool returns the memory pool associated with this tensor, or nil if none.
func (t *Tensor) Pool() *Pool {
	return t.pool
}

// SetPool sets the memory pool for this tensor.
// The pool will be used when Free() is called.
func (t *Tensor) SetPool(p *Pool) {
	t.pool = p
}

// Tape returns the tape attached to this tensor for autograd, or nil.
func (t *Tensor) Tape() *Tape {
	return t.tape
}

// SetTape attaches a tape to this tensor for recording operations.
func (t *Tensor) SetTape(tape *Tape) {
	t.tape = tape
}

// WithTape attaches a tape to this tensor and returns the tensor.
// Useful for chaining: input.WithTape(tape).
func (t *Tensor) WithTape(tape *Tape) *Tensor {
	t.tape = tape
	return t
}

// RequiresGrad returns whether this tensor participates in autograd.
func (t *Tensor) RequiresGrad() bool {
	return t.requiresGrad
}

// SetRequiresGrad sets whether this tensor participates in autograd.
// When enabled, operations on this tensor will be recorded to the tape.
func (t *Tensor) SetRequiresGrad(requires bool) {
	t.requiresGrad = requires
}

// WithRequiresGrad sets requiresGrad and returns the tensor for chaining.
// Example: weight := backend.RandN(768, 768).WithRequiresGrad(true).
func (t *Tensor) WithRequiresGrad(requires bool) *Tensor {
	t.requiresGrad = requires
	return t
}

// Grad returns the accumulated gradient for this tensor, or nil if none.
func (t *Tensor) Grad() *Tensor {
	return t.grad
}

// SetGrad sets the gradient tensor.
func (t *Tensor) SetGrad(grad *Tensor) {
	t.grad = grad
}

// ZeroGrad clears the accumulated gradient.
// Should be called before each training step to prevent gradient accumulation.
func (t *Tensor) ZeroGrad() {
	if t.grad != nil {
		t.grad.Free()
		t.grad = nil
	}
}

// AccumulateGrad adds gradients to the existing gradient tensor.
// If no gradient exists, sets the gradient directly.
// This is used during backward pass when a tensor is used multiple times.
func (t *Tensor) AccumulateGrad(grad *Tensor) error {
	if t.grad == nil {
		t.grad = grad.Clone()
		return nil
	}
	// TODO: Add in-place when we have an Add that modifies in place
	// For now, we'd need a backend reference to do the addition
	// This will be implemented when we add the backward pass
	return fmt.Errorf("gradient accumulation not yet implemented")
}

// String returns a string representation of the tensor metadata.
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, stride=%v, dtype=%s, device=%s)",
		t.shape, t.stride, t.DType(), t.Device())
}

// View returns a new tensor with a different shape but sharing the same storage.
// The tensor must be contiguous.
func (t *Tensor) View(shape ...int) (*Tensor, error) {
	if !t.IsContiguous() {
		return nil, fmt.Errorf("view requires contiguous tensor")
	}

	// Handle -1 dimension
	newShape, err := InferShape(t.Numel(), shape)
	if err != nil {
		return nil, err
	}

	if Numel(newShape) != t.Numel() {
		return nil, fmt.Errorf("shape %v is incompatible with %d elements", shape, t.Numel())
	}

	return &Tensor{
		storage:      t.storage,
		shape:        newShape,
		stride:       ComputeStrides(newShape),
		offset:       t.offset,
		tape:         t.tape,         // propagate tape (same logical data)
		requiresGrad: t.requiresGrad, // propagate grad flag
	}, nil
}

// Reshape returns a reshaped tensor.
// If the tensor is contiguous, returns a view. Otherwise, copies data.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if t.IsContiguous() {
		return t.View(shape...)
	}

	// Non-contiguous: must copy
	contig := t.Contiguous()
	return contig.View(shape...)
}

// Contiguous returns a contiguous copy of the tensor if it's not already contiguous.
// If the tensor is already contiguous and has no offset, returns the same tensor.
func (t *Tensor) Contiguous() *Tensor {
	if t.IsContiguous() && t.offset == 0 {
		return t
	}

	// Try storage-native contiguous copy (GPU path)
	if maker, ok := t.storage.(ContiguousMaker); ok {
		newStorage, err := maker.MakeContiguous(t.shape, t.stride, t.offset)
		if err != nil {
			panic(fmt.Sprintf("contiguous: MakeContiguous failed: %v", err))
		}
		result := NewTensor(newStorage, t.shape, nil)
		if t.pool != nil {
			result.pool = t.pool
		}
		result.tape = t.tape // propagate tape (same logical data)
		return result
	}

	// CPU path
	cpu, ok := t.storage.(CPUDataAccessor)
	if !ok {
		panic(fmt.Sprintf("contiguous: storage type %T supports neither ContiguousMaker nor CPUDataAccessor", t.storage))
	}

	srcData := cpu.Data()
	numel := t.Numel()
	result := make([]float32, numel)

	// Iterate through all logical positions and copy to contiguous array
	IterateStrided(numel, t.shape, t.stride, t.offset, func(i, srcIdx int) {
		result[i] = srcData[srcIdx]
	})

	newStorage := NewCPUStorageFromSlice(result, t.DType())
	newTensor := NewTensor(newStorage, t.shape, nil)
	if t.pool != nil {
		newTensor.pool = t.pool
	}
	newTensor.tape = t.tape // propagate tape (same logical data)
	return newTensor
}

// Data returns the tensor data as a float32 slice.
// For CPU tensors, returns the underlying slice directly.
// For GPU tensors, copies data to host (allocates new slice).
func (t *Tensor) Data() ([]float32, error) {
	if cpu, ok := t.storage.(CPUDataAccessor); ok {
		return cpu.Data(), nil
	}
	if copier, ok := t.storage.(HostCopier); ok {
		return copier.CopyToHost()
	}
	return nil, fmt.Errorf("storage type %T does not support data access", t.storage)
}

// MustData returns the tensor data, panicking on error.
// Intended for use in tests.
func (t *Tensor) MustData() []float32 {
	data, err := t.Data()
	if err != nil {
		panic(err)
	}
	return data
}

// Compile-time check: *Tensor implements the Cloner interface pattern.
var _ interface{ Clone() *Tensor } = (*Tensor)(nil)
