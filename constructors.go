package tendo

import (
	"errors"
	"math/rand"
	"sync"
)

// Constructor errors.
var (
	// ErrShapeMismatch is returned when data length doesn't match shape.
	ErrShapeMismatch = errors.New("data length does not match shape")
	// ErrZeroStep is returned when step is zero in Arange.
	ErrZeroStep = errors.New("step cannot be zero")
)

// defaultDType is the default data type for tensor creation.
var (
	defaultDType   = Float32
	defaultDTypeMu sync.RWMutex
)

// SetDefaultDType sets the default data type for tensor creation.
func SetDefaultDType(d DType) {
	defaultDTypeMu.Lock()
	defer defaultDTypeMu.Unlock()
	defaultDType = d
}

// DefaultDType returns the current default data type.
func DefaultDType() DType {
	defaultDTypeMu.RLock()
	defer defaultDTypeMu.RUnlock()
	return defaultDType
}

// FromSlice creates a CPU tensor from a float32 slice with the given shape.
func FromSlice(data []float32, shape ...int) (*Tensor, error) {
	numel := Numel(shape)
	if numel != len(data) {
		return nil, ErrShapeMismatch
	}
	storage := NewCPUStorageFromSlice(data, DefaultDType())
	return NewTensor(storage, shape, nil), nil
}

// FromSliceOn creates a tensor from a float32 slice using the given backend.
func FromSliceOn(backend Backend, data []float32, shape ...int) (*Tensor, error) {
	return backend.FromSlice(data, shape...)
}

// MustFromSlice creates a CPU tensor from a float32 slice with the given shape.
// Panics on error.
func MustFromSlice(data []float32, shape ...int) *Tensor {
	t, err := FromSlice(data, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// MustFromSliceOn creates a tensor from a float32 slice using the given backend.
// Panics on error.
func MustFromSliceOn(backend Backend, data []float32, shape ...int) *Tensor {
	t, err := FromSliceOn(backend, data, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// Empty creates an uninitialized CPU tensor with the given shape.
func Empty(shape ...int) (*Tensor, error) {
	numel := Numel(shape)
	storage := NewCPUStorage(numel, DefaultDType())
	return NewTensor(storage, shape, nil), nil
}

// EmptyOn creates an uninitialized tensor using the given backend.
func EmptyOn(backend Backend, shape ...int) (*Tensor, error) {
	return backend.Empty(shape...)
}

// MustEmpty creates an uninitialized CPU tensor with the given shape.
// Panics on error.
func MustEmpty(shape ...int) *Tensor {
	t, err := Empty(shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// MustEmptyOn creates an uninitialized tensor using the given backend.
// Panics on error.
func MustEmptyOn(backend Backend, shape ...int) *Tensor {
	t, err := EmptyOn(backend, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// Zeros creates a CPU tensor filled with zeros.
func Zeros(shape ...int) (*Tensor, error) {
	t, err := Empty(shape...)
	if err != nil {
		return nil, err
	}
	t.storage.(CPUDataAccessor).Fill(0) //nolint:errcheck // CPUDataAccessor.Fill has no error return
	return t, nil
}

// ZerosOn creates a tensor filled with zeros using the given backend.
func ZerosOn(backend Backend, shape ...int) (*Tensor, error) {
	return backend.Zeros(shape...)
}

// MustZeros creates a CPU tensor filled with zeros.
// Panics on error.
func MustZeros(shape ...int) *Tensor {
	t, err := Zeros(shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// MustZerosOn creates a tensor filled with zeros using the given backend.
// Panics on error.
func MustZerosOn(backend Backend, shape ...int) *Tensor {
	t, err := ZerosOn(backend, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// Ones creates a CPU tensor filled with ones.
func Ones(shape ...int) (*Tensor, error) {
	t, err := Empty(shape...)
	if err != nil {
		return nil, err
	}
	t.storage.(CPUDataAccessor).Fill(1) //nolint:errcheck // CPUDataAccessor.Fill has no error return
	return t, nil
}

// OnesOn creates a tensor filled with ones using the given backend.
func OnesOn(backend Backend, shape ...int) (*Tensor, error) {
	return backend.Ones(shape...)
}

// MustOnes creates a CPU tensor filled with ones.
// Panics on error.
func MustOnes(shape ...int) *Tensor {
	t, err := Ones(shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// MustOnesOn creates a tensor filled with ones using the given backend.
// Panics on error.
func MustOnesOn(backend Backend, shape ...int) *Tensor {
	t, err := OnesOn(backend, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// Full creates a CPU tensor filled with the given value.
func Full(value float32, shape ...int) (*Tensor, error) {
	t, err := Empty(shape...)
	if err != nil {
		return nil, err
	}
	t.storage.(CPUDataAccessor).Fill(value) //nolint:errcheck // CPUDataAccessor.Fill has no error return
	return t, nil
}

// FullOn creates a tensor filled with the given value using the given backend.
func FullOn(backend Backend, value float32, shape ...int) (*Tensor, error) {
	return backend.Full(value, shape...)
}

// MustFull creates a CPU tensor filled with the given value.
// Panics on error.
func MustFull(value float32, shape ...int) *Tensor {
	t, err := Full(value, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// MustFullOn creates a tensor filled with the given value using the given backend.
// Panics on error.
func MustFullOn(backend Backend, value float32, shape ...int) *Tensor {
	t, err := FullOn(backend, value, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// Rand creates a CPU tensor with uniform random values in [0, 1).
func Rand(shape ...int) (*Tensor, error) {
	numel := Numel(shape)
	data := make([]float32, numel)
	for i := range data {
		data[i] = rand.Float32() //nolint:gosec // ML requires reproducible random, not crypto-secure
	}
	return FromSlice(data, shape...)
}

// RandOn creates a tensor with uniform random values using the given backend.
func RandOn(backend Backend, shape ...int) (*Tensor, error) {
	return backend.Rand(shape...)
}

// MustRand creates a CPU tensor with uniform random values in [0, 1).
// Panics on error.
func MustRand(shape ...int) *Tensor {
	t, err := Rand(shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// MustRandOn creates a tensor with uniform random values using the given backend.
// Panics on error.
func MustRandOn(backend Backend, shape ...int) *Tensor {
	t, err := RandOn(backend, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// RandN creates a CPU tensor with standard normal random values (mean=0, std=1).
func RandN(shape ...int) (*Tensor, error) {
	numel := Numel(shape)
	data := make([]float32, numel)
	for i := range data {
		data[i] = float32(rand.NormFloat64()) //nolint:gosec // ML requires reproducible random, not crypto-secure
	}
	return FromSlice(data, shape...)
}

// RandNOn creates a tensor with normal random values using the given backend.
func RandNOn(backend Backend, shape ...int) (*Tensor, error) {
	return backend.RandN(shape...)
}

// MustRandN creates a CPU tensor with standard normal random values.
// Panics on error.
func MustRandN(shape ...int) *Tensor {
	t, err := RandN(shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// MustRandNOn creates a tensor with normal random values using the given backend.
// Panics on error.
func MustRandNOn(backend Backend, shape ...int) *Tensor {
	t, err := RandNOn(backend, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// Arange creates a 1D CPU tensor with values from start to end (exclusive) with step.
func Arange(start, end, step float32) (*Tensor, error) {
	if step == 0 {
		return nil, ErrZeroStep
	}

	// Calculate number of elements
	n := int((end - start) / step)
	if n <= 0 {
		n = 0
	}

	data := make([]float32, n)
	for i := range data {
		data[i] = start + float32(i)*step
	}

	return FromSlice(data, n)
}

// ArangeOn creates a 1D tensor with values from start to end using the given backend.
func ArangeOn(backend Backend, start, end, step float32) (*Tensor, error) {
	return backend.Arange(start, end, step)
}

// MustArange creates a 1D CPU tensor with values from start to end (exclusive) with step.
// Panics on error.
func MustArange(start, end, step float32) *Tensor {
	t, err := Arange(start, end, step)
	if err != nil {
		panic(err)
	}
	return t
}

// MustArangeOn creates a 1D tensor with values from start to end using the given backend.
// Panics on error.
func MustArangeOn(backend Backend, start, end, step float32) *Tensor {
	t, err := ArangeOn(backend, start, end, step)
	if err != nil {
		panic(err)
	}
	return t
}

// Linspace creates a 1D CPU tensor with n evenly spaced values from start to end (inclusive).
func Linspace(start, end float32, n int) (*Tensor, error) {
	if n <= 0 {
		return Empty(0)
	}
	if n == 1 {
		return FromSlice([]float32{start}, 1)
	}

	data := make([]float32, n)
	step := (end - start) / float32(n-1)
	for i := range data {
		data[i] = start + float32(i)*step
	}
	data[n-1] = end // Ensure exact end value

	return FromSlice(data, n)
}

// LinspaceOn creates a 1D tensor with evenly spaced values using the given backend.
func LinspaceOn(backend Backend, start, end float32, n int) (*Tensor, error) {
	return backend.Linspace(start, end, n)
}

// MustLinspace creates a 1D CPU tensor with n evenly spaced values from start to end.
// Panics on error.
func MustLinspace(start, end float32, n int) *Tensor {
	t, err := Linspace(start, end, n)
	if err != nil {
		panic(err)
	}
	return t
}

// MustLinspaceOn creates a 1D tensor with evenly spaced values using the given backend.
// Panics on error.
func MustLinspaceOn(backend Backend, start, end float32, n int) *Tensor {
	t, err := LinspaceOn(backend, start, end, n)
	if err != nil {
		panic(err)
	}
	return t
}

// Eye creates a 2D CPU identity matrix of size n x n.
func Eye(n int) (*Tensor, error) {
	data := make([]float32, n*n)
	for i := 0; i < n; i++ {
		data[i*n+i] = 1
	}
	return FromSlice(data, n, n)
}

// EyeOn creates a 2D identity matrix using the given backend.
func EyeOn(backend Backend, n int) (*Tensor, error) {
	return backend.Eye(n)
}

// MustEye creates a 2D CPU identity matrix of size n x n.
// Panics on error.
func MustEye(n int) *Tensor {
	t, err := Eye(n)
	if err != nil {
		panic(err)
	}
	return t
}

// MustEyeOn creates a 2D identity matrix using the given backend.
// Panics on error.
func MustEyeOn(backend Backend, n int) *Tensor {
	t, err := EyeOn(backend, n)
	if err != nil {
		panic(err)
	}
	return t
}

// ZerosLike creates a tensor of zeros with the same shape and dtype as the input.
// The tensor is created on CPU.
func ZerosLike(t *Tensor) (*Tensor, error) {
	return Zeros(t.Shape()...)
}

// ZerosLikeOn creates a tensor of zeros with the same shape as the input using the given backend.
func ZerosLikeOn(backend Backend, t *Tensor) (*Tensor, error) {
	return backend.Zeros(t.Shape()...)
}

// MustZerosLike creates a tensor of zeros with the same shape and dtype as the input.
// Panics on error.
func MustZerosLike(t *Tensor) *Tensor {
	result, err := ZerosLike(t)
	if err != nil {
		panic(err)
	}
	return result
}

// OnesLike creates a tensor of ones with the same shape and dtype as the input.
// The tensor is created on CPU.
func OnesLike(t *Tensor) (*Tensor, error) {
	return Ones(t.Shape()...)
}

// OnesLikeOn creates a tensor of ones with the same shape as the input using the given backend.
func OnesLikeOn(backend Backend, t *Tensor) (*Tensor, error) {
	return backend.Ones(t.Shape()...)
}

// MustOnesLike creates a tensor of ones with the same shape and dtype as the input.
// Panics on error.
func MustOnesLike(t *Tensor) *Tensor {
	result, err := OnesLike(t)
	if err != nil {
		panic(err)
	}
	return result
}

// EmptyLike creates an uninitialized tensor with the same shape and dtype as the input.
// The tensor is created on CPU.
func EmptyLike(t *Tensor) (*Tensor, error) {
	return Empty(t.Shape()...)
}

// EmptyLikeOn creates an uninitialized tensor with the same shape as the input using the given backend.
func EmptyLikeOn(backend Backend, t *Tensor) (*Tensor, error) {
	return backend.Empty(t.Shape()...)
}

// MustEmptyLike creates an uninitialized tensor with the same shape and dtype as the input.
// Panics on error.
func MustEmptyLike(t *Tensor) *Tensor {
	result, err := EmptyLike(t)
	if err != nil {
		panic(err)
	}
	return result
}

// RandLike creates a tensor of uniform random values with the same shape as the input.
// The tensor is created on CPU.
func RandLike(t *Tensor) (*Tensor, error) {
	return Rand(t.Shape()...)
}

// RandLikeOn creates a tensor of uniform random values with the same shape as the input using the given backend.
func RandLikeOn(backend Backend, t *Tensor) (*Tensor, error) {
	return backend.Rand(t.Shape()...)
}

// MustRandLike creates a tensor of uniform random values with the same properties as the input.
// Panics on error.
func MustRandLike(t *Tensor) *Tensor {
	result, err := RandLike(t)
	if err != nil {
		panic(err)
	}
	return result
}

// RandNLike creates a tensor of normal random values with the same shape as the input.
// The tensor is created on CPU.
func RandNLike(t *Tensor) (*Tensor, error) {
	return RandN(t.Shape()...)
}

// RandNLikeOn creates a tensor of normal random values with the same shape as the input using the given backend.
func RandNLikeOn(backend Backend, t *Tensor) (*Tensor, error) {
	return backend.RandN(t.Shape()...)
}

// MustRandNLike creates a tensor of normal random values with the same properties as the input.
// Panics on error.
func MustRandNLike(t *Tensor) *Tensor {
	result, err := RandNLike(t)
	if err != nil {
		panic(err)
	}
	return result
}
