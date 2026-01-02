package tendo

import (
	"context"
)

// StorageOps defines storage creation and device transfer operations.
type StorageOps interface {
	// NewStorage creates storage on this backend's device.
	NewStorage(numel int, dtype DType, deviceIndex int) (Storage, error)

	// NewStorageFromSlice creates storage initialized from host data.
	NewStorageFromSlice(data []float32, dtype DType, deviceIndex int) (Storage, error)

	// CopyFrom copies a tensor from another device to this backend's device.
	// Returns a new tensor on this backend's device.
	CopyFrom(t *Tensor) (*Tensor, error)
}

// DeviceInfo provides information about the backend's device.
type DeviceInfo interface {
	// DeviceType returns the type of device this backend uses.
	DeviceType() DeviceType

	// IsAvailable returns true if the backend is functional.
	IsAvailable() bool

	// DeviceCount returns the number of devices available.
	DeviceCount() int
}

// TensorFactory provides convenience methods for creating tensors.
type TensorFactory interface {
	// Zeros creates a tensor filled with zeros.
	Zeros(shape ...int) (*Tensor, error)

	// Ones creates a tensor filled with ones.
	Ones(shape ...int) (*Tensor, error)

	// Empty creates an uninitialized tensor.
	Empty(shape ...int) (*Tensor, error)

	// Full creates a tensor filled with a scalar value.
	Full(value float32, shape ...int) (*Tensor, error)

	// FromSlice creates a tensor from a float32 slice.
	FromSlice(data []float32, shape ...int) (*Tensor, error)

	// Rand creates a tensor with uniform random values in [0, 1).
	Rand(shape ...int) (*Tensor, error)

	// RandN creates a tensor with standard normal random values.
	RandN(shape ...int) (*Tensor, error)

	// Eye creates a 2D identity matrix.
	Eye(n int) (*Tensor, error)

	// Arange creates a 1D tensor with values from start to end with step.
	Arange(start, end, step float32) (*Tensor, error)

	// Linspace creates a 1D tensor with n evenly spaced values.
	Linspace(start, end float32, n int) (*Tensor, error)
}

// UnaryOps defines element-wise unary operations.
type UnaryOps interface {
	// Activation functions
	ReLU(ctx context.Context, t *Tensor) (*Tensor, error)
	Sigmoid(ctx context.Context, t *Tensor) (*Tensor, error)
	Tanh(ctx context.Context, t *Tensor) (*Tensor, error)
	GELU(ctx context.Context, t *Tensor) (*Tensor, error)
	SiLU(ctx context.Context, t *Tensor) (*Tensor, error)

	// Math functions
	Neg(ctx context.Context, t *Tensor) (*Tensor, error)
	Abs(ctx context.Context, t *Tensor) (*Tensor, error)
	Exp(ctx context.Context, t *Tensor) (*Tensor, error)
	Log(ctx context.Context, t *Tensor) (*Tensor, error)
	Sqrt(ctx context.Context, t *Tensor) (*Tensor, error)
	Square(ctx context.Context, t *Tensor) (*Tensor, error)
	Sign(ctx context.Context, t *Tensor) (*Tensor, error)

	// Trigonometric functions
	Sin(ctx context.Context, t *Tensor) (*Tensor, error)
	Cos(ctx context.Context, t *Tensor) (*Tensor, error)
}

// BinaryOps defines element-wise binary operations.
type BinaryOps interface {
	Add(ctx context.Context, a, b *Tensor) (*Tensor, error)
	Sub(ctx context.Context, a, b *Tensor) (*Tensor, error)
	Mul(ctx context.Context, a, b *Tensor) (*Tensor, error)
	Div(ctx context.Context, a, b *Tensor) (*Tensor, error)
	Pow(ctx context.Context, t *Tensor, exp float32) (*Tensor, error)
}

// MatrixOps defines matrix operations.
type MatrixOps interface {
	MatMul(ctx context.Context, a, b *Tensor) (*Tensor, error)
}

// ReduceOps defines reduction operations.
type ReduceOps interface {
	Sum(ctx context.Context, t *Tensor, dims []int, keepdim bool) (*Tensor, error)
	Mean(ctx context.Context, t *Tensor, dims []int, keepdim bool) (*Tensor, error)
	Max(ctx context.Context, t *Tensor, dims []int, keepdim bool) (*Tensor, error)
	Min(ctx context.Context, t *Tensor, dims []int, keepdim bool) (*Tensor, error)
	Var(ctx context.Context, t *Tensor, dims []int, keepdim bool, correction int) (*Tensor, error)
	Std(ctx context.Context, t *Tensor, dims []int, keepdim bool, correction int) (*Tensor, error)
	Prod(ctx context.Context, t *Tensor, dims []int, keepdim bool) (*Tensor, error)
	ArgMax(ctx context.Context, t *Tensor, dim int, keepdim bool) (*Tensor, error)
	ArgMin(ctx context.Context, t *Tensor, dim int, keepdim bool) (*Tensor, error)
}

// ActivationOps defines parameterized activation operations.
type ActivationOps interface {
	Softmax(ctx context.Context, t *Tensor, dim int) (*Tensor, error)
	LogSoftmax(ctx context.Context, t *Tensor, dim int) (*Tensor, error)
	LeakyReLU(ctx context.Context, t *Tensor, negativeSlope float32) (*Tensor, error)
	// Dropout applies dropout and returns (output, mask).
	// During inference (training=false), mask is nil and output equals input.
	// The mask is needed for the backward pass.
	Dropout(ctx context.Context, t *Tensor, p float32, training bool) (output, mask *Tensor, err error)
}

// NormOps defines normalization operations.
type NormOps interface {
	BatchNorm2d(ctx context.Context, input, weight, bias, runningMean, runningVar *Tensor, epsilon, momentum float32, training bool) (*Tensor, error)
	LayerNorm(ctx context.Context, input *Tensor, normalizedShape []int, weight, bias *Tensor, epsilon float32) (*Tensor, error)
	RMSNorm(ctx context.Context, input *Tensor, normalizedShape []int, weight *Tensor, epsilon float32) (*Tensor, error)
	GroupNorm(ctx context.Context, input *Tensor, numGroups int, weight, bias *Tensor, epsilon float32) (*Tensor, error)
	InstanceNorm2d(ctx context.Context, input, weight, bias *Tensor, epsilon float32) (*Tensor, error)
}

// ConvOps defines convolution operations.
type ConvOps interface {
	Conv2d(ctx context.Context, input, weight *Tensor, padding, stride, dilation [2]int, groups int) (*Tensor, error)
	ConvTranspose2d(ctx context.Context, input, weight *Tensor, padding, outputPadding, stride, dilation [2]int, groups int) (*Tensor, error)
}

// PoolOps defines pooling operations.
type PoolOps interface {
	MaxPool2d(ctx context.Context, input *Tensor, kernelSize, stride, padding [2]int) (*Tensor, []int, error)
	AvgPool2d(ctx context.Context, input *Tensor, kernelSize, stride, padding [2]int) (*Tensor, error)
	AdaptiveAvgPool2d(ctx context.Context, input *Tensor, outputSize [2]int) (*Tensor, error)
	AdaptiveMaxPool2d(ctx context.Context, input *Tensor, outputSize [2]int) (*Tensor, []int, error)
}

// LossOps defines loss functions.
type LossOps interface {
	MSELoss(ctx context.Context, input, target *Tensor, reduction string) (*Tensor, error)
	L1Loss(ctx context.Context, input, target *Tensor, reduction string) (*Tensor, error)
	CrossEntropyLoss(ctx context.Context, input, target *Tensor, reduction string) (*Tensor, error)
	NLLLoss(ctx context.Context, input, target *Tensor, reduction string) (*Tensor, error)
}

// CompareOps defines comparison and selection operations.
type CompareOps interface {
	Clamp(ctx context.Context, t *Tensor, minVal, maxVal float32) (*Tensor, error)
	Where(ctx context.Context, condition, x, y *Tensor) (*Tensor, error)

	// Tril returns lower triangular part of a 2D tensor.
	// Elements above diagonal + k are zeroed.
	// k=0: main diagonal, k<0: below, k>0: above.
	Tril(ctx context.Context, t *Tensor, k int) (*Tensor, error)
}

// EmbeddingOps defines embedding operations.
type EmbeddingOps interface {
	// Embedding performs embedding lookup.
	// weight shape: [vocab_size, embed_dim]
	// indices shape: arbitrary (e.g., [batch, seq_len])
	// output shape: indices.shape + [embed_dim]
	Embedding(ctx context.Context, weight, indices *Tensor) (*Tensor, error)
}

// ShapeOps defines tensor shape manipulation operations that require data movement.
type ShapeOps interface {
	// Cat concatenates tensors along an existing dimension.
	Cat(ctx context.Context, tensors []*Tensor, dim int) (*Tensor, error)

	// Stack stacks tensors along a new dimension.
	Stack(ctx context.Context, tensors []*Tensor, dim int) (*Tensor, error)
}

// OptimizerOps defines optimizer operations.
type OptimizerOps interface {
	// AdamW performs a fused AdamW optimizer step.
	// Updates param, m, v tensors in-place.
	// biasCorrection1 = 1 - beta1^step, biasCorrection2 = 1 - beta2^step
	AdamW(ctx context.Context, param, grad, m, v *Tensor, lr, beta1, beta2, epsilon, weightDecay, biasCorrection1, biasCorrection2 float32) error
}

// CoreBackend is the minimal interface required by all backends.
// It provides storage, device information, and tensor creation.
type CoreBackend interface {
	StorageOps
	DeviceInfo
	TensorFactory
}

// ArithmeticBackend provides element-wise arithmetic operations.
type ArithmeticBackend interface {
	UnaryOps
	BinaryOps
}

// MatrixBackend provides matrix operations.
type MatrixBackend interface {
	MatrixOps
}

// NeuralBackend provides neural network operations.
type NeuralBackend interface {
	ActivationOps
	NormOps
	ConvOps
	PoolOps
	LossOps
	EmbeddingOps
	ShapeOps
}

// ReduceBackend provides reduction operations.
type ReduceBackend interface {
	ReduceOps
}

// CompareBackend provides comparison and selection operations.
type CompareBackend interface {
	CompareOps
}

// Backend is the unified interface for compute backends.
// It combines all capability interfaces for full functionality.
type Backend interface {
	CoreBackend
	ArithmeticBackend
	MatrixBackend
	NeuralBackend
	ReduceBackend
	CompareBackend
}

// ErrNotImplemented is returned when a backend doesn't support an operation.
type ErrNotImplemented struct {
	Op      string
	Backend string
}

func (e *ErrNotImplemented) Error() string {
	return e.Op + " not implemented for " + e.Backend + " backend"
}
