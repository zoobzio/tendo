// Package cuda provides CUDA backend implementation for tendo.
package cuda

import (
	"context"
	"math/rand"

	"github.com/zoobzio/tendo"
)

func init() {
	tendo.RegisterBackend(&Backend{})
}

// Backend implements tendo.Backend for CUDA devices.
type Backend struct {
	deviceIndex int
}

// NewBackend creates a new CUDA backend for the default device (0).
func NewBackend() *Backend {
	return &Backend{deviceIndex: 0}
}

// NewBackendWithDevice creates a new CUDA backend for the specified device.
func NewBackendWithDevice(deviceIndex int) *Backend {
	return &Backend{deviceIndex: deviceIndex}
}

// --- DeviceInfo ---

func (b *Backend) DeviceType() tendo.DeviceType { return tendo.CUDA }
func (b *Backend) IsAvailable() bool            { return IsCUDAAvailable() }
func (b *Backend) DeviceCount() int             { return CUDADeviceCount() }

// --- StorageOps ---

func (b *Backend) NewStorage(numel int, dtype tendo.DType, deviceIndex int) (tendo.Storage, error) {
	return NewStorage(numel, dtype, deviceIndex)
}

func (b *Backend) NewStorageFromSlice(data []float32, dtype tendo.DType, deviceIndex int) (tendo.Storage, error) {
	return NewStorageFromSlice(data, dtype, deviceIndex)
}

func (b *Backend) CopyFrom(t *tendo.Tensor) (*tendo.Tensor, error) {
	// Get data from source tensor
	data, err := t.Data()
	if err != nil {
		return nil, err
	}

	// Create new CUDA storage with the data
	storage, err := NewStorageFromSlice(data, t.DType(), b.deviceIndex)
	if err != nil {
		return nil, err
	}

	return tendo.NewTensor(storage, t.Shape(), nil), nil
}

// --- TensorFactory ---

func (b *Backend) Empty(shape ...int) (*tendo.Tensor, error) {
	numel := tendo.Numel(shape)
	storage, err := NewStorage(numel, tendo.Float32, b.deviceIndex)
	if err != nil {
		return nil, err
	}
	return tendo.NewTensor(storage, shape, nil), nil
}

func (b *Backend) Zeros(shape ...int) (*tendo.Tensor, error) {
	t, err := b.Empty(shape...)
	if err != nil {
		return nil, err
	}
	cudaStorage := t.Storage().(*Storage)
	if err := cudaStorage.Zero(); err != nil {
		cudaStorage.Free()
		return nil, err
	}
	return t, nil
}

func (b *Backend) Ones(shape ...int) (*tendo.Tensor, error) {
	return b.Full(1.0, shape...)
}

func (b *Backend) Full(value float32, shape ...int) (*tendo.Tensor, error) {
	numel := tendo.Numel(shape)
	// Create data on CPU and copy to GPU
	data := make([]float32, numel)
	for i := range data {
		data[i] = value
	}
	return b.FromSlice(data, shape...)
}

func (b *Backend) FromSlice(data []float32, shape ...int) (*tendo.Tensor, error) {
	numel := tendo.Numel(shape)
	if numel != len(data) {
		return nil, tendo.ErrShapeMismatch
	}
	storage, err := NewStorageFromSlice(data, tendo.Float32, b.deviceIndex)
	if err != nil {
		return nil, err
	}
	return tendo.NewTensor(storage, shape, nil), nil
}

func (b *Backend) FromInt64Slice(data []int64, shape ...int) (*tendo.Tensor, error) {
	numel := tendo.Numel(shape)
	if numel != len(data) {
		return nil, tendo.ErrShapeMismatch
	}
	storage, err := NewInt64StorageFromSlice(data, b.deviceIndex)
	if err != nil {
		return nil, err
	}
	return tendo.NewTensor(storage, shape, nil), nil
}

func (b *Backend) Rand(shape ...int) (*tendo.Tensor, error) {
	numel := tendo.Numel(shape)
	data := make([]float32, numel)
	for i := range data {
		data[i] = rand.Float32()
	}
	return b.FromSlice(data, shape...)
}

func (b *Backend) RandN(shape ...int) (*tendo.Tensor, error) {
	numel := tendo.Numel(shape)
	data := make([]float32, numel)
	for i := range data {
		data[i] = float32(rand.NormFloat64())
	}
	return b.FromSlice(data, shape...)
}

func (b *Backend) Eye(n int) (*tendo.Tensor, error) {
	data := make([]float32, n*n)
	for i := 0; i < n; i++ {
		data[i*n+i] = 1
	}
	return b.FromSlice(data, n, n)
}

func (b *Backend) Arange(start, end, step float32) (*tendo.Tensor, error) {
	if step == 0 {
		return nil, tendo.ErrZeroStep
	}
	n := int((end - start) / step)
	if n <= 0 {
		n = 0
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = start + float32(i)*step
	}
	return b.FromSlice(data, n)
}

func (b *Backend) Linspace(start, end float32, n int) (*tendo.Tensor, error) {
	if n <= 0 {
		return b.Empty(0)
	}
	if n == 1 {
		return b.FromSlice([]float32{start}, 1)
	}
	data := make([]float32, n)
	step := (end - start) / float32(n-1)
	for i := range data {
		data[i] = start + float32(i)*step
	}
	data[n-1] = end
	return b.FromSlice(data, n)
}

// --- UnaryOps ---

func (b *Backend) ReLU(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return Activation(t, CudnnActivationRelu)
}

func (b *Backend) Sigmoid(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return Activation(t, CudnnActivationSigmoid)
}

func (b *Backend) Tanh(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return Activation(t, CudnnActivationTanh)
}

func (b *Backend) GELU(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return PopcornGelu(t)
}

func (b *Backend) SiLU(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return Activation(t, CudnnActivationSwish)
}

func (b *Backend) Neg(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return ScaleTensor(t, -1.0)
}

func (b *Backend) Abs(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return PopcornAbs(t)
}

func (b *Backend) Exp(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return PopcornExp(t)
}

func (b *Backend) Log(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return PopcornLog(t)
}

func (b *Backend) Sqrt(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	// cudnnOpTensorSqrt is unary: C = sqrt(alpha1 * A), B is ignored
	return OpTensor(t, t, cudnnOpTensorSqrt, 1.0, 0.0)
}

func (b *Backend) Square(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return PopcornSquare(t)
}

func (b *Backend) Sign(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return PopcornSign(t)
}

func (b *Backend) Sin(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return PopcornSin(t)
}

func (b *Backend) Cos(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return PopcornCos(t)
}

// --- BinaryOps ---

func (b *Backend) Add(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	return OpTensor(a, other, cudnnOpTensorAdd, 1.0, 1.0)
}

func (b *Backend) Sub(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	return OpTensor(a, other, cudnnOpTensorAdd, 1.0, -1.0)
}

func (b *Backend) Mul(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	return OpTensor(a, other, cudnnOpTensorMul, 1.0, 1.0)
}

func (b *Backend) Div(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	// Popcorn requires same-shape tensors (no broadcasting)
	// For broadcasting cases, would need to expand tensors first
	return PopcornDiv(a, other)
}

func (b *Backend) Pow(_ context.Context, t *tendo.Tensor, exp float32) (*tendo.Tensor, error) {
	return PopcornPowScalar(t, exp)
}

// --- MatrixOps ---

func (b *Backend) MatMul(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	return MatMul(a, other)
}

// --- ReduceOps ---

func (b *Backend) Sum(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	result, err := ReduceTensor(t, dims, "sum")
	if err != nil {
		return nil, err
	}
	if !keepdim {
		return squeezeDims(result, dims)
	}
	return result, nil
}

func (b *Backend) Mean(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	result, err := ReduceTensor(t, dims, "mean")
	if err != nil {
		return nil, err
	}
	if !keepdim {
		return squeezeDims(result, dims)
	}
	return result, nil
}

func (b *Backend) Max(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	result, err := ReduceTensor(t, dims, "max")
	if err != nil {
		return nil, err
	}
	if !keepdim {
		return squeezeDims(result, dims)
	}
	return result, nil
}

func (b *Backend) Min(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	result, err := ReduceTensor(t, dims, "min")
	if err != nil {
		return nil, err
	}
	if !keepdim {
		return squeezeDims(result, dims)
	}
	return result, nil
}

func (b *Backend) Var(ctx context.Context, t *tendo.Tensor, dims []int, keepdim bool, correction int) (*tendo.Tensor, error) {
	// mean = Mean(t, dims, keepdim=true) for broadcasting
	mean, err := b.Mean(ctx, t, dims, true)
	if err != nil {
		return nil, err
	}

	// diff = t - mean (broadcasts)
	diff, err := b.Sub(ctx, t, mean)
	mean.Free()
	if err != nil {
		return nil, err
	}

	// squared = diff * diff
	squared, err := b.Square(ctx, diff)
	diff.Free()
	if err != nil {
		return nil, err
	}

	// variance = Mean(squared, dims, keepdim)
	// Note: correction (Bessel's) would need count adjustment - skip for now
	result, err := b.Mean(ctx, squared, dims, keepdim)
	squared.Free()
	return result, err
}

func (b *Backend) Std(ctx context.Context, t *tendo.Tensor, dims []int, keepdim bool, correction int) (*tendo.Tensor, error) {
	variance, err := b.Var(ctx, t, dims, keepdim, correction)
	if err != nil {
		return nil, err
	}
	result, err := b.Sqrt(ctx, variance)
	variance.Free()
	return result, err
}

func (b *Backend) Prod(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	result, err := ReduceTensor(t, dims, "prod")
	if err != nil {
		return nil, err
	}
	if !keepdim {
		return squeezeDims(result, dims)
	}
	return result, nil
}

func (b *Backend) ArgMax(_ context.Context, t *tendo.Tensor, dim int, keepdim bool) (*tendo.Tensor, error) {
	ndim := len(t.Shape())
	if dim < 0 {
		dim = ndim + dim
	}

	// Popcorn only supports reduction along last dimension
	if dim != ndim-1 {
		return nil, &tendo.ErrNotImplemented{Op: "ArgMax", Backend: "CUDA (only last dim supported)"}
	}

	result, err := PopcornArgMax(t)
	if err != nil {
		return nil, err
	}
	if keepdim {
		return unsqueezeDim(result, dim)
	}
	return result, nil
}

func (b *Backend) ArgMin(_ context.Context, t *tendo.Tensor, dim int, keepdim bool) (*tendo.Tensor, error) {
	ndim := len(t.Shape())
	if dim < 0 {
		dim = ndim + dim
	}

	// Popcorn only supports reduction along last dimension
	if dim != ndim-1 {
		return nil, &tendo.ErrNotImplemented{Op: "ArgMin", Backend: "CUDA (only last dim supported)"}
	}

	result, err := PopcornArgMin(t)
	if err != nil {
		return nil, err
	}
	if keepdim {
		return unsqueezeDim(result, dim)
	}
	return result, nil
}

// --- ActivationOps ---

func (b *Backend) Softmax(_ context.Context, t *tendo.Tensor, dim int) (*tendo.Tensor, error) {
	return Softmax(t, dim, false)
}

func (b *Backend) LogSoftmax(_ context.Context, t *tendo.Tensor, dim int) (*tendo.Tensor, error) {
	return Softmax(t, dim, true)
}

func (b *Backend) LeakyReLU(_ context.Context, t *tendo.Tensor, negativeSlope float32) (*tendo.Tensor, error) {
	return PopcornLeakyRelu(t, negativeSlope)
}

func (b *Backend) Dropout(ctx context.Context, t *tendo.Tensor, p float32, training bool) (*tendo.Tensor, error) {
	if !training {
		return t, nil
	}
	return Dropout(ctx, t, p)
}

// --- NormOps ---

func (b *Backend) BatchNorm2d(ctx context.Context, input, weight, bias, runningMean, runningVar *tendo.Tensor, epsilon, momentum float32, training bool) (*tendo.Tensor, error) {
	// BatchNorm2d uses training flag from context
	// Store training flag in context for CUDA implementation
	if training {
		ctx = tendo.WithTraining(ctx)
	}
	return BatchNorm2d(ctx, input, weight, bias, runningMean, runningVar, epsilon, momentum)
}

func (b *Backend) LayerNorm(_ context.Context, input *tendo.Tensor, normalizedShape []int, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	return PopcornLayerNorm(input, weight, bias, normalizedShape, epsilon)
}

func (b *Backend) RMSNorm(_ context.Context, input *tendo.Tensor, normalizedShape []int, weight *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	return PopcornRMSNorm(input, weight, normalizedShape, epsilon)
}

func (b *Backend) GroupNorm(_ context.Context, input *tendo.Tensor, numGroups int, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	return GroupNorm(input, numGroups, weight, bias, epsilon)
}

func (b *Backend) InstanceNorm2d(_ context.Context, input, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	return InstanceNorm2d(input, weight, bias, epsilon)
}

// --- ConvOps ---

func (b *Backend) Conv2d(_ context.Context, input, weight *tendo.Tensor, padding, stride, dilation [2]int, groups int) (*tendo.Tensor, error) {
	return Conv2d(input, weight, padding, stride, dilation, groups)
}

func (b *Backend) ConvTranspose2d(_ context.Context, input, weight *tendo.Tensor, padding, outputPadding, stride, dilation [2]int, groups int) (*tendo.Tensor, error) {
	return ConvTranspose2d(input, weight, padding, outputPadding, stride, dilation, groups)
}

// --- PoolOps ---

func (b *Backend) MaxPool2d(_ context.Context, input *tendo.Tensor, kernelSize, stride, padding [2]int) (*tendo.Tensor, error) {
	return Pool2d(input, kernelSize, stride, padding, true)
}

func (b *Backend) AvgPool2d(_ context.Context, input *tendo.Tensor, kernelSize, stride, padding [2]int) (*tendo.Tensor, error) {
	return Pool2d(input, kernelSize, stride, padding, false)
}

func (b *Backend) AdaptiveAvgPool2d(_ context.Context, input *tendo.Tensor, outputSize [2]int) (*tendo.Tensor, error) {
	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, &tendo.ShapeError{Op: "adaptiveavgpool2d", Message: "input must be 4D [N, C, H, W]"}
	}

	inH, inW := inShape[2], inShape[3]
	outH, outW := outputSize[0], outputSize[1]

	// Calculate kernel size and stride to achieve desired output size
	// Using floor division approach
	kernelH := (inH + outH - 1) / outH
	kernelW := (inW + outW - 1) / outW
	strideH := inH / outH
	strideW := inW / outW

	if strideH == 0 {
		strideH = 1
	}
	if strideW == 0 {
		strideW = 1
	}

	return Pool2d(input, [2]int{kernelH, kernelW}, [2]int{strideH, strideW}, [2]int{0, 0}, false)
}

func (b *Backend) AdaptiveMaxPool2d(_ context.Context, input *tendo.Tensor, outputSize [2]int) (*tendo.Tensor, error) {
	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, &tendo.ShapeError{Op: "adaptivemaxpool2d", Message: "input must be 4D [N, C, H, W]"}
	}

	inH, inW := inShape[2], inShape[3]
	outH, outW := outputSize[0], outputSize[1]

	// Calculate kernel size and stride to achieve desired output size
	kernelH := (inH + outH - 1) / outH
	kernelW := (inW + outW - 1) / outW
	strideH := inH / outH
	strideW := inW / outW

	if strideH == 0 {
		strideH = 1
	}
	if strideW == 0 {
		strideW = 1
	}

	return Pool2d(input, [2]int{kernelH, kernelW}, [2]int{strideH, strideW}, [2]int{0, 0}, true)
}

// --- LossOps ---

func (b *Backend) MSELoss(ctx context.Context, input, target *tendo.Tensor, reduction string) (*tendo.Tensor, error) {
	// diff = input - target
	diff, err := b.Sub(ctx, input, target)
	if err != nil {
		return nil, err
	}

	// squared = diff * diff
	squared, err := b.Square(ctx, diff)
	diff.Free()
	if err != nil {
		return nil, err
	}

	// Apply reduction
	switch reduction {
	case "mean":
		result, err := b.Mean(ctx, squared, nil, false)
		squared.Free()
		return result, err
	case "sum":
		result, err := b.Sum(ctx, squared, nil, false)
		squared.Free()
		return result, err
	case "none":
		return squared, nil
	default:
		squared.Free()
		return nil, &tendo.ShapeError{Op: "MSELoss", Message: "invalid reduction: " + reduction}
	}
}

func (b *Backend) L1Loss(ctx context.Context, input, target *tendo.Tensor, reduction string) (*tendo.Tensor, error) {
	// diff = input - target
	diff, err := b.Sub(ctx, input, target)
	if err != nil {
		return nil, err
	}

	// absDiff = abs(diff)
	absDiff, err := b.Abs(ctx, diff)
	diff.Free()
	if err != nil {
		return nil, err
	}

	// Apply reduction
	switch reduction {
	case "mean":
		result, err := b.Mean(ctx, absDiff, nil, false)
		absDiff.Free()
		return result, err
	case "sum":
		result, err := b.Sum(ctx, absDiff, nil, false)
		absDiff.Free()
		return result, err
	case "none":
		return absDiff, nil
	default:
		absDiff.Free()
		return nil, &tendo.ShapeError{Op: "L1Loss", Message: "invalid reduction: " + reduction}
	}
}

func (b *Backend) CrossEntropyLoss(ctx context.Context, input, target *tendo.Tensor, reduction string) (*tendo.Tensor, error) {
	// CrossEntropy = LogSoftmax + NLLLoss
	logSoftmax, err := b.LogSoftmax(ctx, input, -1)
	if err != nil {
		return nil, err
	}

	result, err := b.NLLLoss(ctx, logSoftmax, target, reduction)
	logSoftmax.Free()
	return result, err
}

func (b *Backend) NLLLoss(ctx context.Context, input, target *tendo.Tensor, reduction string) (*tendo.Tensor, error) {
	// Gather log probabilities at target indices
	gathered, err := PopcornGather(input, target)
	if err != nil {
		return nil, err
	}

	// Negate (NLL = negative log likelihood)
	negated, err := b.Neg(ctx, gathered)
	gathered.Free()
	if err != nil {
		return nil, err
	}

	// Apply reduction
	switch reduction {
	case "mean":
		result, err := b.Mean(ctx, negated, nil, false)
		negated.Free()
		return result, err
	case "sum":
		result, err := b.Sum(ctx, negated, nil, false)
		negated.Free()
		return result, err
	case "none":
		return negated, nil
	default:
		negated.Free()
		return nil, &tendo.ShapeError{Op: "NLLLoss", Message: "invalid reduction: " + reduction}
	}
}

// --- CompareOps ---

func (b *Backend) Clamp(_ context.Context, t *tendo.Tensor, min, max float32) (*tendo.Tensor, error) {
	return PopcornClamp(t, min, max)
}

func (b *Backend) Where(_ context.Context, condition, x, y *tendo.Tensor) (*tendo.Tensor, error) {
	// Popcorn requires same-shape tensors (no broadcasting)
	// For broadcasting cases, would need to expand tensors first
	return PopcornWhere(condition, x, y)
}

func (b *Backend) Tril(_ context.Context, t *tendo.Tensor, k int) (*tendo.Tensor, error) {
	return PopcornTril(t, k)
}

// --- EmbeddingOps ---

func (b *Backend) Embedding(_ context.Context, weight, indices *tendo.Tensor) (*tendo.Tensor, error) {
	return PopcornEmbedding(weight, indices)
}

// --- ShapeOps ---

func (b *Backend) Cat(_ context.Context, tensors []*tendo.Tensor, dim int) (*tendo.Tensor, error) {
	return PopcornCat(tensors, dim)
}

func (b *Backend) Stack(_ context.Context, tensors []*tendo.Tensor, dim int) (*tendo.Tensor, error) {
	return PopcornStack(tensors, dim)
}

// --- QuantizedOps ---

// DequantizeMatmul performs fused dequantize + matmul for quantized linear layers.
// Implements nn.QuantizedLinearBackend interface.
// x: [batch*seq, in_features], qweight: [out_features, in_features], scale: depends on groupSize
// Output: [batch*seq, out_features]
func (b *Backend) DequantizeMatmul(_ context.Context, x *tendo.Tensor, qweight, scale *tendo.Tensor, groupSize int) (*tendo.Tensor, error) {
	// Flatten x to 2D if needed: [batch, seq, in] -> [batch*seq, in]
	xShape := x.Shape()
	var x2d *tendo.Tensor
	M := 1
	for i := 0; i < len(xShape)-1; i++ {
		M *= xShape[i]
	}
	K := xShape[len(xShape)-1]

	if len(xShape) > 2 {
		// Reshape to 2D
		var err error
		x2d, err = tendo.NewReshape(M, K).Process(context.Background(), x)
		if err != nil {
			return nil, err
		}
	} else {
		x2d = x
	}

	// Call the appropriate kernel based on groupSize
	var out *tendo.Tensor
	var err error
	if groupSize == 0 {
		// Per-channel quantization
		out, err = PopcornDequantizeMatmul(x2d, qweight, scale)
	} else {
		// Per-group quantization
		out, err = PopcornDequantizeMatmulGrouped(x2d, qweight, scale, groupSize)
	}
	if err != nil {
		return nil, err
	}

	// Reshape output back to match input batch dims
	// out is [M, N], we want [batch, seq, N] if input was [batch, seq, in]
	if len(xShape) > 2 {
		outShape := make([]int, len(xShape))
		copy(outShape, xShape[:len(xShape)-1])
		outShape[len(outShape)-1] = out.Shape()[1] // N = out_features
		return tendo.NewReshape(outShape...).Process(context.Background(), out)
	}

	return out, nil
}

// --- Helper functions ---

// squeezeDims removes the specified dimensions from the tensor shape.
func squeezeDims(t *tendo.Tensor, dims []int) (*tendo.Tensor, error) {
	shape := t.Shape()
	normDims := make(map[int]bool)
	for _, d := range dims {
		if d < 0 {
			d = len(shape) + d
		}
		normDims[d] = true
	}

	var newShape []int
	for i, s := range shape {
		if !normDims[i] {
			newShape = append(newShape, s)
		}
	}
	if len(newShape) == 0 {
		newShape = []int{}
	}

	return tendo.NewTensor(t.Storage(), newShape, nil), nil
}

// unsqueezeDim inserts a dimension of size 1 at the specified position.
func unsqueezeDim(t *tendo.Tensor, dim int) (*tendo.Tensor, error) {
	shape := t.Shape()
	if dim < 0 {
		dim = len(shape) + 1 + dim
	}

	newShape := make([]int, len(shape)+1)
	copy(newShape[:dim], shape[:dim])
	newShape[dim] = 1
	copy(newShape[dim+1:], shape[dim:])

	return tendo.NewTensor(t.Storage(), newShape, nil), nil
}

// Compile-time check
var _ tendo.Backend = (*Backend)(nil)
