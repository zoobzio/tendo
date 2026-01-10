// Package cpu provides CPU backend implementation for tendo.
package cpu

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"github.com/zoobzio/tendo"
)

// Reduction modes for loss functions.
const (
	reductionNone = "none"
	reductionSum  = "sum"
	reductionMean = "mean"
)

// Backend implements tendo.Backend for CPU devices.
type Backend struct{}

// NewBackend creates a new CPU backend.
func NewBackend() *Backend {
	return &Backend{}
}

// --- DeviceInfo ---

func (b *Backend) DeviceType() tendo.DeviceType { return tendo.CPU }
func (b *Backend) IsAvailable() bool            { return true }
func (b *Backend) DeviceCount() int             { return 1 }

// --- StorageOps ---

func (b *Backend) NewStorage(numel int, dtype tendo.DType, _ int) (tendo.Storage, error) {
	return NewStorage(numel, dtype), nil
}

func (b *Backend) NewStorageFromSlice(data []float32, dtype tendo.DType, _ int) (tendo.Storage, error) {
	return NewStorageFromSlice(data, dtype), nil
}

func (b *Backend) CopyFrom(t *tendo.Tensor) (*tendo.Tensor, error) {
	data, err := t.Data()
	if err != nil {
		return nil, err
	}
	storage := NewStorageFromSlice(data, t.DType())
	return tendo.NewTensor(storage, t.Shape(), nil), nil
}

// --- TensorFactory ---

func (b *Backend) Empty(shape ...int) (*tendo.Tensor, error) {
	numel := tendo.Numel(shape)
	storage := NewStorage(numel, tendo.Float32)
	return tendo.NewTensor(storage, shape, nil), nil
}

func (b *Backend) Zeros(shape ...int) (*tendo.Tensor, error) {
	t, err := b.Empty(shape...)
	if err != nil {
		return nil, err
	}
	t.Storage().(*Storage).Fill(0)
	return t, nil
}

func (b *Backend) Ones(shape ...int) (*tendo.Tensor, error) {
	t, err := b.Empty(shape...)
	if err != nil {
		return nil, err
	}
	t.Storage().(*Storage).Fill(1)
	return t, nil
}

func (b *Backend) Full(value float32, shape ...int) (*tendo.Tensor, error) {
	t, err := b.Empty(shape...)
	if err != nil {
		return nil, err
	}
	t.Storage().(*Storage).Fill(value)
	return t, nil
}

func (b *Backend) FromSlice(data []float32, shape ...int) (*tendo.Tensor, error) {
	numel := tendo.Numel(shape)
	if numel != len(data) {
		return nil, tendo.ErrShapeMismatch
	}
	storage := NewStorageFromSlice(data, tendo.Float32)
	return tendo.NewTensor(storage, shape, nil), nil
}

func (b *Backend) FromInt64Slice(data []int64, shape ...int) (*tendo.Tensor, error) {
	numel := tendo.Numel(shape)
	if numel != len(data) {
		return nil, tendo.ErrShapeMismatch
	}
	storage := NewInt64StorageFromSlice(data)
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
	return b.unaryElementwise(t, func(x float32) float32 {
		if x > 0 {
			return x
		}
		return 0
	})
}

func (b *Backend) Sigmoid(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		return float32(1.0 / (1.0 + math.Exp(-float64(x))))
	})
}

func (b *Backend) Tanh(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		return float32(math.Tanh(float64(x)))
	})
}

func (b *Backend) GELU(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	sqrt2 := math.Sqrt(2.0)
	return b.unaryElementwise(t, func(x float32) float32 {
		return float32(float64(x) * 0.5 * (1.0 + math.Erf(float64(x)/sqrt2)))
	})
}

func (b *Backend) SiLU(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		sigmoid := float32(1.0 / (1.0 + math.Exp(-float64(x))))
		return x * sigmoid
	})
}

func (b *Backend) Neg(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 { return -x })
}

func (b *Backend) Abs(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		if x < 0 {
			return -x
		}
		return x
	})
}

func (b *Backend) Exp(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		return float32(math.Exp(float64(x)))
	})
}

func (b *Backend) Log(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		return float32(math.Log(float64(x)))
	})
}

func (b *Backend) Sqrt(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		return float32(math.Sqrt(float64(x)))
	})
}

func (b *Backend) Square(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 { return x * x })
}

func (b *Backend) Sign(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		if x > 0 {
			return 1
		}
		if x < 0 {
			return -1
		}
		return 0
	})
}

func (b *Backend) Sin(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 { return float32(math.Sin(float64(x))) })
}

func (b *Backend) Cos(_ context.Context, t *tendo.Tensor) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 { return float32(math.Cos(float64(x))) })
}

// --- BinaryOps ---

func (b *Backend) Add(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	return b.binaryElementwise(a, other, func(x, y float32) float32 { return x + y })
}

func (b *Backend) Sub(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	return b.binaryElementwise(a, other, func(x, y float32) float32 { return x - y })
}

func (b *Backend) Mul(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	return b.binaryElementwise(a, other, func(x, y float32) float32 { return x * y })
}

func (b *Backend) Div(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	return b.binaryElementwise(a, other, func(x, y float32) float32 { return x / y })
}

func (b *Backend) Pow(_ context.Context, t *tendo.Tensor, exp float32) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		return float32(math.Pow(float64(x), float64(exp)))
	})
}

// --- MatrixOps ---

func (b *Backend) MatMul(_ context.Context, a, other *tendo.Tensor) (*tendo.Tensor, error) {
	outShape, err := tendo.ValidateMatMul(a.Shape(), other.Shape())
	if err != nil {
		return nil, err
	}

	// Ensure contiguous for BLAS operations
	aContig := a
	if !a.IsContiguous() || a.Offset() != 0 {
		aContig = a.Contiguous()
	}
	otherContig := other
	if !other.IsContiguous() || other.Offset() != 0 {
		otherContig = other.Contiguous()
	}

	cpuA, ok := aContig.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: a.Device().Type}
	}
	cpuB, ok := otherContig.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: other.Device().Type}
	}

	out := cpuMatMul(cpuA, cpuB, aContig.Shape(), otherContig.Shape(), outShape)
	return tendo.NewTensor(out, outShape, nil), nil
}

// --- ReduceOps ---

func (b *Backend) Sum(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	return b.reduce(t, dims, keepdim, func(acc, val float32) float32 { return acc + val }, 0)
}

func (b *Backend) Mean(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	out, err := b.reduce(t, dims, keepdim, func(acc, val float32) float32 { return acc + val }, 0)
	if err != nil {
		return nil, err
	}

	// Calculate divisor
	shape := t.Shape()
	divisor := 1
	if len(dims) == 0 {
		divisor = t.Numel()
	} else {
		for _, d := range dims {
			if d < 0 {
				d = len(shape) + d
			}
			if d >= 0 && d < len(shape) {
				divisor *= shape[d]
			}
		}
	}

	// Divide by count
	if cpu, ok := out.Storage().(tendo.CPUDataAccessor); ok {
		data := cpu.Data()
		for i := range data {
			data[i] /= float32(divisor)
		}
	}

	return out, nil
}

func (b *Backend) Max(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	return b.reduce(t, dims, keepdim, func(acc, val float32) float32 {
		if val > acc {
			return val
		}
		return acc
	}, float32(-math.MaxFloat32))
}

func (b *Backend) Min(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	return b.reduce(t, dims, keepdim, func(acc, val float32) float32 {
		if val < acc {
			return val
		}
		return acc
	}, float32(math.MaxFloat32))
}

func (b *Backend) Var(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool, correction int) (*tendo.Tensor, error) {
	return b.computeVariance(t, dims, keepdim, correction, false)
}

func (b *Backend) Std(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool, correction int) (*tendo.Tensor, error) {
	return b.computeVariance(t, dims, keepdim, correction, true)
}

func (b *Backend) Prod(_ context.Context, t *tendo.Tensor, dims []int, keepdim bool) (*tendo.Tensor, error) {
	return b.reduce(t, dims, keepdim, func(acc, val float32) float32 { return acc * val }, 1)
}

func (b *Backend) ArgMax(_ context.Context, t *tendo.Tensor, dim int, keepdim bool) (*tendo.Tensor, error) {
	return b.argReduce(t, dim, keepdim, true)
}

func (b *Backend) ArgMin(_ context.Context, t *tendo.Tensor, dim int, keepdim bool) (*tendo.Tensor, error) {
	return b.argReduce(t, dim, keepdim, false)
}

// --- ActivationOps ---

func (b *Backend) Softmax(_ context.Context, t *tendo.Tensor, dim int) (*tendo.Tensor, error) {
	return b.applySoftmax(t, dim, false)
}

func (b *Backend) LogSoftmax(_ context.Context, t *tendo.Tensor, dim int) (*tendo.Tensor, error) {
	return b.applySoftmax(t, dim, true)
}

func (b *Backend) LeakyReLU(_ context.Context, t *tendo.Tensor, negativeSlope float32) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		if x > 0 {
			return x
		}
		return negativeSlope * x
	})
}

func (b *Backend) Dropout(_ context.Context, t *tendo.Tensor, p float32, training bool) (*tendo.Tensor, error) {
	if !training {
		return t, nil
	}

	cpu, ok := t.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: t.Device().Type}
	}

	data := cpu.Data()
	numel := len(data)
	result := make([]float32, numel)
	scale := float32(1.0) / (1.0 - p)

	for i, v := range data {
		if rand.Float32() >= p {
			result[i] = v * scale
		} else {
			result[i] = 0
		}
	}

	outStorage := NewStorageFromSlice(result, t.DType())
	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// --- NormOps ---

func (b *Backend) BatchNorm2d(ctx context.Context, input, weight, bias, runningMean, runningVar *tendo.Tensor, epsilon, momentum float32, training bool) (*tendo.Tensor, error) {
	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, &tendo.ShapeError{Op: "batchnorm2d", Message: "input must be 4D [N, C, H, W]"}
	}

	N, C, H, W := inShape[0], inShape[1], inShape[2], inShape[3]
	spatialSize := H * W
	batchSpatialSize := N * spatialSize

	cpuInput, ok := input.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: input.Device().Type}
	}

	inputData := cpuInput.Data()
	result := make([]float32, len(inputData))

	weightData, err := weight.Data()
	if err != nil {
		return nil, err
	}
	biasData, err := bias.Data()
	if err != nil {
		return nil, err
	}

	// For each channel, normalize
	for c := 0; c < C; c++ {
		var mean, variance float32

		if training {
			// Compute batch mean for this channel
			sum := float32(0)
			for n := 0; n < N; n++ {
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						idx := n*C*H*W + c*H*W + h*W + w
						sum += inputData[idx]
					}
				}
			}
			mean = sum / float32(batchSpatialSize)

			// Compute batch variance for this channel
			sumSq := float32(0)
			for n := 0; n < N; n++ {
				for h := 0; h < H; h++ {
					for w := 0; w < W; w++ {
						idx := n*C*H*W + c*H*W + h*W + w
						diff := inputData[idx] - mean
						sumSq += diff * diff
					}
				}
			}
			variance = sumSq / float32(batchSpatialSize)
		} else {
			// Use running statistics in eval mode
			meanData, errMean := runningMean.Data()
			if errMean != nil {
				return nil, errMean
			}
			varData, errVar := runningVar.Data()
			if errVar != nil {
				return nil, errVar
			}
			mean = meanData[c]
			variance = varData[c]
		}

		gamma := weightData[c]
		beta := biasData[c]
		invStd := float32(1.0 / math.Sqrt(float64(variance+epsilon)))

		for n := 0; n < N; n++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					idx := n*C*H*W + c*H*W + h*W + w
					normalized := (inputData[idx] - mean) * invStd
					result[idx] = gamma*normalized + beta
				}
			}
		}
	}

	storage := NewStorageFromSlice(result, input.DType())
	return tendo.NewTensor(storage, inShape, nil), nil
}

func (b *Backend) LayerNorm(_ context.Context, input *tendo.Tensor, normalizedShape []int, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	cpu, ok := input.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: input.Device().Type}
	}

	data := cpu.Data()
	inShape := input.Shape()

	// Validate normalizedShape matches trailing dimensions of input
	if len(normalizedShape) > len(inShape) {
		return nil, &tendo.ShapeError{Op: "layernorm", Message: "normalizedShape has more dimensions than input"}
	}
	offset := len(inShape) - len(normalizedShape)
	for i, dim := range normalizedShape {
		if inShape[offset+i] != dim {
			return nil, &tendo.ShapeError{Op: "layernorm", Message: fmt.Sprintf("normalizedShape %v doesn't match input trailing dimensions %v", normalizedShape, inShape[offset:])}
		}
	}

	// Calculate the number of elements to normalize
	normSize := tendo.Numel(normalizedShape)
	outerSize := tendo.Numel(inShape) / normSize

	result := make([]float32, len(data))

	// Handle optional weight and bias (default: weight=1, bias=0)
	var weightData, biasData []float32
	if weight != nil {
		var err error
		weightData, err = weight.Data()
		if err != nil {
			return nil, err
		}
	}
	if bias != nil {
		var err error
		biasData, err = bias.Data()
		if err != nil {
			return nil, err
		}
	}

	for outer := 0; outer < outerSize; outer++ {
		offset := outer * normSize

		// Compute mean
		mean := float32(0)
		for i := 0; i < normSize; i++ {
			mean += data[offset+i]
		}
		mean /= float32(normSize)

		// Compute variance
		variance := float32(0)
		for i := 0; i < normSize; i++ {
			diff := data[offset+i] - mean
			variance += diff * diff
		}
		variance /= float32(normSize)

		invStd := float32(1.0 / math.Sqrt(float64(variance+epsilon)))

		// Normalize and apply weight/bias
		for i := 0; i < normSize; i++ {
			normalized := (data[offset+i] - mean) * invStd
			w := float32(1)
			if weightData != nil {
				w = weightData[i]
			}
			b := float32(0)
			if biasData != nil {
				b = biasData[i]
			}
			result[offset+i] = w*normalized + b
		}
	}

	storage := NewStorageFromSlice(result, input.DType())
	return tendo.NewTensor(storage, inShape, nil), nil
}

func (b *Backend) RMSNorm(_ context.Context, input *tendo.Tensor, normalizedShape []int, weight *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	cpu, ok := input.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: input.Device().Type}
	}

	data := cpu.Data()
	inShape := input.Shape()

	// Calculate norm size from normalizedShape
	normSize := tendo.Numel(normalizedShape)
	outerSize := tendo.Numel(inShape) / normSize

	result := make([]float32, len(data))

	var weightData []float32
	if weight != nil {
		var err error
		weightData, err = weight.Data()
		if err != nil {
			return nil, err
		}
	}

	for outer := 0; outer < outerSize; outer++ {
		offset := outer * normSize

		// Compute RMS: sqrt(mean(x^2) + eps)
		sumSq := float32(0)
		for i := 0; i < normSize; i++ {
			sumSq += data[offset+i] * data[offset+i]
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(normSize) + epsilon)))

		// Normalize: x / rms * weight
		for i := 0; i < normSize; i++ {
			normalized := data[offset+i] / rms
			if weightData != nil {
				normalized *= weightData[i]
			}
			result[offset+i] = normalized
		}
	}

	storage := NewStorageFromSlice(result, input.DType())
	return tendo.NewTensor(storage, inShape, nil), nil
}

func (b *Backend) GroupNorm(_ context.Context, input *tendo.Tensor, numGroups int, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	cpu, ok := input.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: input.Device().Type}
	}

	inShape := input.Shape()
	if len(inShape) < 2 {
		return nil, &tendo.ShapeError{Op: "groupnorm", Message: "input must have at least 2 dimensions [N, C, ...]"}
	}

	N, C := inShape[0], inShape[1]
	if C%numGroups != 0 {
		return nil, &tendo.ShapeError{Op: "groupnorm", Message: fmt.Sprintf("channels %d not divisible by numGroups %d", C, numGroups)}
	}

	channelsPerGroup := C / numGroups
	spatialSize := tendo.Numel(inShape[2:])
	if spatialSize == 0 {
		spatialSize = 1
	}

	data := cpu.Data()
	result := make([]float32, len(data))

	var weightData, biasData []float32
	if weight != nil {
		var err error
		weightData, err = weight.Data()
		if err != nil {
			return nil, err
		}
	}
	if bias != nil {
		var err error
		biasData, err = bias.Data()
		if err != nil {
			return nil, err
		}
	}

	for n := 0; n < N; n++ {
		for g := 0; g < numGroups; g++ {
			// Calculate mean and variance for this group
			groupSize := channelsPerGroup * spatialSize
			sum := float32(0)

			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := g*channelsPerGroup + c
				for s := 0; s < spatialSize; s++ {
					idx := n*C*spatialSize + channelIdx*spatialSize + s
					sum += data[idx]
				}
			}
			mean := sum / float32(groupSize)

			sumSq := float32(0)
			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := g*channelsPerGroup + c
				for s := 0; s < spatialSize; s++ {
					idx := n*C*spatialSize + channelIdx*spatialSize + s
					diff := data[idx] - mean
					sumSq += diff * diff
				}
			}
			variance := sumSq / float32(groupSize)
			invStd := float32(1.0 / math.Sqrt(float64(variance+epsilon)))

			// Normalize
			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := g*channelsPerGroup + c
				gamma := float32(1)
				beta := float32(0)
				if weightData != nil {
					gamma = weightData[channelIdx]
				}
				if biasData != nil {
					beta = biasData[channelIdx]
				}

				for s := 0; s < spatialSize; s++ {
					idx := n*C*spatialSize + channelIdx*spatialSize + s
					normalized := (data[idx] - mean) * invStd
					result[idx] = gamma*normalized + beta
				}
			}
		}
	}

	storage := NewStorageFromSlice(result, input.DType())
	return tendo.NewTensor(storage, inShape, nil), nil
}

func (b *Backend) InstanceNorm2d(_ context.Context, input, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	cpu, ok := input.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: input.Device().Type}
	}

	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, &tendo.ShapeError{Op: "instancenorm2d", Message: "input must be 4D [N, C, H, W]"}
	}

	N, C, H, W := inShape[0], inShape[1], inShape[2], inShape[3]
	spatialSize := H * W

	data := cpu.Data()
	result := make([]float32, len(data))

	var weightData, biasData []float32
	if weight != nil {
		var err error
		weightData, err = weight.Data()
		if err != nil {
			return nil, err
		}
	}
	if bias != nil {
		var err error
		biasData, err = bias.Data()
		if err != nil {
			return nil, err
		}
	}

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			// Compute mean over spatial dimensions
			sum := float32(0)
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					idx := n*C*H*W + c*H*W + h*W + w
					sum += data[idx]
				}
			}
			mean := sum / float32(spatialSize)

			// Compute variance
			sumSq := float32(0)
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					idx := n*C*H*W + c*H*W + h*W + w
					diff := data[idx] - mean
					sumSq += diff * diff
				}
			}
			variance := sumSq / float32(spatialSize)
			invStd := float32(1.0 / math.Sqrt(float64(variance+epsilon)))

			gamma := float32(1)
			beta := float32(0)
			if weightData != nil {
				gamma = weightData[c]
			}
			if biasData != nil {
				beta = biasData[c]
			}

			// Normalize
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					idx := n*C*H*W + c*H*W + h*W + w
					normalized := (data[idx] - mean) * invStd
					result[idx] = gamma*normalized + beta
				}
			}
		}
	}

	storage := NewStorageFromSlice(result, input.DType())
	return tendo.NewTensor(storage, inShape, nil), nil
}

// --- ConvOps ---

func (b *Backend) Conv2d(_ context.Context, input, weight *tendo.Tensor, padding, stride, dilation [2]int, groups int) (*tendo.Tensor, error) {
	inShape := input.Shape()
	wShape := weight.Shape()

	if len(inShape) != 4 || len(wShape) != 4 {
		return nil, &tendo.ShapeError{Op: "conv2d", Message: "input and weight must be 4D"}
	}

	N, inC, inH, inW := inShape[0], inShape[1], inShape[2], inShape[3]
	outC, inCPerGroup, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]

	// Validate channel dimensions
	if inC != inCPerGroup*groups {
		return nil, &tendo.ShapeError{
			Op:      "conv2d",
			Message: fmt.Sprintf("input channels %d don't match weight's expected channels %d (groups=%d)", inC, inCPerGroup*groups, groups),
		}
	}

	outH := (inH+2*padding[0]-dilation[0]*(kH-1)-1)/stride[0] + 1
	outW := (inW+2*padding[1]-dilation[1]*(kW-1)-1)/stride[1] + 1

	cpuInput, ok := input.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: input.Device().Type}
	}
	cpuWeight, ok := weight.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: weight.Device().Type}
	}
	inputData := cpuInput.Data()
	weightData := cpuWeight.Data()

	outCPerGroup := outC / groups
	result := make([]float32, N*outC*outH*outW)

	// GEMM dimensions for im2col approach
	colRows := inCPerGroup * kH * kW
	colCols := outH * outW

	// Pre-allocate buffers for reuse across batches/groups
	colBuffer := make([]float32, colRows*colCols)
	gemmOutput := make([]float32, outCPerGroup*colCols)

	for n := 0; n < N; n++ {
		for g := 0; g < groups; g++ {
			// Extract group's input slice: [inCPerGroup, inH, inW]
			inputOffset := n*inC*inH*inW + g*inCPerGroup*inH*inW
			groupInput := inputData[inputOffset : inputOffset+inCPerGroup*inH*inW]

			// Transform input patches to columns
			im2colInPlace(groupInput, colBuffer,
				inCPerGroup, inH, inW, kH, kW,
				padding[0], padding[1],
				stride[0], stride[1],
				dilation[0], dilation[1])

			// Weight slice for this group: [outCPerGroup, inCPerGroup * kH * kW]
			weightOffset := g * outCPerGroup * inCPerGroup * kH * kW
			groupWeight := weightData[weightOffset : weightOffset+outCPerGroup*inCPerGroup*kH*kW]

			// GEMM: [outCPerGroup, colRows] @ [colRows, colCols] = [outCPerGroup, colCols]
			blasMatMul2D(groupWeight, colBuffer, gemmOutput, outCPerGroup, colRows, colCols)

			// Copy result to output tensor
			for oc := 0; oc < outCPerGroup; oc++ {
				srcOffset := oc * colCols
				dstOffset := n*outC*outH*outW + (g*outCPerGroup+oc)*outH*outW
				copy(result[dstOffset:dstOffset+colCols], gemmOutput[srcOffset:srcOffset+colCols])
			}
		}
	}

	storage := NewStorageFromSlice(result, input.DType())
	return tendo.NewTensor(storage, []int{N, outC, outH, outW}, nil), nil
}

func (b *Backend) ConvTranspose2d(_ context.Context, input, weight *tendo.Tensor, padding, outputPadding, stride, dilation [2]int, groups int) (*tendo.Tensor, error) {
	inShape := input.Shape()
	wShape := weight.Shape()

	if len(inShape) != 4 || len(wShape) != 4 {
		return nil, &tendo.ShapeError{Op: "convtranspose2d", Message: "input and weight must be 4D"}
	}

	N, inC, inH, inW := inShape[0], inShape[1], inShape[2], inShape[3]
	_, outCPerGroup, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]

	outC := outCPerGroup * groups
	inCPerGroup := inC / groups

	// Calculate output size for transposed convolution
	outH := (inH-1)*stride[0] - 2*padding[0] + dilation[0]*(kH-1) + outputPadding[0] + 1
	outW := (inW-1)*stride[1] - 2*padding[1] + dilation[1]*(kW-1) + outputPadding[1] + 1

	cpuInput := input.Storage().(tendo.CPUDataAccessor)
	cpuWeight := weight.Storage().(tendo.CPUDataAccessor)
	inputData := cpuInput.Data()
	weightData := cpuWeight.Data()

	result := make([]float32, N*outC*outH*outW)

	// ConvTranspose2d: scatter input values through weight kernel
	for n := 0; n < N; n++ {
		for g := 0; g < groups; g++ {
			for ic := 0; ic < inCPerGroup; ic++ {
				inChannelIdx := g*inCPerGroup + ic
				for oc := 0; oc < outCPerGroup; oc++ {
					outChannelIdx := g*outCPerGroup + oc

					for ih := 0; ih < inH; ih++ {
						for iw := 0; iw < inW; iw++ {
							inVal := inputData[n*inC*inH*inW+inChannelIdx*inH*inW+ih*inW+iw]

							for kh := 0; kh < kH; kh++ {
								for kw := 0; kw < kW; kw++ {
									oh := ih*stride[0] - padding[0] + kh*dilation[0]
									ow := iw*stride[1] - padding[1] + kw*dilation[1]

									if oh >= 0 && oh < outH && ow >= 0 && ow < outW {
										// Weight layout: [inC, outC/groups, kH, kW]
										wIdx := inChannelIdx*outCPerGroup*kH*kW + oc*kH*kW + kh*kW + kw
										outIdx := n*outC*outH*outW + outChannelIdx*outH*outW + oh*outW + ow
										result[outIdx] += inVal * weightData[wIdx]
									}
								}
							}
						}
					}
				}
			}
		}
	}

	storage := NewStorageFromSlice(result, input.DType())
	return tendo.NewTensor(storage, []int{N, outC, outH, outW}, nil), nil
}

// --- PoolOps ---

func (b *Backend) MaxPool2d(_ context.Context, input *tendo.Tensor, kernelSize, stride, padding [2]int) (*tendo.Tensor, error) {
	return b.pool2d(input, kernelSize, stride, padding, true)
}

func (b *Backend) AvgPool2d(_ context.Context, input *tendo.Tensor, kernelSize, stride, padding [2]int) (*tendo.Tensor, error) {
	return b.pool2d(input, kernelSize, stride, padding, false)
}

func (b *Backend) AdaptiveAvgPool2d(_ context.Context, input *tendo.Tensor, outputSize [2]int) (*tendo.Tensor, error) {
	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, &tendo.ShapeError{Op: "adaptiveavgpool2d", Message: "input must be 4D [N, C, H, W]"}
	}

	N, C, inH, inW := inShape[0], inShape[1], inShape[2], inShape[3]
	outH, outW := outputSize[0], outputSize[1]

	cpu := input.Storage().(tendo.CPUDataAccessor)
	data := cpu.Data()
	result := make([]float32, N*C*outH*outW)

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					// Calculate input region for this output element
					ihStart := oh * inH / outH
					ihEnd := (oh+1)*inH/outH
					if ihEnd <= ihStart {
						ihEnd = ihStart + 1
					}
					iwStart := ow * inW / outW
					iwEnd := (ow+1)*inW/outW
					if iwEnd <= iwStart {
						iwEnd = iwStart + 1
					}

					// Average pool over this region
					sum := float32(0)
					count := 0
					for ih := ihStart; ih < ihEnd; ih++ {
						for iw := iwStart; iw < iwEnd; iw++ {
							sum += data[n*C*inH*inW+c*inH*inW+ih*inW+iw]
							count++
						}
					}

					result[n*C*outH*outW+c*outH*outW+oh*outW+ow] = sum / float32(count)
				}
			}
		}
	}

	storage := NewStorageFromSlice(result, input.DType())
	return tendo.NewTensor(storage, []int{N, C, outH, outW}, nil), nil
}

func (b *Backend) AdaptiveMaxPool2d(_ context.Context, input *tendo.Tensor, outputSize [2]int) (*tendo.Tensor, error) {
	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, &tendo.ShapeError{Op: "adaptivemaxpool2d", Message: "input must be 4D [N, C, H, W]"}
	}

	N, C, inH, inW := inShape[0], inShape[1], inShape[2], inShape[3]
	outH, outW := outputSize[0], outputSize[1]

	cpu := input.Storage().(tendo.CPUDataAccessor)
	data := cpu.Data()
	result := make([]float32, N*C*outH*outW)

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					// Calculate input region for this output element
					ihStart := oh * inH / outH
					ihEnd := (oh+1)*inH/outH
					if ihEnd <= ihStart {
						ihEnd = ihStart + 1
					}
					iwStart := ow * inW / outW
					iwEnd := (ow+1)*inW/outW
					if iwEnd <= iwStart {
						iwEnd = iwStart + 1
					}

					// Max pool over this region
					maxVal := data[n*C*inH*inW+c*inH*inW+ihStart*inW+iwStart]
					for ih := ihStart; ih < ihEnd; ih++ {
						for iw := iwStart; iw < iwEnd; iw++ {
							val := data[n*C*inH*inW+c*inH*inW+ih*inW+iw]
							if val > maxVal {
								maxVal = val
							}
						}
					}

					outIdx := n*C*outH*outW + c*outH*outW + oh*outW + ow
					result[outIdx] = maxVal
				}
			}
		}
	}

	storage := NewStorageFromSlice(result, input.DType())
	return tendo.NewTensor(storage, []int{N, C, outH, outW}, nil), nil
}

// --- LossOps ---

func (b *Backend) MSELoss(ctx context.Context, input, target *tendo.Tensor, reduction string) (*tendo.Tensor, error) {
	diff, err := b.Sub(ctx, input, target)
	if err != nil {
		return nil, err
	}
	squared, err := b.Square(ctx, diff)
	if err != nil {
		return nil, err
	}

	switch reduction {
	case reductionNone:
		return squared, nil
	case reductionSum:
		return b.Sum(ctx, squared, nil, false)
	case reductionMean:
		return b.Mean(ctx, squared, nil, false)
	default:
		return b.Mean(ctx, squared, nil, false)
	}
}

func (b *Backend) L1Loss(ctx context.Context, input, target *tendo.Tensor, reduction string) (*tendo.Tensor, error) {
	diff, err := b.Sub(ctx, input, target)
	if err != nil {
		return nil, err
	}
	absVal, err := b.Abs(ctx, diff)
	if err != nil {
		return nil, err
	}

	switch reduction {
	case reductionNone:
		return absVal, nil
	case reductionSum:
		return b.Sum(ctx, absVal, nil, false)
	case reductionMean:
		return b.Mean(ctx, absVal, nil, false)
	default:
		return b.Mean(ctx, absVal, nil, false)
	}
}

func (b *Backend) CrossEntropyLoss(ctx context.Context, input, target *tendo.Tensor, reduction string) (*tendo.Tensor, error) {
	// input: [N, C] logits
	// target: [N] class indices
	logSoftmax, err := b.LogSoftmax(ctx, input, -1)
	if err != nil {
		return nil, err
	}
	return b.NLLLoss(ctx, logSoftmax, target, reduction)
}

func (b *Backend) NLLLoss(_ context.Context, input, target *tendo.Tensor, reduction string) (*tendo.Tensor, error) {
	inShape := input.Shape()
	if len(inShape) != 2 {
		return nil, &tendo.ShapeError{Op: "nllloss", Message: "input must be 2D [N, C]"}
	}

	N, C := inShape[0], inShape[1]

	cpuInput := input.Storage().(tendo.CPUDataAccessor)
	cpuTarget := target.Storage().(tendo.CPUDataAccessor)

	inputData := cpuInput.Data()
	targetData := cpuTarget.Data()

	losses := make([]float32, N)
	for n := 0; n < N; n++ {
		classIdx := int(targetData[n])
		if classIdx >= 0 && classIdx < C {
			losses[n] = -inputData[n*C+classIdx]
		}
	}

	switch reduction {
	case reductionNone:
		storage := NewStorageFromSlice(losses, input.DType())
		return tendo.NewTensor(storage, []int{N}, nil), nil
	case reductionSum:
		sum := float32(0)
		for _, l := range losses {
			sum += l
		}
		storage := NewStorageFromSlice([]float32{sum}, input.DType())
		return tendo.NewTensor(storage, []int{}, nil), nil
	default: // mean
		sum := float32(0)
		for _, l := range losses {
			sum += l
		}
		storage := NewStorageFromSlice([]float32{sum / float32(N)}, input.DType())
		return tendo.NewTensor(storage, []int{}, nil), nil
	}
}

// --- CompareOps ---

func (b *Backend) Clamp(_ context.Context, t *tendo.Tensor, minVal, maxVal float32) (*tendo.Tensor, error) {
	return b.unaryElementwise(t, func(x float32) float32 {
		if x < minVal {
			return minVal
		}
		if x > maxVal {
			return maxVal
		}
		return x
	})
}

func (b *Backend) Where(_ context.Context, condition, x, y *tendo.Tensor) (*tendo.Tensor, error) {
	cpuCond, ok := condition.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: condition.Device().Type}
	}
	cpuX, ok := x.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: x.Device().Type}
	}
	cpuY, ok := y.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: y.Device().Type}
	}

	// Compute broadcast output shape
	outShape, err := tendo.BroadcastShapes(condition.Shape(), x.Shape())
	if err != nil {
		return nil, err
	}
	outShape, err = tendo.BroadcastShapes(outShape, y.Shape())
	if err != nil {
		return nil, err
	}

	outNumel := tendo.Numel(outShape)
	result := make([]float32, outNumel)

	condData := cpuCond.Data()
	xData := cpuX.Data()
	yData := cpuY.Data()

	// Compute broadcast strides for each input
	stridesCond := broadcastStridesWithActual(condition.Shape(), condition.Stride(), outShape)
	stridesX := broadcastStridesWithActual(x.Shape(), x.Stride(), outShape)
	stridesY := broadcastStridesWithActual(y.Shape(), y.Stride(), outShape)
	outStrides := tendo.ComputeStrides(outShape)

	offsetCond := condition.Offset()
	offsetX := x.Offset()
	offsetY := y.Offset()

	for i := 0; i < outNumel; i++ {
		// Convert flat index to per-input indices using broadcast strides
		idxCond := offsetCond
		idxX := offsetX
		idxY := offsetY
		tmp := i
		for d := 0; d < len(outShape); d++ {
			coord := tmp / outStrides[d]
			tmp %= outStrides[d]
			idxCond += coord * stridesCond[d]
			idxX += coord * stridesX[d]
			idxY += coord * stridesY[d]
		}

		if condData[idxCond] > 0 {
			result[i] = xData[idxX]
		} else {
			result[i] = yData[idxY]
		}
	}

	storage := NewStorageFromSlice(result, x.DType())
	return tendo.NewTensor(storage, outShape, nil), nil
}

func (b *Backend) Tril(_ context.Context, t *tendo.Tensor, k int) (*tendo.Tensor, error) {
	cpu, ok := t.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: t.Device().Type}
	}

	shape := t.Shape()
	if len(shape) != 2 {
		return nil, &tendo.ShapeError{Op: "tril", Message: "input must be 2D"}
	}
	rows, cols := shape[0], shape[1]

	data := cpu.Data()
	result := make([]float32, rows*cols)

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			idx := row*cols + col
			if col <= row+k {
				result[idx] = data[idx]
			} else {
				result[idx] = 0
			}
		}
	}

	storage := NewStorageFromSlice(result, t.DType())
	return tendo.NewTensor(storage, shape, nil), nil
}

// --- EmbeddingOps ---

func (b *Backend) Embedding(_ context.Context, weight, indices *tendo.Tensor) (*tendo.Tensor, error) {
	cpuWeight, ok := weight.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: weight.Device().Type}
	}

	// weight shape: [vocab_size, embed_dim]
	weightShape := weight.Shape()
	if len(weightShape) != 2 {
		return nil, &tendo.ShapeError{Op: "embedding", Message: "weight must be 2D [vocab_size, embed_dim]"}
	}
	embedDim := weightShape[1]
	n := indices.Numel()

	// Output shape: indices.shape + [embed_dim]
	outShape := append(indices.Shape(), embedDim)
	outNumel := tendo.Numel(outShape)
	result := make([]float32, outNumel)

	weightData := cpuWeight.Data()

	// Handle both int64 and float32 indices (float32 for backward compatibility)
	if indices.DType() == tendo.Int64 {
		int64Accessor, ok := indices.Storage().(tendo.CPUInt64DataAccessor)
		if !ok {
			return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: indices.Device().Type}
		}
		indicesData := int64Accessor.Int64Data()
		for i := 0; i < n; i++ {
			idx := int(indicesData[i])
			srcOffset := idx * embedDim
			dstOffset := i * embedDim
			copy(result[dstOffset:dstOffset+embedDim], weightData[srcOffset:srcOffset+embedDim])
		}
	} else {
		floatAccessor, ok := indices.Storage().(tendo.CPUDataAccessor)
		if !ok {
			return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: indices.Device().Type}
		}
		indicesData := floatAccessor.Data()
		for i := 0; i < n; i++ {
			idx := int(indicesData[i])
			srcOffset := idx * embedDim
			dstOffset := i * embedDim
			copy(result[dstOffset:dstOffset+embedDim], weightData[srcOffset:srcOffset+embedDim])
		}
	}

	storage := NewStorageFromSlice(result, weight.DType())
	return tendo.NewTensor(storage, outShape, nil), nil
}

// --- ShapeOps ---

func (b *Backend) Cat(_ context.Context, tensors []*tendo.Tensor, dim int) (*tendo.Tensor, error) {
	if len(tensors) == 0 {
		return nil, &tendo.ShapeError{Op: "cat", Message: "cat requires at least one tensor"}
	}

	shape := tensors[0].Shape()
	ndim := len(shape)
	if dim < 0 {
		dim = ndim + dim
	}
	if dim < 0 || dim >= ndim {
		return nil, &tendo.ShapeError{Op: "cat", Message: "dimension out of range"}
	}

	// Validate all tensors have compatible shapes
	totalSize := 0
	for _, tensor := range tensors {
		tShape := tensor.Shape()
		if len(tShape) != ndim {
			return nil, &tendo.ShapeError{Op: "cat", Message: "all tensors must have same number of dimensions"}
		}
		for d := 0; d < ndim; d++ {
			if d == dim {
				totalSize += tShape[d]
			} else if tShape[d] != shape[d] {
				return nil, &tendo.ShapeError{Op: "cat", Message: "tensors must match in non-cat dimensions"}
			}
		}
	}

	// Compute output shape
	outShape := make([]int, ndim)
	copy(outShape, shape)
	outShape[dim] = totalSize

	// Allocate output storage
	outNumel := tendo.Numel(outShape)
	outData := make([]float32, outNumel)

	// Copy data from each tensor
	outStrides := tendo.ComputeStrides(outShape)
	offset := 0
	for _, tensor := range tensors {
		src := tensor.Contiguous()
		srcCPU, ok := src.Storage().(tendo.CPUDataAccessor)
		if !ok {
			return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: tensor.Device().Type}
		}
		srcData := srcCPU.Data()
		srcShape := src.Shape()

		// Copy this tensor's data into the output
		srcNumel := tendo.Numel(srcShape)
		for i := 0; i < srcNumel; i++ {
			coords := tendo.FlatToCoords(i, srcShape)
			coords[dim] += offset
			dstIdx := tendo.CoordsToFlat(coords, outStrides)
			outData[dstIdx] = srcData[i]
		}
		offset += srcShape[dim]
	}

	storage := NewStorageFromSlice(outData, tensors[0].DType())
	return tendo.NewTensor(storage, outShape, nil), nil
}

func (b *Backend) Stack(_ context.Context, tensors []*tendo.Tensor, dim int) (*tendo.Tensor, error) {
	if len(tensors) == 0 {
		return nil, &tendo.ShapeError{Op: "stack", Message: "stack requires at least one tensor"}
	}

	shape := tensors[0].Shape()
	ndim := len(shape)
	if dim < 0 {
		dim = ndim + dim + 1
	}
	if dim < 0 || dim > ndim {
		return nil, &tendo.ShapeError{Op: "stack", Message: "dimension out of range"}
	}

	// Validate all tensors have identical shapes
	for i, tensor := range tensors {
		tShape := tensor.Shape()
		if len(tShape) != ndim {
			return nil, &tendo.ShapeError{Op: "stack", Message: "all tensors must have same number of dimensions"}
		}
		for d := 0; d < ndim; d++ {
			if tShape[d] != shape[d] {
				return nil, &tendo.ShapeError{
					Op:      "stack",
					Message: fmt.Sprintf("tensor %d has different shape", i),
				}
			}
		}
	}

	// Compute output shape: insert new dimension at 'dim'
	outShape := make([]int, ndim+1)
	for d := 0; d < dim; d++ {
		outShape[d] = shape[d]
	}
	outShape[dim] = len(tensors)
	for d := dim; d < ndim; d++ {
		outShape[d+1] = shape[d]
	}

	// Allocate output storage
	outNumel := tendo.Numel(outShape)
	outData := make([]float32, outNumel)

	// Copy data from each tensor
	outStrides := tendo.ComputeStrides(outShape)
	tensorNumel := tendo.Numel(shape)
	srcStrides := tendo.ComputeStrides(shape)

	for ti, tensor := range tensors {
		src := tensor.Contiguous()
		srcCPU, ok := src.Storage().(tendo.CPUDataAccessor)
		if !ok {
			return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: tensor.Device().Type}
		}
		srcData := srcCPU.Data()

		for i := 0; i < tensorNumel; i++ {
			coords := tendo.FlatToCoords(i, shape)

			// Build output coordinates with new dimension inserted at dim
			outCoords := make([]int, ndim+1)
			copy(outCoords[:dim], coords[:dim])
			outCoords[dim] = ti
			copy(outCoords[dim+1:], coords[dim:])

			srcIdx := tendo.CoordsToFlat(coords, srcStrides)
			dstIdx := tendo.CoordsToFlat(outCoords, outStrides)
			outData[dstIdx] = srcData[srcIdx]
		}
	}

	storage := NewStorageFromSlice(outData, tensors[0].DType())
	return tendo.NewTensor(storage, outShape, nil), nil
}

// --- Helper methods ---

func (b *Backend) unaryElementwise(t *tendo.Tensor, fn func(float32) float32) (*tendo.Tensor, error) {
	cpu, ok := t.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: t.Device().Type}
	}

	data := cpu.Data()
	numel := t.Numel()
	result := make([]float32, numel)

	if t.IsContiguous() && t.Offset() == 0 {
		for i := 0; i < numel; i++ {
			result[i] = fn(data[i])
		}
	} else {
		tendo.IterateStrided(numel, t.Shape(), t.Stride(), t.Offset(), func(i, srcIdx int) {
			result[i] = fn(data[srcIdx])
		})
	}

	storage := NewStorageFromSlice(result, t.DType())
	return tendo.NewTensor(storage, t.Shape(), nil), nil
}

func (b *Backend) binaryElementwise(a, other *tendo.Tensor, fn func(float32, float32) float32) (*tendo.Tensor, error) {
	cpuA, ok := a.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: a.Device().Type}
	}
	cpuB, ok := other.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: other.Device().Type}
	}

	outShape, err := tendo.BroadcastShapes(a.Shape(), other.Shape())
	if err != nil {
		return nil, err
	}

	outNumel := tendo.Numel(outShape)
	result := make([]float32, outNumel)
	dataA := cpuA.Data()
	dataB := cpuB.Data()

	// Simple case: same shape, both contiguous with no offset
	if shapesEqual(a.Shape(), other.Shape()) && a.IsContiguous() && other.IsContiguous() && a.Offset() == 0 && other.Offset() == 0 {
		for i := range result {
			result[i] = fn(dataA[i], dataB[i])
		}
	} else {
		// General case: handle broadcasting, non-contiguous tensors, and offsets
		stridesA := broadcastStridesWithActual(a.Shape(), a.Stride(), outShape)
		stridesB := broadcastStridesWithActual(other.Shape(), other.Stride(), outShape)
		outStrides := tendo.ComputeStrides(outShape)
		offsetA := a.Offset()
		offsetB := other.Offset()

		for i := 0; i < outNumel; i++ {
			idxA := offsetA
			idxB := offsetB
			tmp := i
			for d := 0; d < len(outShape); d++ {
				coord := tmp / outStrides[d]
				tmp %= outStrides[d]
				idxA += coord * stridesA[d]
				idxB += coord * stridesB[d]
			}
			result[i] = fn(dataA[idxA], dataB[idxB])
		}
	}

	storage := NewStorageFromSlice(result, a.DType())
	return tendo.NewTensor(storage, outShape, nil), nil
}

func (b *Backend) reduce(t *tendo.Tensor, dims []int, keepdim bool, fn func(float32, float32) float32, init float32) (*tendo.Tensor, error) {
	cpu, ok := t.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: t.Device().Type}
	}

	shape := t.Shape()
	data := cpu.Data()

	// Normalize and validate dimensions
	normDims := make(map[int]bool)
	for _, d := range dims {
		if d < 0 {
			d = len(shape) + d
		}
		if d < 0 || d >= len(shape) {
			return nil, &tendo.ShapeError{Op: "reduce", Message: fmt.Sprintf("dimension %d out of range for %d-dimensional tensor", d, len(shape))}
		}
		normDims[d] = true
	}

	// If no dims, reduce all
	if len(dims) == 0 {
		result := init
		for _, v := range data {
			result = fn(result, v)
		}
		storage := NewStorageFromSlice([]float32{result}, t.DType())
		return tendo.NewTensor(storage, []int{}, nil), nil
	}

	// Compute output shape
	var outShape []int
	for i, s := range shape {
		if normDims[i] {
			if keepdim {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, s)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{}
	}

	outNumel := tendo.Numel(outShape)
	if outNumel == 0 {
		outNumel = 1
	}

	result := make([]float32, outNumel)
	for i := range result {
		result[i] = init
	}

	strides := tendo.ComputeStrides(shape)
	outStrides := tendo.ComputeStrides(outShape)

	for i := 0; i < len(data); i++ {
		outIdx := tendo.FlatToReducedIndex(i, shape, strides, outStrides, normDims)
		result[outIdx] = fn(result[outIdx], data[i])
	}

	storage := NewStorageFromSlice(result, t.DType())
	return tendo.NewTensor(storage, outShape, nil), nil
}

func (b *Backend) argReduce(t *tendo.Tensor, dim int, keepdim bool, findMax bool) (*tendo.Tensor, error) {
	cpu, ok := t.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: t.Device().Type}
	}

	shape := t.Shape()
	if dim < 0 {
		dim = len(shape) + dim
	}

	data := cpu.Data()
	strides := tendo.ComputeStrides(shape)
	dimSize := shape[dim]

	// Output shape
	var outShape []int
	for i, s := range shape {
		if i == dim {
			if keepdim {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, s)
		}
	}

	outNumel := tendo.Numel(outShape)
	if outNumel == 0 {
		outNumel = 1
	}

	result := make([]int64, outNumel)
	resultIdx := 0

	tendo.IterateSlices(shape, strides, dim, func(baseIdx, dimStride int) {
		bestIdx := 0
		bestVal := data[baseIdx]
		for j := 1; j < dimSize; j++ {
			val := data[baseIdx+j*dimStride]
			if findMax {
				if val > bestVal {
					bestVal = val
					bestIdx = j
				}
			} else {
				if val < bestVal {
					bestVal = val
					bestIdx = j
				}
			}
		}
		result[resultIdx] = int64(bestIdx)
		resultIdx++
	})

	storage := NewInt64StorageFromSlice(result)
	return tendo.NewTensor(storage, outShape, nil), nil
}

func (b *Backend) applySoftmax(t *tendo.Tensor, dim int, logSoftmax bool) (*tendo.Tensor, error) {
	cpu, ok := t.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: t.Device().Type}
	}

	shape := t.Shape()
	if dim < 0 {
		dim = len(shape) + dim
	}
	if dim < 0 || dim >= len(shape) {
		return nil, &tendo.ShapeError{Op: "softmax", Message: fmt.Sprintf("dimension %d out of range for %d-dimensional tensor", dim, len(shape))}
	}

	data := cpu.Data()
	result := make([]float32, len(data))
	copy(result, data)

	strides := t.Stride()
	dimSize := shape[dim]

	tendo.IterateSlices(shape, strides, dim, func(baseIdx, dimStride int) {
		// Find max for numerical stability
		maxVal := float32(-math.MaxFloat32)
		for j := 0; j < dimSize; j++ {
			idx := baseIdx + j*dimStride
			if data[idx] > maxVal {
				maxVal = data[idx]
			}
		}

		// Compute exp(x - max) and sum
		sumExp := float32(0)
		for j := 0; j < dimSize; j++ {
			idx := baseIdx + j*dimStride
			expVal := float32(math.Exp(float64(data[idx] - maxVal)))
			result[idx] = expVal
			sumExp += expVal
		}

		// Normalize
		logSumExp := float32(math.Log(float64(sumExp))) + maxVal
		for j := 0; j < dimSize; j++ {
			idx := baseIdx + j*dimStride
			if logSoftmax {
				result[idx] = data[idx] - logSumExp
			} else {
				result[idx] /= sumExp
			}
		}
	})

	storage := NewStorageFromSlice(result, t.DType())
	return tendo.NewTensor(storage, shape, nil), nil
}

func (b *Backend) computeVariance(t *tendo.Tensor, dims []int, keepdim bool, correction int, sqrt bool) (*tendo.Tensor, error) {
	cpu, ok := t.Storage().(tendo.CPUDataAccessor)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CPU, Got: t.Device().Type}
	}

	shape := t.Shape()
	data := cpu.Data()

	// Calculate divisor
	divisor := 1
	if len(dims) == 0 {
		divisor = t.Numel()
	} else {
		for _, d := range dims {
			if d < 0 {
				d = len(shape) + d
			}
			if d >= 0 && d < len(shape) {
				divisor *= shape[d]
			}
		}
	}

	correctedDivisor := divisor - correction
	if correctedDivisor < 1 {
		correctedDivisor = 1
	}

	// All elements case
	if len(dims) == 0 {
		sum := float32(0)
		for _, v := range data {
			sum += v
		}
		mean := sum / float32(divisor)

		variance := float32(0)
		for _, v := range data {
			diff := v - mean
			variance += diff * diff
		}
		variance /= float32(correctedDivisor)

		if sqrt {
			variance = float32(math.Sqrt(float64(variance)))
		}

		storage := NewStorageFromSlice([]float32{variance}, t.DType())
		return tendo.NewTensor(storage, []int{}, nil), nil
	}

	// First compute mean
	ctx := context.Background()
	meanTensor, err := b.Mean(ctx, t, dims, keepdim)
	if err != nil {
		return nil, err
	}

	// Compute variance
	normDims := make(map[int]bool)
	for _, d := range dims {
		if d < 0 {
			d = len(shape) + d
		}
		normDims[d] = true
	}

	var outShape []int
	for i, s := range shape {
		if normDims[i] {
			if keepdim {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, s)
		}
	}

	outNumel := tendo.Numel(outShape)
	if outNumel == 0 {
		outNumel = 1
	}

	variances := make([]float32, outNumel)
	strides := tendo.ComputeStrides(shape)
	outStrides := tendo.ComputeStrides(outShape)
	meanData, err := meanTensor.Data()
	if err != nil {
		return nil, err
	}

	for i := 0; i < len(data); i++ {
		outIdx := tendo.FlatToReducedIndex(i, shape, strides, outStrides, normDims)
		diff := data[i] - meanData[outIdx]
		variances[outIdx] += diff * diff
	}

	for i := range variances {
		variances[i] /= float32(correctedDivisor)
		if sqrt {
			variances[i] = float32(math.Sqrt(float64(variances[i])))
		}
	}

	storage := NewStorageFromSlice(variances, t.DType())
	return tendo.NewTensor(storage, outShape, nil), nil
}

func (b *Backend) pool2d(input *tendo.Tensor, kernelSize, stride, padding [2]int, maxPool bool) (*tendo.Tensor, error) {
	inShape := input.Shape()

	// Support both 3D [C, H, W] and 4D [N, C, H, W] input
	var N, C, inH, inW int
	is3D := false
	if len(inShape) == 3 {
		is3D = true
		N, C, inH, inW = 1, inShape[0], inShape[1], inShape[2]
	} else if len(inShape) == 4 {
		N, C, inH, inW = inShape[0], inShape[1], inShape[2], inShape[3]
	} else {
		return nil, &tendo.ShapeError{Op: "pool2d", Message: "input must be 3D [C, H, W] or 4D [N, C, H, W]"}
	}

	outH := (inH+2*padding[0]-kernelSize[0])/stride[0] + 1
	outW := (inW+2*padding[1]-kernelSize[1])/stride[1] + 1

	cpu := input.Storage().(tendo.CPUDataAccessor)
	data := cpu.Data()

	result := make([]float32, N*C*outH*outW)

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					var poolVal float32
					if maxPool {
						poolVal = float32(-math.MaxFloat32)
					} else {
						poolVal = 0
					}
					count := 0

					for kh := 0; kh < kernelSize[0]; kh++ {
						for kw := 0; kw < kernelSize[1]; kw++ {
							ih := oh*stride[0] - padding[0] + kh
							iw := ow*stride[1] - padding[1] + kw

							if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
								idx := n*C*inH*inW + c*inH*inW + ih*inW + iw
								if maxPool {
									if data[idx] > poolVal {
										poolVal = data[idx]
									}
								} else {
									poolVal += data[idx]
								}
								count++
							}
						}
					}

					if !maxPool && count > 0 {
						poolVal /= float32(count)
					}

					outIdx := n*C*outH*outW + c*outH*outW + oh*outW + ow
					result[outIdx] = poolVal
				}
			}
		}
	}

	storage := NewStorageFromSlice(result, input.DType())

	// Return 3D output if input was 3D
	var outShape []int
	if is3D {
		outShape = []int{C, outH, outW}
	} else {
		outShape = []int{N, C, outH, outW}
	}

	return tendo.NewTensor(storage, outShape, nil), nil
}

// Helper functions

// im2colInPlace fills colBuffer with the im2col transformation of input.
// colBuffer must be pre-allocated with size [inC * kH * kW, outH * outW].
// This transforms overlapping input patches into columns for convolution via GEMM.
func im2colInPlace(
	input []float32,
	colBuffer []float32,
	inC, inH, inW int,
	kH, kW int,
	padH, padW int,
	strideH, strideW int,
	dilationH, dilationW int,
) {
	outH := (inH+2*padH-dilationH*(kH-1)-1)/strideH + 1
	outW := (inW+2*padW-dilationW*(kW-1)-1)/strideW + 1
	colCols := outH * outW

	// Clear buffer (implicit zero-padding)
	for i := range colBuffer {
		colBuffer[i] = 0
	}

	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			colIdx := oh*outW + ow
			for ic := 0; ic < inC; ic++ {
				for kh := 0; kh < kH; kh++ {
					for kw := 0; kw < kW; kw++ {
						ih := oh*strideH - padH + kh*dilationH
						iw := ow*strideW - padW + kw*dilationW

						if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
							rowIdx := ic*kH*kW + kh*kW + kw
							colBuffer[rowIdx*colCols+colIdx] = input[ic*inH*inW+ih*inW+iw]
						}
					}
				}
			}
		}
	}
}

func cpuMatMul(a, b tendo.CPUDataAccessor, shapeA, shapeB, shapeOut []int) *Storage {
	dataA := a.Data()
	dataB := b.Data()

	m := shapeA[len(shapeA)-2]
	k := shapeA[len(shapeA)-1]
	n := shapeB[len(shapeB)-1]

	batchSize := tendo.Numel(shapeOut[:len(shapeOut)-2])
	if batchSize == 0 {
		batchSize = 1
	}

	strideA := m * k
	strideB := k * n
	strideOut := m * n

	batchA := tendo.Numel(shapeA[:len(shapeA)-2])
	batchB := tendo.Numel(shapeB[:len(shapeB)-2])
	if batchA == 0 {
		batchA = 1
	}
	if batchB == 0 {
		batchB = 1
	}

	result := make([]float32, batchSize*m*n)

	for batch := 0; batch < batchSize; batch++ {
		batchIdxA := batch % batchA
		batchIdxB := batch % batchB

		offsetA := batchIdxA * strideA
		offsetB := batchIdxB * strideB
		offsetOut := batch * strideOut

		blasMatMul2D(
			dataA[offsetA:offsetA+strideA],
			dataB[offsetB:offsetB+strideB],
			result[offsetOut:offsetOut+strideOut],
			m, k, n,
		)
	}

	return NewStorageFromSlice(result, tendo.Float32)
}

func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// broadcastStridesWithActual computes broadcast strides using the tensor's actual strides
// instead of computing contiguous strides from shape. This handles non-contiguous tensors.
func broadcastStridesWithActual(srcShape, actualStrides []int, targetShape []int) []int {
	strides := make([]int, len(targetShape))

	dimOffset := len(targetShape) - len(srcShape)
	for i := 0; i < len(targetShape); i++ {
		srcIdx := i - dimOffset
		if srcIdx < 0 || srcShape[srcIdx] == 1 {
			strides[i] = 0
		} else {
			strides[i] = actualStrides[srcIdx]
		}
	}

	return strides
}

// Compile-time check.
var _ tendo.Backend = (*Backend)(nil)
