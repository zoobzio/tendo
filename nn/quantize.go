package nn

import (
	"context"
	"fmt"
	"math"

	"github.com/zoobzio/tendo"
)

// QuantType represents the quantization bit width.
type QuantType int

const (
	QuantInt8 QuantType = 8
	QuantInt4 QuantType = 4
)

// QuantizedTensor holds quantized weights with dequantization parameters.
// Supports symmetric per-channel or per-group quantization.
type QuantizedTensor struct {
	// Quantized data - stored as int8 (for int4, two values packed per byte)
	Data *tendo.Tensor

	// Dequantization scales
	// Per-channel: [out_features]
	// Per-group: [out_features, num_groups] where num_groups = in_features / GroupSize
	Scale *tendo.Tensor

	// Original shape before quantization [out_features, in_features]
	Shape []int

	// Quantization parameters
	Type      QuantType
	GroupSize int // 0 = per-channel, >0 = per-group (e.g., 128)
}

// Quantize converts a float32 tensor to int8 with symmetric quantization.
// weight shape: [out_features, in_features]
// groupSize: 0 for per-channel, or group size (e.g., 128) for per-group
func Quantize(weight *tendo.Tensor, groupSize int) (*QuantizedTensor, error) {
	shape := weight.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("nn.Quantize: weight must be 2D, got %dD", len(shape))
	}

	outFeatures, inFeatures := shape[0], shape[1]

	data, err := weight.Data()
	if err != nil {
		return nil, fmt.Errorf("nn.Quantize: get data: %w", err)
	}

	if groupSize == 0 {
		// Per-channel quantization
		return quantizePerChannel(data, outFeatures, inFeatures)
	}

	// Per-group quantization
	if inFeatures%groupSize != 0 {
		return nil, fmt.Errorf("nn.Quantize: in_features %d not divisible by group_size %d", inFeatures, groupSize)
	}
	return quantizePerGroup(data, outFeatures, inFeatures, groupSize)
}

func quantizePerChannel(data []float32, outFeatures, inFeatures int) (*QuantizedTensor, error) {
	quantized := make([]float32, len(data)) // stored as float32 but values are int8 range
	scales := make([]float32, outFeatures)

	for out := 0; out < outFeatures; out++ {
		rowStart := out * inFeatures

		// Find max abs value in this row
		maxAbs := float32(0)
		for in := 0; in < inFeatures; in++ {
			v := data[rowStart+in]
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}

		// Compute scale (avoid division by zero)
		scale := maxAbs / 127.0
		if scale == 0 {
			scale = 1.0
		}
		scales[out] = scale

		// Quantize row
		for in := 0; in < inFeatures; in++ {
			idx := rowStart + in
			q := data[idx] / scale
			// Clamp to int8 range and round
			q = float32(math.Round(float64(q)))
			if q > 127 {
				q = 127
			} else if q < -128 {
				q = -128
			}
			quantized[idx] = q
		}
	}

	dataTensor, err := tendo.FromSlice(quantized, outFeatures, inFeatures)
	if err != nil {
		return nil, err
	}
	scaleTensor, err := tendo.FromSlice(scales, outFeatures)
	if err != nil {
		return nil, err
	}

	return &QuantizedTensor{
		Data:      dataTensor,
		Scale:     scaleTensor,
		Shape:     []int{outFeatures, inFeatures},
		Type:      QuantInt8,
		GroupSize: 0,
	}, nil
}

func quantizePerGroup(data []float32, outFeatures, inFeatures, groupSize int) (*QuantizedTensor, error) {
	numGroups := inFeatures / groupSize
	quantized := make([]float32, len(data))
	scales := make([]float32, outFeatures*numGroups)

	for out := 0; out < outFeatures; out++ {
		rowStart := out * inFeatures

		for g := 0; g < numGroups; g++ {
			groupStart := rowStart + g*groupSize

			// Find max abs value in this group
			maxAbs := float32(0)
			for i := 0; i < groupSize; i++ {
				v := data[groupStart+i]
				if v < 0 {
					v = -v
				}
				if v > maxAbs {
					maxAbs = v
				}
			}

			// Compute scale
			scale := maxAbs / 127.0
			if scale == 0 {
				scale = 1.0
			}
			scales[out*numGroups+g] = scale

			// Quantize group
			for i := 0; i < groupSize; i++ {
				idx := groupStart + i
				q := data[idx] / scale
				q = float32(math.Round(float64(q)))
				if q > 127 {
					q = 127
				} else if q < -128 {
					q = -128
				}
				quantized[idx] = q
			}
		}
	}

	dataTensor, err := tendo.FromSlice(quantized, outFeatures, inFeatures)
	if err != nil {
		return nil, err
	}
	scaleTensor, err := tendo.FromSlice(scales, outFeatures, numGroups)
	if err != nil {
		return nil, err
	}

	return &QuantizedTensor{
		Data:      dataTensor,
		Scale:     scaleTensor,
		Shape:     []int{outFeatures, inFeatures},
		Type:      QuantInt8,
		GroupSize: groupSize,
	}, nil
}

// Dequantize converts quantized weights back to float32.
// This is the CPU reference implementation.
func (q *QuantizedTensor) Dequantize() (*tendo.Tensor, error) {
	quantData, err := q.Data.Data()
	if err != nil {
		return nil, err
	}
	scaleData, err := q.Scale.Data()
	if err != nil {
		return nil, err
	}

	outFeatures, inFeatures := q.Shape[0], q.Shape[1]
	result := make([]float32, outFeatures*inFeatures)

	if q.GroupSize == 0 {
		// Per-channel
		for out := 0; out < outFeatures; out++ {
			scale := scaleData[out]
			rowStart := out * inFeatures
			for in := 0; in < inFeatures; in++ {
				result[rowStart+in] = quantData[rowStart+in] * scale
			}
		}
	} else {
		// Per-group
		numGroups := inFeatures / q.GroupSize
		for out := 0; out < outFeatures; out++ {
			rowStart := out * inFeatures
			for g := 0; g < numGroups; g++ {
				scale := scaleData[out*numGroups+g]
				groupStart := rowStart + g*q.GroupSize
				for i := 0; i < q.GroupSize; i++ {
					result[groupStart+i] = quantData[groupStart+i] * scale
				}
			}
		}
	}

	return tendo.FromSlice(result, outFeatures, inFeatures)
}

// ToDevice copies quantized tensor to target device.
func (q *QuantizedTensor) ToDevice(backend StorageBackend) (*QuantizedTensor, error) {
	data, err := backend.CopyFrom(q.Data)
	if err != nil {
		return nil, fmt.Errorf("nn.QuantizedTensor: copy data: %w", err)
	}
	scale, err := backend.CopyFrom(q.Scale)
	if err != nil {
		return nil, fmt.Errorf("nn.QuantizedTensor: copy scale: %w", err)
	}
	return &QuantizedTensor{
		Data:      data,
		Scale:     scale,
		Shape:     q.Shape,
		Type:      q.Type,
		GroupSize: q.GroupSize,
	}, nil
}

// QuantizedLinearBackend defines operations for quantized linear layers.
type QuantizedLinearBackend interface {
	// DequantizeMatmul performs: output = x @ dequantize(qweight).T
	// x: [batch, seq, in_features]
	// qweight: quantized [out_features, in_features]
	// scale: [out_features] or [out_features, num_groups]
	// output: [batch, seq, out_features]
	DequantizeMatmul(ctx context.Context, x *tendo.Tensor, qweight, scale *tendo.Tensor, groupSize int) (*tendo.Tensor, error)
}

// QuantizedLinear is a linear layer with quantized weights.
type QuantizedLinear struct {
	Weight *QuantizedTensor
	Bias   *tendo.Tensor // optional, kept in fp32
}

// NewQuantizedLinear creates a quantized linear layer from a regular Linear.
func NewQuantizedLinear(linear *Linear, groupSize int) (*QuantizedLinear, error) {
	qweight, err := Quantize(linear.Weight, groupSize)
	if err != nil {
		return nil, fmt.Errorf("nn.QuantizedLinear: quantize weight: %w", err)
	}
	return &QuantizedLinear{
		Weight: qweight,
		Bias:   linear.Bias,
	}, nil
}

// NewQuantizedLinearFromTensors creates a quantized linear from pre-quantized data.
func NewQuantizedLinearFromTensors(data, scale *tendo.Tensor, shape []int, groupSize int, bias *tendo.Tensor) *QuantizedLinear {
	return &QuantizedLinear{
		Weight: &QuantizedTensor{
			Data:      data,
			Scale:     scale,
			Shape:     shape,
			Type:      QuantInt8,
			GroupSize: groupSize,
		},
		Bias: bias,
	}
}

// Forward computes y = x @ dequantize(W).T + b
func (l *QuantizedLinear) Forward(ctx context.Context, x *tendo.Tensor, backend QuantizedLinearBackend) (*tendo.Tensor, error) {
	out, err := backend.DequantizeMatmul(ctx, x, l.Weight.Data, l.Weight.Scale, l.Weight.GroupSize)
	if err != nil {
		return nil, fmt.Errorf("nn.QuantizedLinear: dequantize matmul: %w", err)
	}

	if l.Bias != nil {
		// Need a backend that can do Add
		if addBackend, ok := backend.(interface {
			Add(context.Context, *tendo.Tensor, *tendo.Tensor) (*tendo.Tensor, error)
		}); ok {
			out, err = addBackend.Add(ctx, out, l.Bias)
			if err != nil {
				return nil, fmt.Errorf("nn.QuantizedLinear: add bias: %w", err)
			}
		}
	}

	return out, nil
}

// InFeatures returns the input dimension.
func (l *QuantizedLinear) InFeatures() int {
	return l.Weight.Shape[1]
}

// OutFeatures returns the output dimension.
func (l *QuantizedLinear) OutFeatures() int {
	return l.Weight.Shape[0]
}

// ToDevice moves the quantized linear layer to target device.
func (l *QuantizedLinear) ToDevice(backend StorageBackend) (*QuantizedLinear, error) {
	qweight, err := l.Weight.ToDevice(backend)
	if err != nil {
		return nil, err
	}

	var bias *tendo.Tensor
	if l.Bias != nil {
		bias, err = backend.CopyFrom(l.Bias)
		if err != nil {
			return nil, fmt.Errorf("nn.QuantizedLinear: copy bias: %w", err)
		}
	}

	return &QuantizedLinear{
		Weight: qweight,
		Bias:   bias,
	}, nil
}
