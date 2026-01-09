package tendo

import (
	"context"

	"github.com/zoobzio/capitan"
	"github.com/zoobzio/pipz"
)

// VariantDType is the capitan variant for tensor data types.
const VariantDType capitan.Variant = "tendo.DType"

// Observability keys for dtype operations.
var (
	OpCast     = capitan.NewSignal("tendo.op.cast", "Type cast")
	KeySrcType = capitan.NewKey[DType]("src_dtype", VariantDType)
	KeyDstType = capitan.NewKey[DType]("dst_dtype", VariantDType)
)

// ToFloat32 returns a Chainable that converts a tensor to Float32.
func ToFloat32() pipz.Chainable[*Tensor] {
	return pipz.Apply(pipz.NewIdentity("to_float32", "Convert to Float32"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
		if t.DType() == Float32 {
			// Already float32, return as-is
			return t, nil
		}

		result, err := castCPU(t, Float32)
		if err != nil {
			return nil, err
		}

		emitWithTrace(ctx, OpCast,
			KeyInput.Field(t),
			KeyOutput.Field(result),
			KeySrcType.Field(t.DType()),
			KeyDstType.Field(Float32),
		)

		return result, nil
	})
}

// ToFloat16 returns a Chainable that converts a tensor to Float16.
func ToFloat16() pipz.Chainable[*Tensor] {
	return pipz.Apply(pipz.NewIdentity("to_float16", "Convert to Float16"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
		if t.DType() == Float16 {
			return t, nil
		}

		result, err := castCPU(t, Float16)
		if err != nil {
			return nil, err
		}

		emitWithTrace(ctx, OpCast,
			KeyInput.Field(t),
			KeyOutput.Field(result),
			KeySrcType.Field(t.DType()),
			KeyDstType.Field(Float16),
		)

		return result, nil
	})
}

// ToBFloat16 returns a Chainable that converts a tensor to BFloat16.
func ToBFloat16() pipz.Chainable[*Tensor] {
	return pipz.Apply(pipz.NewIdentity("to_bfloat16", "Convert to BFloat16"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
		if t.DType() == BFloat16 {
			return t, nil
		}

		result, err := castCPU(t, BFloat16)
		if err != nil {
			return nil, err
		}

		emitWithTrace(ctx, OpCast,
			KeyInput.Field(t),
			KeyOutput.Field(result),
			KeySrcType.Field(t.DType()),
			KeyDstType.Field(BFloat16),
		)

		return result, nil
	})
}

// ToDType returns a Chainable that converts a tensor to the specified dtype.
func ToDType(dtype DType) pipz.Chainable[*Tensor] {
	return pipz.Apply(pipz.NewIdentity("to_dtype", "Convert dtype"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
		if t.DType() == dtype {
			return t, nil
		}

		result, err := castCPU(t, dtype)
		if err != nil {
			return nil, err
		}

		emitWithTrace(ctx, OpCast,
			KeyInput.Field(t),
			KeyOutput.Field(result),
			KeySrcType.Field(t.DType()),
			KeyDstType.Field(dtype),
		)

		return result, nil
	})
}

// castCPU converts a CPU tensor to a different dtype.
func castCPU(t *Tensor, dstDtype DType) (*Tensor, error) {
	cpu, ok := t.storage.(CPUDataAccessor)
	if !ok {
		return nil, &DeviceError{Expected: CPU, Got: t.Device().Type}
	}

	srcData := cpu.Data()
	srcDtype := t.DType()

	// First convert to float32 if needed
	var f32Data []float32
	switch srcDtype {
	case Float32:
		f32Data = srcData
	case Float16:
		// CPU storage stores float32 internally even for Float16
		// So srcData is already float32
		f32Data = srcData
	case BFloat16:
		// Same for BFloat16
		f32Data = srcData
	default:
		return nil, &ShapeError{Op: "cast", Message: "unsupported source dtype"}
	}

	// Then convert to destination dtype
	var storage *CPUStorage
	switch dstDtype {
	case Float32:
		storage = NewCPUStorageFromSlice(f32Data, Float32)
	case Float16:
		// Store as float32 internally, but mark as Float16
		// Actual conversion happens on CUDA transfer
		storage = NewCPUStorageFromSlice(f32Data, Float16)
	case BFloat16:
		// Store as float32 internally, but mark as BFloat16
		storage = NewCPUStorageFromSlice(f32Data, BFloat16)
	default:
		return nil, &ShapeError{Op: "cast", Message: "unsupported destination dtype"}
	}

	return NewTensor(storage, t.Shape(), nil), nil
}

// Half is an alias for ToFloat16.
func Half() pipz.Chainable[*Tensor] {
	return ToFloat16()
}

// Float is an alias for ToFloat32.
func Float() pipz.Chainable[*Tensor] {
	return ToFloat32()
}

// BFloat is an alias for ToBFloat16.
func BFloat() pipz.Chainable[*Tensor] {
	return ToBFloat16()
}
