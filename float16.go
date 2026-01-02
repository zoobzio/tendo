package tendo

import (
	"math"
)

// Float16 conversion utilities
//
// IEEE 754 half-precision (float16):
//   - Sign: 1 bit
//   - Exponent: 5 bits (bias 15)
//   - Mantissa: 10 bits
//
// BFloat16 (brain floating point):
//   - Sign: 1 bit
//   - Exponent: 8 bits (bias 127, same as float32)
//   - Mantissa: 7 bits
//   - Essentially the upper 16 bits of float32

// Float32ToFloat16 converts a float32 to float16 (IEEE 754 half-precision).
// Returns the 16-bit representation as uint16.
func Float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)

	sign := (bits >> 16) & 0x8000
	f32Exp := (bits >> 23) & 0xFF
	f32Mant := bits & 0x7FFFFF

	// Handle special cases first
	if f32Exp == 255 {
		// Float32 Infinity or NaN
		if f32Mant != 0 {
			// NaN - preserve some mantissa bits to keep it NaN
			return uint16(sign | 0x7C00 | 0x0200) // Use a non-zero mantissa
		}
		return uint16(sign | 0x7C00) // Infinity
	}

	exp := int(f32Exp) - 127 + 15 // rebias from 127 to 15
	mant := (bits >> 13) & 0x3FF  // top 10 bits of mantissa

	// Handle special cases
	if exp <= 0 {
		// Subnormal or zero
		if exp < -10 {
			// Too small, flush to zero
			return uint16(sign)
		}
		// Subnormal: shift mantissa
		mant = (mant | 0x400) >> uint(1-exp)
		return uint16(sign | mant)
	}

	if exp >= 31 {
		// Overflow to infinity
		return uint16(sign | 0x7C00)
	}

	// Normal number
	return uint16(sign | uint32(exp)<<10 | mant)
}

// Float16ToFloat32 converts a float16 (IEEE 754 half-precision) to float32.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h & 0x3FF)

	switch exp {
	case 0:
		if mant == 0 {
			// Zero
			return math.Float32frombits(sign)
		}
		// Subnormal: normalize
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	case 31:
		// Infinity or NaN
		if mant == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return math.Float32frombits(sign | 0x7F800000 | (mant << 13))
	}

	// Normal number: rebias exponent from 15 to 127
	exp = exp - 15 + 127
	return math.Float32frombits(sign | (exp << 23) | (mant << 13))
}

// Float32ToBFloat16 converts a float32 to bfloat16.
// BFloat16 is the upper 16 bits of float32 with rounding.
func Float32ToBFloat16(f float32) uint16 {
	bits := math.Float32bits(f)

	// Round to nearest even (banker's rounding)
	// Add 0x7FFF + (bit 16 for tie-breaking)
	rounding := uint32(0x7FFF) + ((bits >> 16) & 1)
	bits += rounding

	// Take upper 16 bits
	return uint16(bits >> 16)
}

// BFloat16ToFloat32 converts a bfloat16 to float32.
// Simply places the 16 bits in the upper half of float32.
func BFloat16ToFloat32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// Float32SliceToFloat16 converts a slice of float32 to float16.
func Float32SliceToFloat16(f32 []float32) []uint16 {
	result := make([]uint16, len(f32))
	for i, v := range f32 {
		result[i] = Float32ToFloat16(v)
	}
	return result
}

// Float16SliceToFloat32 converts a slice of float16 to float32.
func Float16SliceToFloat32(f16 []uint16) []float32 {
	result := make([]float32, len(f16))
	for i, v := range f16 {
		result[i] = Float16ToFloat32(v)
	}
	return result
}

// Float32SliceToBFloat16 converts a slice of float32 to bfloat16.
func Float32SliceToBFloat16(f32 []float32) []uint16 {
	result := make([]uint16, len(f32))
	for i, v := range f32 {
		result[i] = Float32ToBFloat16(v)
	}
	return result
}

// BFloat16SliceToFloat32 converts a slice of bfloat16 to float32.
func BFloat16SliceToFloat32(bf16 []uint16) []float32 {
	result := make([]float32, len(bf16))
	for i, v := range bf16 {
		result[i] = BFloat16ToFloat32(v)
	}
	return result
}
