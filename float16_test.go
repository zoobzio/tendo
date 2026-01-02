package tendo

import (
	"math"
	"testing"
)

func TestFloat16Conversion(t *testing.T) {
	tests := []struct {
		name     string
		input    float32
		expected float32
		tolerance float32
	}{
		{"zero", 0.0, 0.0, 0},
		{"one", 1.0, 1.0, 0},
		{"negative_one", -1.0, -1.0, 0},
		{"small", 0.5, 0.5, 0},
		{"pi", float32(math.Pi), 3.140625, 0.001}, // Float16 has limited precision
		{"large", 1000.0, 1000.0, 0},
		{"tiny", 0.0001, 0.0001, 0.00001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := Float32ToFloat16(tt.input)
			result := Float16ToFloat32(h)

			diff := float32(math.Abs(float64(result - tt.expected)))
			if diff > tt.tolerance {
				t.Errorf("Float16 round-trip: input=%v, got=%v, expected=%v (diff=%v)",
					tt.input, result, tt.expected, diff)
			}
		})
	}
}

func TestFloat16SpecialValues(t *testing.T) {
	// Test infinity
	posInf := float32(math.Inf(1))
	h := Float32ToFloat16(posInf)
	result := Float16ToFloat32(h)
	if !math.IsInf(float64(result), 1) {
		t.Errorf("Positive infinity: got %v", result)
	}

	negInf := float32(math.Inf(-1))
	h = Float32ToFloat16(negInf)
	result = Float16ToFloat32(h)
	if !math.IsInf(float64(result), -1) {
		t.Errorf("Negative infinity: got %v", result)
	}

	// Test NaN
	nan := float32(math.NaN())
	h = Float32ToFloat16(nan)
	result = Float16ToFloat32(h)
	if !math.IsNaN(float64(result)) {
		t.Errorf("NaN: got %v", result)
	}

	// Test negative zero - either preserved as -0 or converted to +0 is acceptable
	negZero := float32(math.Copysign(0, -1))
	h = Float32ToFloat16(negZero)
	result = Float16ToFloat32(h)
	// Both +0 and -0 compare equal to 0, so just verify it's a zero
	if result != 0 {
		t.Errorf("Negative zero conversion failed: got %v", result)
	}
}

func TestBFloat16Conversion(t *testing.T) {
	tests := []struct {
		name      string
		input     float32
		tolerance float32
	}{
		{"zero", 0.0, 0},
		{"one", 1.0, 0},
		{"negative_one", -1.0, 0},
		{"small", 0.5, 0},
		{"pi", float32(math.Pi), 0.02}, // BFloat16 has better range but less precision (7 bits)
		{"large", 10000.0, 128},        // ~1% relative error for larger values
		{"very_large", 1e30, 1e27},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := Float32ToBFloat16(tt.input)
			result := BFloat16ToFloat32(b)

			diff := float32(math.Abs(float64(result - tt.input)))
			if diff > tt.tolerance {
				t.Errorf("BFloat16 round-trip: input=%v, got=%v (diff=%v, tolerance=%v)",
					tt.input, result, diff, tt.tolerance)
			}
		})
	}
}

func TestBFloat16SpecialValues(t *testing.T) {
	// Test infinity
	posInf := float32(math.Inf(1))
	b := Float32ToBFloat16(posInf)
	result := BFloat16ToFloat32(b)
	if !math.IsInf(float64(result), 1) {
		t.Errorf("Positive infinity: got %v", result)
	}

	// Test NaN
	nan := float32(math.NaN())
	b = Float32ToBFloat16(nan)
	result = BFloat16ToFloat32(b)
	if !math.IsNaN(float64(result)) {
		t.Errorf("NaN: got %v", result)
	}
}

func TestFloat16SliceConversion(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, -1.0, 0.5, 0.0}

	h := Float32SliceToFloat16(input)
	if len(h) != len(input) {
		t.Fatalf("Float16 slice length: got %d, expected %d", len(h), len(input))
	}

	result := Float16SliceToFloat32(h)
	if len(result) != len(input) {
		t.Fatalf("Result slice length: got %d, expected %d", len(result), len(input))
	}

	for i, v := range result {
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > 0.001 {
			t.Errorf("Index %d: got %v, expected %v", i, v, input[i])
		}
	}
}

func TestBFloat16SliceConversion(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, -1.0, 0.5, 0.0, 1000.0}

	b := Float32SliceToBFloat16(input)
	if len(b) != len(input) {
		t.Fatalf("BFloat16 slice length: got %d, expected %d", len(b), len(input))
	}

	result := BFloat16SliceToFloat32(b)
	if len(result) != len(input) {
		t.Fatalf("Result slice length: got %d, expected %d", len(result), len(input))
	}

	for i, v := range result {
		tolerance := float32(0.1)
		if math.Abs(float64(input[i])) > 100 {
			tolerance = float32(math.Abs(float64(input[i])) * 0.01)
		}
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > tolerance {
			t.Errorf("Index %d: got %v, expected %v (diff=%v)", i, v, input[i], diff)
		}
	}
}

func TestDTypeSize(t *testing.T) {
	if Float32.Size() != 4 {
		t.Errorf("Float32 size: got %d, expected 4", Float32.Size())
	}
	if Float16.Size() != 2 {
		t.Errorf("Float16 size: got %d, expected 2", Float16.Size())
	}
	if BFloat16.Size() != 2 {
		t.Errorf("BFloat16 size: got %d, expected 2", BFloat16.Size())
	}
}

func TestDTypeString(t *testing.T) {
	if Float32.String() != "float32" {
		t.Errorf("Float32 string: got %q", Float32.String())
	}
	if Float16.String() != "float16" {
		t.Errorf("Float16 string: got %q", Float16.String())
	}
	if BFloat16.String() != "bfloat16" {
		t.Errorf("BFloat16 string: got %q", BFloat16.String())
	}
}

func BenchmarkFloat32ToFloat16(b *testing.B) {
	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Float32SliceToFloat16(data)
	}
}

func BenchmarkFloat32ToBFloat16(b *testing.B) {
	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Float32SliceToBFloat16(data)
	}
}
