package nn

import (
	"context"
	"math"
	"testing"

	"github.com/zoobzio/tendo/cpu"
)

func TestRoPE_Apply(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim := 8
	maxSeq := 32
	rope, err := NewRoPE(dim, maxSeq, 10000.0)
	if err != nil {
		t.Fatalf("NewRoPE() error: %v", err)
	}

	// q, k shape: [batch, heads, seq, head_dim]
	batch, heads, seq := 1, 2, 4
	q, _ := backend.RandN(batch, heads, seq, dim)
	k, _ := backend.RandN(batch, heads, seq, dim)

	qRot, kRot, err := rope.Apply(ctx, q, k, 0, backend)
	if err != nil {
		t.Fatalf("Apply() error: %v", err)
	}

	// Output shapes should match input
	if qRot.Size(0) != batch || qRot.Size(1) != heads || qRot.Size(2) != seq || qRot.Size(3) != dim {
		t.Errorf("qRot shape = %v, want [%d, %d, %d, %d]", qRot.Shape(), batch, heads, seq, dim)
	}
	if kRot.Size(0) != batch || kRot.Size(1) != heads || kRot.Size(2) != seq || kRot.Size(3) != dim {
		t.Errorf("kRot shape = %v, want [%d, %d, %d, %d]", kRot.Shape(), batch, heads, seq, dim)
	}
}

func TestRoPE_Apply_WithOffset(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim := 8
	maxSeq := 32
	rope, _ := NewRoPE(dim, maxSeq, 10000.0)

	q, _ := backend.RandN(1, 2, 1, dim) // single token
	k, _ := backend.RandN(1, 2, 1, dim)

	// Apply with offset (simulating KV cache scenario)
	_, _, err := rope.Apply(ctx, q, k, 10, backend)
	if err != nil {
		t.Fatalf("Apply() with offset error: %v", err)
	}
}

func TestRoPE_Apply_ExceedsMaxSeq(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim := 8
	maxSeq := 10
	rope, _ := NewRoPE(dim, maxSeq, 10000.0)

	q, _ := backend.RandN(1, 1, 5, dim)
	k, _ := backend.RandN(1, 1, 5, dim)

	// offset + seq > maxSeq should error
	_, _, err := rope.Apply(ctx, q, k, 8, backend)
	if err == nil {
		t.Error("Apply() should error when position exceeds max")
	}
}

func TestNewRoPE(t *testing.T) {
	tests := []struct {
		name      string
		dim       int
		maxSeq    int
		base      float32
		wantErr   bool
		errSubstr string
	}{
		{
			name:    "valid even dim",
			dim:     64,
			maxSeq:  2048,
			base:    10000.0,
			wantErr: false,
		},
		{
			name:    "valid small dim",
			dim:     8,
			maxSeq:  512,
			base:    10000.0,
			wantErr: false,
		},
		{
			name:      "odd dim",
			dim:       63,
			maxSeq:    2048,
			base:      10000.0,
			wantErr:   true,
			errSubstr: "dim must be even",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rope, err := NewRoPE(tt.dim, tt.maxSeq, tt.base)
			if tt.wantErr {
				if err == nil {
					t.Errorf("NewRoPE() error = nil, want error containing %q", tt.errSubstr)
				}
				return
			}
			if err != nil {
				t.Errorf("NewRoPE() unexpected error: %v", err)
				return
			}
			if rope == nil {
				t.Error("NewRoPE() returned nil without error")
				return
			}

			// Verify fields
			if rope.Dim != tt.dim {
				t.Errorf("Dim = %d, want %d", rope.Dim, tt.dim)
			}
			if rope.MaxSeqLen != tt.maxSeq {
				t.Errorf("MaxSeqLen = %d, want %d", rope.MaxSeqLen, tt.maxSeq)
			}
			if rope.Base != tt.base {
				t.Errorf("Base = %f, want %f", rope.Base, tt.base)
			}
			if rope.CosCache == nil {
				t.Error("CosCache is nil")
			}
			if rope.SinCache == nil {
				t.Error("SinCache is nil")
			}
		})
	}
}

func TestRoPE_CacheShape(t *testing.T) {
	dim := 64
	maxSeq := 128
	rope, err := NewRoPE(dim, maxSeq, 10000.0)
	if err != nil {
		t.Fatalf("NewRoPE() error: %v", err)
	}

	// Cache shape should be [maxSeq, dim/2]
	halfDim := dim / 2
	expectedShape := []int{maxSeq, halfDim}

	cosShape := rope.CosCache.Shape()
	if len(cosShape) != len(expectedShape) {
		t.Errorf("CosCache shape dims = %d, want %d", len(cosShape), len(expectedShape))
	}
	for i, v := range expectedShape {
		if cosShape[i] != v {
			t.Errorf("CosCache shape[%d] = %d, want %d", i, cosShape[i], v)
		}
	}

	sinShape := rope.SinCache.Shape()
	if len(sinShape) != len(expectedShape) {
		t.Errorf("SinCache shape dims = %d, want %d", len(sinShape), len(expectedShape))
	}
	for i, v := range expectedShape {
		if sinShape[i] != v {
			t.Errorf("SinCache shape[%d] = %d, want %d", i, sinShape[i], v)
		}
	}
}

func TestRoPE_FrequencyValues(t *testing.T) {
	// Test with simple parameters for verification
	dim := 4
	maxSeq := 4
	base := float32(10000.0)

	rope, err := NewRoPE(dim, maxSeq, base)
	if err != nil {
		t.Fatalf("NewRoPE() error: %v", err)
	}

	// Get the cached values
	cosData, err := rope.CosCache.Data()
	if err != nil {
		t.Fatalf("CosCache.Data() error: %v", err)
	}
	sinData, err := rope.SinCache.Data()
	if err != nil {
		t.Fatalf("SinCache.Data() error: %v", err)
	}

	halfDim := dim / 2

	// Verify position 0 has cos=1, sin=0 for all frequencies
	for i := 0; i < halfDim; i++ {
		if math.Abs(float64(cosData[i]-1.0)) > 1e-5 {
			t.Errorf("cos(pos=0, i=%d) = %f, want 1.0", i, cosData[i])
		}
		if math.Abs(float64(sinData[i])) > 1e-5 {
			t.Errorf("sin(pos=0, i=%d) = %f, want 0.0", i, sinData[i])
		}
	}
}

func TestRoPE_ToDevice(t *testing.T) {
	backend := cpu.NewBackend()

	rope, err := NewRoPE(16, 128, 10000.0)
	if err != nil {
		t.Fatalf("NewRoPE() error: %v", err)
	}

	ropeDevice, err := rope.ToDevice(backend)
	if err != nil {
		t.Fatalf("ToDevice() error: %v", err)
	}

	if ropeDevice.CosCache == nil {
		t.Error("ToDevice() CosCache is nil")
	}
	if ropeDevice.SinCache == nil {
		t.Error("ToDevice() SinCache is nil")
	}
	if ropeDevice.Dim != rope.Dim {
		t.Errorf("Dim = %d, want %d", ropeDevice.Dim, rope.Dim)
	}
	if ropeDevice.MaxSeqLen != rope.MaxSeqLen {
		t.Errorf("MaxSeqLen = %d, want %d", ropeDevice.MaxSeqLen, rope.MaxSeqLen)
	}
}
