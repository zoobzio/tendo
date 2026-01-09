package nn

import (
	"context"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cpu"
)

func TestNewLinear(t *testing.T) {
	tests := []struct {
		name      string
		weight    func() *tendo.Tensor
		bias      func() *tendo.Tensor
		wantErr   bool
		errSubstr string
	}{
		{
			name: "valid without bias",
			weight: func() *tendo.Tensor {
				w, _ := tendo.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
				return w
			},
			bias:    func() *tendo.Tensor { return nil },
			wantErr: false,
		},
		{
			name: "valid with bias",
			weight: func() *tendo.Tensor {
				w, _ := tendo.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
				return w
			},
			bias: func() *tendo.Tensor {
				b, _ := tendo.FromSlice([]float32{1, 2}, 2)
				return b
			},
			wantErr: false,
		},
		{
			name:      "nil weight",
			weight:   func() *tendo.Tensor { return nil },
			bias:     func() *tendo.Tensor { return nil },
			wantErr:  true,
			errSubstr: "weight cannot be nil",
		},
		{
			name: "weight not 2D",
			weight: func() *tendo.Tensor {
				w, _ := tendo.FromSlice([]float32{1, 2, 3}, 3)
				return w
			},
			bias:      func() *tendo.Tensor { return nil },
			wantErr:   true,
			errSubstr: "weight must be 2D",
		},
		{
			name: "bias not 1D",
			weight: func() *tendo.Tensor {
				w, _ := tendo.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
				return w
			},
			bias: func() *tendo.Tensor {
				b, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 2, 2)
				return b
			},
			wantErr:   true,
			errSubstr: "bias must be 1D",
		},
		{
			name: "bias size mismatch",
			weight: func() *tendo.Tensor {
				w, _ := tendo.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
				return w
			},
			bias: func() *tendo.Tensor {
				b, _ := tendo.FromSlice([]float32{1, 2, 3}, 3)
				return b
			},
			wantErr:   true,
			errSubstr: "bias size",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			linear, err := NewLinear(tt.weight(), tt.bias())
			if tt.wantErr {
				if err == nil {
					t.Errorf("NewLinear() error = nil, want error containing %q", tt.errSubstr)
				}
				return
			}
			if err != nil {
				t.Errorf("NewLinear() unexpected error: %v", err)
				return
			}
			if linear == nil {
				t.Error("NewLinear() returned nil without error")
			}
		})
	}
}

func TestLinear_Features(t *testing.T) {
	weight, err := tendo.FromSlice([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}, 3, 4) // [out=3, in=4]
	if err != nil {
		t.Fatalf("failed to create weight tensor: %v", err)
	}

	linear, err := NewLinear(weight, nil)
	if err != nil {
		t.Fatalf("NewLinear() error: %v", err)
	}

	if got := linear.InFeatures(); got != 4 {
		t.Errorf("InFeatures() = %d, want 4", got)
	}
	if got := linear.OutFeatures(); got != 3 {
		t.Errorf("OutFeatures() = %d, want 3", got)
	}
}

func TestLinear_WeightBiasAccess(t *testing.T) {
	weight, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 2, 2)
	bias, _ := tendo.FromSlice([]float32{0.5, 0.5}, 2)

	linear, err := NewLinear(weight, bias)
	if err != nil {
		t.Fatalf("NewLinear() error: %v", err)
	}

	if linear.Weight == nil {
		t.Error("Linear.Weight is nil")
	}
	if linear.Bias == nil {
		t.Error("Linear.Bias is nil")
	}

	// Without bias
	linearNoBias, err := NewLinear(weight, nil)
	if err != nil {
		t.Fatalf("NewLinear() error: %v", err)
	}
	if linearNoBias.Bias != nil {
		t.Error("Linear.Bias should be nil when no bias provided")
	}
}

func TestLinear_Forward(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	// Weight [out=2, in=3], Bias [2]
	weight, _ := backend.FromSlice([]float32{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)
	bias, _ := backend.FromSlice([]float32{0.1, 0.2}, 2)

	linear, err := NewLinear(weight, bias)
	if err != nil {
		t.Fatalf("NewLinear() error: %v", err)
	}

	// Input [batch=2, in=3]
	x, _ := backend.FromSlice([]float32{
		1, 0, 0,
		0, 1, 0,
	}, 2, 3)

	out, err := linear.Forward(ctx, x, backend)
	if err != nil {
		t.Fatalf("Forward() error: %v", err)
	}

	// Output shape should be [batch=2, out=2]
	if out.Dim() != 2 {
		t.Errorf("output dims = %d, want 2", out.Dim())
	}
	if out.Size(0) != 2 || out.Size(1) != 2 {
		t.Errorf("output shape = %v, want [2, 2]", out.Shape())
	}
}

func TestLinear_Forward_NoBias(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	weight, _ := backend.FromSlice([]float32{1, 2, 3, 4}, 2, 2)
	linear, _ := NewLinear(weight, nil)

	x, _ := backend.FromSlice([]float32{1, 1}, 1, 2)
	out, err := linear.Forward(ctx, x, backend)
	if err != nil {
		t.Fatalf("Forward() error: %v", err)
	}

	if out.Size(0) != 1 || out.Size(1) != 2 {
		t.Errorf("output shape = %v, want [1, 2]", out.Shape())
	}
}
