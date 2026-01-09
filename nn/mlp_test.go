package nn

import (
	"context"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cpu"
)

func TestActivationConstants(t *testing.T) {
	// Verify activation constants are distinct
	activations := []Activation{GELU, SiLU, ReLU}
	seen := make(map[Activation]bool)

	for _, act := range activations {
		if seen[act] {
			t.Errorf("duplicate activation constant: %d", act)
		}
		seen[act] = true
	}
}

func TestMLPConfig(t *testing.T) {
	cfg := MLPConfig{
		Dim:        512,
		HiddenDim:  2048,
		Activation: GELU,
		Gated:      false,
		Bias:       true,
	}

	if cfg.Dim != 512 {
		t.Errorf("Dim = %d, want 512", cfg.Dim)
	}
	if cfg.HiddenDim != 2048 {
		t.Errorf("HiddenDim = %d, want 2048", cfg.HiddenDim)
	}
	if cfg.Activation != GELU {
		t.Errorf("Activation = %d, want GELU (%d)", cfg.Activation, GELU)
	}
	if cfg.Gated {
		t.Error("Gated = true, want false")
	}
	if !cfg.Bias {
		t.Error("Bias = false, want true")
	}
}

func TestNewMLP(t *testing.T) {
	makeWeight := func(out, in int) *tendo.Tensor {
		data := make([]float32, out*in)
		w, _ := tendo.FromSlice(data, out, in)
		return w
	}

	tests := []struct {
		name      string
		cfg       MLPConfig
		gateNil   bool
		wantErr   bool
		errSubstr string
	}{
		{
			name: "standard MLP",
			cfg: MLPConfig{
				Dim:        64,
				HiddenDim:  256,
				Activation: GELU,
				Gated:      false,
			},
			gateNil: true,
			wantErr: false,
		},
		{
			name: "gated MLP",
			cfg: MLPConfig{
				Dim:        64,
				HiddenDim:  256,
				Activation: SiLU,
				Gated:      true,
			},
			gateNil: false,
			wantErr: false,
		},
		{
			name: "gated MLP without gate weight",
			cfg: MLPConfig{
				Dim:        64,
				HiddenDim:  256,
				Activation: SiLU,
				Gated:      true,
			},
			gateNil:   true,
			wantErr:   true,
			errSubstr: "gate weight",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upWeight := makeWeight(tt.cfg.HiddenDim, tt.cfg.Dim)
			downWeight := makeWeight(tt.cfg.Dim, tt.cfg.HiddenDim)
			var gateWeight *tendo.Tensor
			if !tt.gateNil {
				gateWeight = makeWeight(tt.cfg.HiddenDim, tt.cfg.Dim)
			}

			mlp, err := NewMLP(tt.cfg, upWeight, downWeight, gateWeight, nil, nil, nil)
			if tt.wantErr {
				if err == nil {
					t.Errorf("NewMLP() error = nil, want error containing %q", tt.errSubstr)
				}
				return
			}
			if err != nil {
				t.Errorf("NewMLP() unexpected error: %v", err)
				return
			}
			if mlp == nil {
				t.Error("NewMLP() returned nil without error")
				return
			}

			// Verify fields
			if mlp.UpProj == nil {
				t.Error("MLP.UpProj is nil")
			}
			if mlp.DownProj == nil {
				t.Error("MLP.DownProj is nil")
			}
			if mlp.Activation != tt.cfg.Activation {
				t.Errorf("MLP.Activation = %d, want %d", mlp.Activation, tt.cfg.Activation)
			}
			if mlp.Gated != tt.cfg.Gated {
				t.Errorf("MLP.Gated = %v, want %v", mlp.Gated, tt.cfg.Gated)
			}
			if tt.cfg.Gated && mlp.GateProj == nil {
				t.Error("MLP.GateProj is nil for gated MLP")
			}
			if !tt.cfg.Gated && mlp.GateProj != nil {
				t.Error("MLP.GateProj should be nil for non-gated MLP")
			}
		})
	}
}

func TestMLP_Forward_Standard(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim, hiddenDim := 4, 8
	upWeight, _ := backend.FromSlice(make([]float32, hiddenDim*dim), hiddenDim, dim)
	downWeight, _ := backend.FromSlice(make([]float32, dim*hiddenDim), dim, hiddenDim)

	cfg := MLPConfig{Dim: dim, HiddenDim: hiddenDim, Activation: GELU, Gated: false}
	mlp, err := NewMLP(cfg, upWeight, downWeight, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewMLP() error: %v", err)
	}

	x, _ := backend.FromSlice(make([]float32, 2*dim), 2, dim)
	out, err := mlp.Forward(ctx, x, backend)
	if err != nil {
		t.Fatalf("Forward() error: %v", err)
	}

	if out.Size(0) != 2 || out.Size(1) != dim {
		t.Errorf("output shape = %v, want [2, %d]", out.Shape(), dim)
	}
}

func TestMLP_Forward_Gated(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim, hiddenDim := 4, 8
	upWeight, _ := backend.FromSlice(make([]float32, hiddenDim*dim), hiddenDim, dim)
	downWeight, _ := backend.FromSlice(make([]float32, dim*hiddenDim), dim, hiddenDim)
	gateWeight, _ := backend.FromSlice(make([]float32, hiddenDim*dim), hiddenDim, dim)

	cfg := MLPConfig{Dim: dim, HiddenDim: hiddenDim, Activation: SiLU, Gated: true}
	mlp, err := NewMLP(cfg, upWeight, downWeight, gateWeight, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewMLP() error: %v", err)
	}

	x, _ := backend.FromSlice(make([]float32, 2*dim), 2, dim)
	out, err := mlp.Forward(ctx, x, backend)
	if err != nil {
		t.Fatalf("Forward() error: %v", err)
	}

	if out.Size(0) != 2 || out.Size(1) != dim {
		t.Errorf("output shape = %v, want [2, %d]", out.Shape(), dim)
	}
}

func TestMLP_Forward_AllActivations(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	activations := []Activation{GELU, SiLU, ReLU}

	for _, act := range activations {
		t.Run(act.String(), func(t *testing.T) {
			dim, hiddenDim := 4, 8
			upWeight, _ := backend.FromSlice(make([]float32, hiddenDim*dim), hiddenDim, dim)
			downWeight, _ := backend.FromSlice(make([]float32, dim*hiddenDim), dim, hiddenDim)

			cfg := MLPConfig{Dim: dim, HiddenDim: hiddenDim, Activation: act, Gated: false}
			mlp, _ := NewMLP(cfg, upWeight, downWeight, nil, nil, nil, nil)

			x, _ := backend.FromSlice(make([]float32, dim), 1, dim)
			_, err := mlp.Forward(ctx, x, backend)
			if err != nil {
				t.Errorf("Forward() with %v error: %v", act, err)
			}
		})
	}
}

func (a Activation) String() string {
	switch a {
	case GELU:
		return "GELU"
	case SiLU:
		return "SiLU"
	case ReLU:
		return "ReLU"
	default:
		return "Unknown"
	}
}
