package nn

import (
	"context"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cpu"
)

func TestNewAttention(t *testing.T) {
	// Helper to create weight tensors of given shape
	makeWeight := func(out, in int) *tendo.Tensor {
		data := make([]float32, out*in)
		for i := range data {
			data[i] = float32(i) * 0.01
		}
		w, _ := tendo.FromSlice(data, out, in)
		return w
	}

	makeBias := func(size int) *tendo.Tensor {
		data := make([]float32, size)
		b, _ := tendo.FromSlice(data, size)
		return b
	}

	tests := []struct {
		name      string
		cfg       AttentionConfig
		wantErr   bool
		errSubstr string
	}{
		{
			name: "valid 8 heads",
			cfg: AttentionConfig{
				Dim:      64,
				NumHeads: 8,
				Bias:     false,
			},
			wantErr: false,
		},
		{
			name: "valid 4 heads with bias",
			cfg: AttentionConfig{
				Dim:      128,
				NumHeads: 4,
				Bias:     true,
			},
			wantErr: false,
		},
		{
			name: "dim not divisible by heads",
			cfg: AttentionConfig{
				Dim:      65,
				NumHeads: 8,
				Bias:     false,
			},
			wantErr:   true,
			errSubstr: "not divisible",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dim := tt.cfg.Dim

			qWeight := makeWeight(dim, dim)
			kWeight := makeWeight(dim, dim)
			vWeight := makeWeight(dim, dim)
			oWeight := makeWeight(dim, dim)

			var qBias, kBias, vBias, oBias *tendo.Tensor
			if tt.cfg.Bias {
				qBias = makeBias(dim)
				kBias = makeBias(dim)
				vBias = makeBias(dim)
				oBias = makeBias(dim)
			}

			attn, err := NewAttention(tt.cfg, qWeight, kWeight, vWeight, oWeight, qBias, kBias, vBias, oBias)
			if tt.wantErr {
				if err == nil {
					t.Errorf("NewAttention() error = nil, want error containing %q", tt.errSubstr)
				}
				return
			}
			if err != nil {
				t.Errorf("NewAttention() unexpected error: %v", err)
				return
			}
			if attn == nil {
				t.Error("NewAttention() returned nil without error")
				return
			}

			// Verify computed fields
			expectedHeadDim := tt.cfg.Dim / tt.cfg.NumHeads
			if attn.HeadDim != expectedHeadDim {
				t.Errorf("HeadDim = %d, want %d", attn.HeadDim, expectedHeadDim)
			}
			if attn.NumHeads != tt.cfg.NumHeads {
				t.Errorf("NumHeads = %d, want %d", attn.NumHeads, tt.cfg.NumHeads)
			}
			if attn.Scale <= 0 {
				t.Errorf("Scale = %f, want > 0", attn.Scale)
			}
		})
	}
}

func TestAttentionConfig(t *testing.T) {
	cfg := AttentionConfig{
		Dim:      512,
		NumHeads: 8,
		Bias:     true,
	}

	if cfg.Dim != 512 {
		t.Errorf("Dim = %d, want 512", cfg.Dim)
	}
	if cfg.NumHeads != 8 {
		t.Errorf("NumHeads = %d, want 8", cfg.NumHeads)
	}
	if !cfg.Bias {
		t.Error("Bias = false, want true")
	}
}

func TestKVCache(t *testing.T) {
	k, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v, _ := tendo.FromSlice([]float32{5, 6, 7, 8}, 1, 1, 2, 2)

	cache := &KVCache{K: k, V: v}

	if cache.K == nil {
		t.Error("KVCache.K is nil")
	}
	if cache.V == nil {
		t.Error("KVCache.V is nil")
	}
}

func TestAttentionOutput(t *testing.T) {
	hidden, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 1, 2, 2)
	k, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v, _ := tendo.FromSlice([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	cache := &KVCache{K: k, V: v}

	output := &AttentionOutput{
		Hidden:  hidden,
		KVCache: cache,
	}

	if output.Hidden == nil {
		t.Error("AttentionOutput.Hidden is nil")
	}
	if output.KVCache == nil {
		t.Error("AttentionOutput.KVCache is nil")
	}
}

func TestAttention_Forward(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim := 16
	numHeads := 4
	headDim := dim / numHeads

	// Create weight tensors [dim, dim]
	qWeight, _ := backend.RandN(dim, dim)
	kWeight, _ := backend.RandN(dim, dim)
	vWeight, _ := backend.RandN(dim, dim)
	oWeight, _ := backend.RandN(dim, dim)

	cfg := AttentionConfig{Dim: dim, NumHeads: numHeads, Bias: false}
	attn, err := NewAttention(cfg, qWeight, kWeight, vWeight, oWeight, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewAttention() error: %v", err)
	}

	// Input [batch, seq, dim]
	batch, seq := 1, 4
	x, _ := backend.RandN(batch, seq, dim)

	out, err := attn.Forward(ctx, x, nil, false, backend)
	if err != nil {
		t.Fatalf("Forward() error: %v", err)
	}

	// Output shape should be [batch, seq, dim]
	if out.Hidden.Size(0) != batch || out.Hidden.Size(1) != seq || out.Hidden.Size(2) != dim {
		t.Errorf("output shape = %v, want [%d, %d, %d]", out.Hidden.Shape(), batch, seq, dim)
	}

	// KVCache should be populated
	if out.KVCache == nil {
		t.Error("KVCache is nil")
	}
	if out.KVCache.K.Size(3) != headDim {
		t.Errorf("KVCache.K head_dim = %d, want %d", out.KVCache.K.Size(3), headDim)
	}
}

func TestAttention_Forward_Causal(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim := 16
	numHeads := 4

	qWeight, _ := backend.RandN(dim, dim)
	kWeight, _ := backend.RandN(dim, dim)
	vWeight, _ := backend.RandN(dim, dim)
	oWeight, _ := backend.RandN(dim, dim)

	cfg := AttentionConfig{Dim: dim, NumHeads: numHeads, Bias: false}
	attn, _ := NewAttention(cfg, qWeight, kWeight, vWeight, oWeight, nil, nil, nil, nil)

	x, _ := backend.RandN(1, 8, dim)

	// With causal masking
	out, err := attn.Forward(ctx, x, nil, true, backend)
	if err != nil {
		t.Fatalf("Forward() with causal error: %v", err)
	}

	if out.Hidden.Size(1) != 8 {
		t.Errorf("output seq = %d, want 8", out.Hidden.Size(1))
	}
}

func TestAttention_Forward_WithCache(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim := 16
	numHeads := 4
	headDim := dim / numHeads

	qWeight, _ := backend.RandN(dim, dim)
	kWeight, _ := backend.RandN(dim, dim)
	vWeight, _ := backend.RandN(dim, dim)
	oWeight, _ := backend.RandN(dim, dim)

	cfg := AttentionConfig{Dim: dim, NumHeads: numHeads, Bias: false}
	attn, _ := NewAttention(cfg, qWeight, kWeight, vWeight, oWeight, nil, nil, nil, nil)

	// First pass: build cache
	x1, _ := backend.RandN(1, 4, dim)
	out1, _ := attn.Forward(ctx, x1, nil, true, backend)

	// Second pass: use cache, single token
	x2, _ := backend.RandN(1, 1, dim)
	out2, err := attn.Forward(ctx, x2, out1.KVCache, true, backend)
	if err != nil {
		t.Fatalf("Forward() with cache error: %v", err)
	}

	// Output should be single token
	if out2.Hidden.Size(1) != 1 {
		t.Errorf("output seq = %d, want 1", out2.Hidden.Size(1))
	}

	// Cache should have grown: 4 + 1 = 5 tokens
	if out2.KVCache.K.Size(2) != 5 {
		t.Errorf("cache seq = %d, want 5", out2.KVCache.K.Size(2))
	}
	if out2.KVCache.K.Size(3) != headDim {
		t.Errorf("cache head_dim = %d, want %d", out2.KVCache.K.Size(3), headDim)
	}
}
