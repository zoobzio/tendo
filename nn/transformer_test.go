package nn

import (
	"context"
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/cpu"
)

func TestNormTypeConstants(t *testing.T) {
	// Verify normalization type constants are distinct
	if RMSNorm == LayerNorm {
		t.Error("RMSNorm and LayerNorm should be distinct")
	}
}

func TestTransformerConfig(t *testing.T) {
	cfg := TransformerConfig{
		Dim:        512,
		NumHeads:   8,
		HiddenDim:  2048,
		NormType:   RMSNorm,
		Activation: SiLU,
		Epsilon:    1e-5,
		GatedMLP:   true,
		Bias:       false,
	}

	if cfg.Dim != 512 {
		t.Errorf("Dim = %d, want 512", cfg.Dim)
	}
	if cfg.NumHeads != 8 {
		t.Errorf("NumHeads = %d, want 8", cfg.NumHeads)
	}
	if cfg.HiddenDim != 2048 {
		t.Errorf("HiddenDim = %d, want 2048", cfg.HiddenDim)
	}
	if cfg.NormType != RMSNorm {
		t.Errorf("NormType = %d, want RMSNorm (%d)", cfg.NormType, RMSNorm)
	}
	if cfg.Activation != SiLU {
		t.Errorf("Activation = %d, want SiLU (%d)", cfg.Activation, SiLU)
	}
	if cfg.Epsilon != 1e-5 {
		t.Errorf("Epsilon = %e, want 1e-5", cfg.Epsilon)
	}
	if !cfg.GatedMLP {
		t.Error("GatedMLP = false, want true")
	}
	if cfg.Bias {
		t.Error("Bias = true, want false")
	}
}

func TestTransformerLayerOutput(t *testing.T) {
	hidden, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 1, 2, 2)
	k, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v, _ := tendo.FromSlice([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	cache := &KVCache{K: k, V: v}

	output := &TransformerLayerOutput{
		Hidden:  hidden,
		KVCache: cache,
	}

	if output.Hidden == nil {
		t.Error("TransformerLayerOutput.Hidden is nil")
	}
	if output.KVCache == nil {
		t.Error("TransformerLayerOutput.KVCache is nil")
	}
}

func TestTransformerOutput(t *testing.T) {
	hidden, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 1, 2, 2)

	k1, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v1, _ := tendo.FromSlice([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	k2, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v2, _ := tendo.FromSlice([]float32{5, 6, 7, 8}, 1, 1, 2, 2)

	caches := []*KVCache{
		{K: k1, V: v1},
		{K: k2, V: v2},
	}

	output := &TransformerOutput{
		Hidden:   hidden,
		KVCaches: caches,
	}

	if output.Hidden == nil {
		t.Error("TransformerOutput.Hidden is nil")
	}
	if len(output.KVCaches) != 2 {
		t.Errorf("len(KVCaches) = %d, want 2", len(output.KVCaches))
	}
}

func TestTransformerLayer_Fields(t *testing.T) {
	// Create minimal layer struct to test field access
	normWeight, _ := tendo.FromSlice([]float32{1, 1, 1, 1}, 4)

	layer := &TransformerLayer{
		AttnNormWeight: normWeight,
		MLPNormWeight:  normWeight,
		NormType:       RMSNorm,
		Epsilon:        1e-6,
		Dim:            4,
	}

	if layer.NormType != RMSNorm {
		t.Errorf("NormType = %d, want RMSNorm (%d)", layer.NormType, RMSNorm)
	}
	if layer.Epsilon != 1e-6 {
		t.Errorf("Epsilon = %e, want 1e-6", layer.Epsilon)
	}
	if layer.Dim != 4 {
		t.Errorf("Dim = %d, want 4", layer.Dim)
	}
}

func TestTransformer_Fields(t *testing.T) {
	normWeight, _ := tendo.FromSlice([]float32{1, 1, 1, 1}, 4)

	transformer := &Transformer{
		FinalNormWeight: normWeight,
		Layers:          make([]*TransformerLayer, 12),
		NormType:        LayerNorm,
		Dim:             4,
		Epsilon:         1e-5,
	}

	if len(transformer.Layers) != 12 {
		t.Errorf("len(Layers) = %d, want 12", len(transformer.Layers))
	}
	if transformer.NormType != LayerNorm {
		t.Errorf("NormType = %d, want LayerNorm (%d)", transformer.NormType, LayerNorm)
	}
	if transformer.Dim != 4 {
		t.Errorf("Dim = %d, want 4", transformer.Dim)
	}
}

func createTestTransformerLayer(t *testing.T, backend *cpu.Backend, dim, numHeads, hiddenDim int, normType NormType) *TransformerLayer {
	t.Helper()

	// Attention weights
	qWeight, _ := backend.RandN(dim, dim)
	kWeight, _ := backend.RandN(dim, dim)
	vWeight, _ := backend.RandN(dim, dim)
	oWeight, _ := backend.RandN(dim, dim)

	cfg := AttentionConfig{Dim: dim, NumHeads: numHeads, Bias: false}
	attn, err := NewAttention(cfg, qWeight, kWeight, vWeight, oWeight, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewAttention() error: %v", err)
	}

	// MLP weights (gated for SwiGLU style)
	upWeight, _ := backend.RandN(hiddenDim, dim)
	downWeight, _ := backend.RandN(dim, hiddenDim)
	gateWeight, _ := backend.RandN(hiddenDim, dim)

	mlpCfg := MLPConfig{Dim: dim, HiddenDim: hiddenDim, Activation: SiLU, Gated: true}
	mlp, err := NewMLP(mlpCfg, upWeight, downWeight, gateWeight, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewMLP() error: %v", err)
	}

	// Norm weights
	attnNorm, _ := backend.Ones(dim)
	mlpNorm, _ := backend.Ones(dim)

	return &TransformerLayer{
		Attention:      attn,
		MLP:            mlp,
		AttnNormWeight: attnNorm,
		MLPNormWeight:  mlpNorm,
		NormType:       normType,
		Epsilon:        1e-5,
		Dim:            dim,
	}
}

func TestTransformerLayer_Forward_RMSNorm(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim, numHeads, hiddenDim := 16, 4, 32
	layer := createTestTransformerLayer(t, backend, dim, numHeads, hiddenDim, RMSNorm)

	x, _ := backend.RandN(1, 4, dim)
	out, err := layer.Forward(ctx, x, nil, true, backend)
	if err != nil {
		t.Fatalf("Forward() error: %v", err)
	}

	if out.Hidden.Size(0) != 1 || out.Hidden.Size(1) != 4 || out.Hidden.Size(2) != dim {
		t.Errorf("output shape = %v, want [1, 4, %d]", out.Hidden.Shape(), dim)
	}
	if out.KVCache == nil {
		t.Error("KVCache is nil")
	}
}

func TestTransformerLayer_Forward_LayerNorm(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim, numHeads, hiddenDim := 16, 4, 32
	layer := createTestTransformerLayer(t, backend, dim, numHeads, hiddenDim, LayerNorm)

	// Add bias for LayerNorm
	layer.AttnNormBias, _ = backend.Zeros(dim)
	layer.MLPNormBias, _ = backend.Zeros(dim)

	x, _ := backend.RandN(1, 4, dim)
	out, err := layer.Forward(ctx, x, nil, false, backend)
	if err != nil {
		t.Fatalf("Forward() error: %v", err)
	}

	if out.Hidden.Size(2) != dim {
		t.Errorf("output dim = %d, want %d", out.Hidden.Size(2), dim)
	}
}

func TestTransformer_Forward(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim, numHeads, hiddenDim, numLayers := 16, 4, 32, 2

	layers := make([]*TransformerLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		layers[i] = createTestTransformerLayer(t, backend, dim, numHeads, hiddenDim, RMSNorm)
	}

	finalNorm, _ := backend.Ones(dim)
	transformer := &Transformer{
		Layers:          layers,
		FinalNormWeight: finalNorm,
		NormType:        RMSNorm,
		Dim:             dim,
		Epsilon:         1e-5,
	}

	x, _ := backend.RandN(1, 4, dim)
	out, err := transformer.Forward(ctx, x, nil, true, backend)
	if err != nil {
		t.Fatalf("Forward() error: %v", err)
	}

	if out.Hidden.Size(0) != 1 || out.Hidden.Size(1) != 4 || out.Hidden.Size(2) != dim {
		t.Errorf("output shape = %v, want [1, 4, %d]", out.Hidden.Shape(), dim)
	}
	if len(out.KVCaches) != numLayers {
		t.Errorf("len(KVCaches) = %d, want %d", len(out.KVCaches), numLayers)
	}
}

func TestTransformer_Forward_WithCache(t *testing.T) {
	backend := cpu.NewBackend()
	ctx := context.Background()

	dim, numHeads, hiddenDim, numLayers := 16, 4, 32, 2

	layers := make([]*TransformerLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		layers[i] = createTestTransformerLayer(t, backend, dim, numHeads, hiddenDim, RMSNorm)
	}

	finalNorm, _ := backend.Ones(dim)
	transformer := &Transformer{
		Layers:          layers,
		FinalNormWeight: finalNorm,
		NormType:        RMSNorm,
		Dim:             dim,
		Epsilon:         1e-5,
	}

	// First pass
	x1, _ := backend.RandN(1, 4, dim)
	out1, _ := transformer.Forward(ctx, x1, nil, true, backend)

	// Second pass with cache
	x2, _ := backend.RandN(1, 1, dim)
	out2, err := transformer.Forward(ctx, x2, out1.KVCaches, true, backend)
	if err != nil {
		t.Fatalf("Forward() with cache error: %v", err)
	}

	if out2.Hidden.Size(1) != 1 {
		t.Errorf("output seq = %d, want 1", out2.Hidden.Size(1))
	}
	// Each layer's cache should have 5 tokens now
	for i, cache := range out2.KVCaches {
		if cache.K.Size(2) != 5 {
			t.Errorf("layer %d cache seq = %d, want 5", i, cache.K.Size(2))
		}
	}
}
