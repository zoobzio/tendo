package nn

import (
	"context"
	"fmt"

	"github.com/zoobzio/tendo"
)

// NormType represents the normalization type.
type NormType int

// Supported normalization types.
const (
	RMSNorm NormType = iota
	LayerNorm
)

// TransformerBackend combines all operations needed for a transformer layer.
type TransformerBackend interface {
	AttentionBackend
	MLPBackend
	RMSNorm(ctx context.Context, input *tendo.Tensor, normalizedShape []int, weight *tendo.Tensor, epsilon float32) (*tendo.Tensor, error)
	LayerNorm(ctx context.Context, input *tendo.Tensor, normalizedShape []int, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error)
}

// TransformerLayer represents a single transformer block.
// Architecture: x -> norm -> attention -> residual -> norm -> mlp -> residual.
type TransformerLayer struct {
	Attention *Attention
	MLP       *MLP

	AttnNormWeight *tendo.Tensor // normalization weight before attention
	AttnNormBias   *tendo.Tensor // normalization bias (LayerNorm only)
	MLPNormWeight  *tendo.Tensor // normalization weight before MLP
	MLPNormBias    *tendo.Tensor // normalization bias (LayerNorm only)

	NormType NormType
	Epsilon  float32
	Dim      int
}

// TransformerConfig configures a transformer layer.
type TransformerConfig struct {
	Dim        int
	NumHeads   int
	HiddenDim  int
	NormType   NormType
	Activation Activation
	Epsilon    float32
	GatedMLP   bool
	Bias       bool
}

// TransformerLayerOutput contains the layer output and optional KV cache.
type TransformerLayerOutput struct {
	Hidden  *tendo.Tensor
	KVCache *KVCache
}

// Forward computes the transformer layer output.
// Input shape: [batch, seq, dim]
// Output shape: [batch, seq, dim].
func (l *TransformerLayer) Forward(ctx context.Context, x *tendo.Tensor, cache *KVCache, causal bool, backend TransformerBackend) (*TransformerLayerOutput, error) {
	// Pre-attention normalization
	normed, err := l.applyNorm(ctx, x, l.AttnNormWeight, l.AttnNormBias, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.TransformerLayer: attention norm: %w", err)
	}

	// Self-attention
	attnOut, err := l.Attention.Forward(ctx, normed, cache, causal, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.TransformerLayer: attention: %w", err)
	}

	// Residual connection
	h, err := backend.Add(ctx, x, attnOut.Hidden)
	if err != nil {
		return nil, fmt.Errorf("nn.TransformerLayer: attention residual: %w", err)
	}

	// Pre-MLP normalization
	normed, err = l.applyNorm(ctx, h, l.MLPNormWeight, l.MLPNormBias, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.TransformerLayer: mlp norm: %w", err)
	}

	// MLP
	mlpOut, err := l.MLP.Forward(ctx, normed, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.TransformerLayer: mlp: %w", err)
	}

	// Residual connection
	out, err := backend.Add(ctx, h, mlpOut)
	if err != nil {
		return nil, fmt.Errorf("nn.TransformerLayer: mlp residual: %w", err)
	}

	return &TransformerLayerOutput{
		Hidden:  out,
		KVCache: attnOut.KVCache,
	}, nil
}

func (l *TransformerLayer) applyNorm(ctx context.Context, x *tendo.Tensor, weight, bias *tendo.Tensor, backend TransformerBackend) (*tendo.Tensor, error) {
	normalizedShape := []int{l.Dim}

	switch l.NormType {
	case RMSNorm:
		return backend.RMSNorm(ctx, x, normalizedShape, weight, l.Epsilon)
	case LayerNorm:
		return backend.LayerNorm(ctx, x, normalizedShape, weight, bias, l.Epsilon)
	default:
		return nil, fmt.Errorf("unknown norm type: %d", l.NormType)
	}
}

// Transformer represents a stack of transformer layers.
type Transformer struct {
	FinalNormWeight *tendo.Tensor
	FinalNormBias   *tendo.Tensor
	Layers          []*TransformerLayer
	NormType        NormType
	Dim             int
	Epsilon         float32
}

// TransformerOutput contains the model output and KV caches for all layers.
type TransformerOutput struct {
	Hidden   *tendo.Tensor
	KVCaches []*KVCache
}

// Forward computes the transformer output through all layers.
// Input shape: [batch, seq, dim]
// Output shape: [batch, seq, dim].
func (t *Transformer) Forward(ctx context.Context, x *tendo.Tensor, caches []*KVCache, causal bool, backend TransformerBackend) (*TransformerOutput, error) {
	h := x
	newCaches := make([]*KVCache, len(t.Layers))

	for i, layer := range t.Layers {
		var cache *KVCache
		if caches != nil && i < len(caches) {
			cache = caches[i]
		}

		out, err := layer.Forward(ctx, h, cache, causal, backend)
		if err != nil {
			return nil, fmt.Errorf("nn.Transformer: layer %d: %w", i, err)
		}

		h = out.Hidden
		newCaches[i] = out.KVCache
	}

	// Final normalization
	if t.FinalNormWeight != nil {
		var err error
		normalizedShape := []int{t.Dim}

		switch t.NormType {
		case RMSNorm:
			h, err = backend.RMSNorm(ctx, h, normalizedShape, t.FinalNormWeight, t.Epsilon)
		case LayerNorm:
			h, err = backend.LayerNorm(ctx, h, normalizedShape, t.FinalNormWeight, t.FinalNormBias, t.Epsilon)
		}
		if err != nil {
			return nil, fmt.Errorf("nn.Transformer: final norm: %w", err)
		}
	}

	return &TransformerOutput{
		Hidden:   h,
		KVCaches: newCaches,
	}, nil
}
