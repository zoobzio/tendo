package gpt2

import (
	"context"
	"fmt"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/nn"
)

// Backend defines operations needed for GPT-2 inference.
type Backend interface {
	nn.TransformerBackend
	Embedding(ctx context.Context, weight, indices *tendo.Tensor) (*tendo.Tensor, error)
}

// Output contains the model output and KV caches.
type Output struct {
	Logits   *tendo.Tensor   // [batch, seq, vocab_size]
	KVCaches []*nn.KVCache   // one per layer
}

// Forward runs the model on input token IDs.
// Input: token IDs [batch, seq]
// Output: logits [batch, seq, vocab_size]
func (m *Model) Forward(ctx context.Context, tokenIDs *tendo.Tensor, caches []*nn.KVCache, backend Backend) (*Output, error) {
	batch := tokenIDs.Size(0)
	seq := tokenIDs.Size(1)

	// Determine position offset from cache
	posOffset := 0
	if caches != nil && len(caches) > 0 && caches[0] != nil {
		posOffset = caches[0].K.Size(2) // cached sequence length
	}

	// Token embeddings: [batch, seq] -> [batch, seq, dim]
	tokEmb, err := backend.Embedding(ctx, m.TokenEmbed, tokenIDs)
	if err != nil {
		return nil, fmt.Errorf("gpt2: token embedding: %w", err)
	}

	// Position embeddings: need positions [posOffset, posOffset+seq)
	posIDs, err := m.createPositionIDs(ctx, batch, seq, posOffset, backend)
	if err != nil {
		return nil, fmt.Errorf("gpt2: create position ids: %w", err)
	}
	posEmb, err := backend.Embedding(ctx, m.PositionEmbed, posIDs)
	if err != nil {
		return nil, fmt.Errorf("gpt2: position embedding: %w", err)
	}

	// Combine embeddings
	h, err := backend.Add(ctx, tokEmb, posEmb)
	if err != nil {
		return nil, fmt.Errorf("gpt2: add embeddings: %w", err)
	}

	// Run through transformer layers
	newCaches := make([]*nn.KVCache, len(m.Layers))
	for i, layer := range m.Layers {
		var cache *nn.KVCache
		if caches != nil && i < len(caches) {
			cache = caches[i]
		}

		h, newCaches[i], err = m.forwardLayer(ctx, h, layer, cache, backend)
		if err != nil {
			return nil, fmt.Errorf("gpt2: layer %d: %w", i, err)
		}
	}

	// Final layer norm
	normalizedShape := []int{m.Config.Dim}
	h, err = backend.LayerNorm(ctx, h, normalizedShape, m.FinalNormWeight, m.FinalNormBias, m.Config.Epsilon)
	if err != nil {
		return nil, fmt.Errorf("gpt2: final norm: %w", err)
	}

	// Output logits: project to vocabulary
	// Use token embedding as output projection (weight tying)
	// h: [batch, seq, dim], TokenEmbed: [vocab, dim]
	// We want h @ TokenEmbed.T -> [batch, seq, vocab]
	logits, err := backend.MatMul(ctx, h, m.TokenEmbed)
	if err != nil {
		return nil, fmt.Errorf("gpt2: output projection: %w", err)
	}

	return &Output{
		Logits:   logits,
		KVCaches: newCaches,
	}, nil
}

func (m *Model) forwardLayer(ctx context.Context, x *tendo.Tensor, layer *Layer, cache *nn.KVCache, backend Backend) (*tendo.Tensor, *nn.KVCache, error) {
	// Pre-attention layer norm
	normalizedShape := []int{m.Config.Dim}
	normed, err := backend.LayerNorm(ctx, x, normalizedShape, layer.AttnNormWeight, layer.AttnNormBias, m.Config.Epsilon)
	if err != nil {
		return nil, nil, fmt.Errorf("attention norm: %w", err)
	}

	// Self-attention (causal)
	attnOut, err := layer.Attention.Forward(ctx, normed, cache, true, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("attention: %w", err)
	}

	// Residual connection
	h, err := backend.Add(ctx, x, attnOut.Hidden)
	if err != nil {
		return nil, nil, fmt.Errorf("attention residual: %w", err)
	}

	// Pre-MLP layer norm
	normed, err = backend.LayerNorm(ctx, h, normalizedShape, layer.MLPNormWeight, layer.MLPNormBias, m.Config.Epsilon)
	if err != nil {
		return nil, nil, fmt.Errorf("mlp norm: %w", err)
	}

	// MLP
	mlpOut, err := layer.MLP.Forward(ctx, normed, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("mlp: %w", err)
	}

	// Residual connection
	out, err := backend.Add(ctx, h, mlpOut)
	if err != nil {
		return nil, nil, fmt.Errorf("mlp residual: %w", err)
	}

	return out, attnOut.KVCache, nil
}

func (m *Model) createPositionIDs(ctx context.Context, batch, seq, offset int, backend Backend) (*tendo.Tensor, error) {
	// Create position IDs: [batch, seq] with values [offset, offset+1, ..., offset+seq-1]
	data := make([]float32, batch*seq)
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			data[b*seq+s] = float32(offset + s)
		}
	}
	return tendo.FromSlice(data, batch, seq)
}
