package llama

import (
	"context"
	"fmt"
	"math"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/nn"
)

// tensorCleanup tracks tensors that own storage and need to be freed.
// Views (from Reshape, Permute, etc.) should NOT be added here.
type tensorCleanup struct {
	tensors []*tendo.Tensor
}

// track adds a tensor to be freed later. Returns the tensor for chaining.
func (c *tensorCleanup) track(t *tendo.Tensor) *tendo.Tensor {
	if t != nil {
		c.tensors = append(c.tensors, t)
	}
	return t
}

// free releases all tracked tensors.
func (c *tensorCleanup) free() {
	for _, t := range c.tensors {
		if t != nil {
			t.Free()
		}
	}
	c.tensors = nil
}

// Backend defines operations needed for Llama inference.
type Backend interface {
	nn.LinearBackend
	nn.RoPEBackend
	nn.StorageBackend
	Softmax(ctx context.Context, t *tendo.Tensor, dim int) (*tendo.Tensor, error)
	RMSNorm(ctx context.Context, input *tendo.Tensor, normalizedShape []int, weight *tendo.Tensor, epsilon float32) (*tendo.Tensor, error)
	SiLU(ctx context.Context, t *tendo.Tensor) (*tendo.Tensor, error)
	Full(value float32, shape ...int) (*tendo.Tensor, error)
	Embedding(ctx context.Context, weight, indices *tendo.Tensor) (*tendo.Tensor, error)
	Tril(ctx context.Context, t *tendo.Tensor, k int) (*tendo.Tensor, error)
	Where(ctx context.Context, condition, x, y *tendo.Tensor) (*tendo.Tensor, error)
	Ones(shape ...int) (*tendo.Tensor, error)
	FromSlice(data []float32, shape ...int) (*tendo.Tensor, error)
	FromInt64Slice(data []int64, shape ...int) (*tendo.Tensor, error)
}

// KVCache holds cached key and value tensors for a single layer.
type KVCache struct {
	K *tendo.Tensor // [batch, num_kv_heads, seq, head_dim]
	V *tendo.Tensor // [batch, num_kv_heads, seq, head_dim]
}

// Output contains the model output and KV caches.
type Output struct {
	Logits   *tendo.Tensor // [batch, seq, vocab_size]
	KVCaches []*KVCache    // one per layer
}

// Forward runs the model on input token IDs.
func (m *Model) Forward(ctx context.Context, tokenIDs *tendo.Tensor, caches []*KVCache, backend Backend) (*Output, error) {
	// Determine position offset from cache
	posOffset := 0
	if caches != nil && len(caches) > 0 && caches[0] != nil {
		posOffset = caches[0].K.Size(2)
	}

	// Token embeddings: [batch, seq] -> [batch, seq, dim]
	h, err := backend.Embedding(ctx, m.TokenEmbed, tokenIDs)
	if err != nil {
		return nil, fmt.Errorf("llama: token embedding: %w", err)
	}

	// Run through transformer layers
	newCaches := make([]*KVCache, len(m.Layers))
	for i, layer := range m.Layers {
		var cache *KVCache
		if caches != nil && i < len(caches) {
			cache = caches[i]
		}

		prevH := h
		h, newCaches[i], err = m.forwardLayer(ctx, h, layer, cache, posOffset, backend)
		prevH.Free() // Free previous layer's output
		if err != nil {
			return nil, fmt.Errorf("llama: layer %d: %w", i, err)
		}
	}

	// Final RMSNorm
	normalizedShape := []int{m.Config.Dim}
	hNormed, err := backend.RMSNorm(ctx, h, normalizedShape, m.FinalNormWeight, m.Config.RMSEpsilon)
	h.Free() // Free pre-norm hidden state
	if err != nil {
		return nil, fmt.Errorf("llama: final norm: %w", err)
	}

	// Output logits: [batch, seq, dim] @ [vocab, dim].T -> [batch, seq, vocab]
	// OutputWeight is [vocab, dim], we need [dim, vocab] for matmul
	// Note: NewT creates a view (no new storage), so don't free outputT
	outputT, err := tendo.NewT().Process(ctx, m.OutputWeight)
	if err != nil {
		return nil, fmt.Errorf("llama: transpose output weight: %w", err)
	}
	logits, err := backend.MatMul(ctx, hNormed, outputT)
	hNormed.Free() // Free normalized hidden state
	if err != nil {
		return nil, fmt.Errorf("llama: output projection: %w", err)
	}

	return &Output{
		Logits:   logits,
		KVCaches: newCaches,
	}, nil
}

func (m *Model) forwardLayer(ctx context.Context, x *tendo.Tensor, layer *Layer, cache *KVCache, posOffset int, backend Backend) (*tendo.Tensor, *KVCache, error) {
	batch := x.Size(0)
	seq := x.Size(1)

	// Pre-attention RMSNorm
	normalizedShape := []int{m.Config.Dim}
	normed, err := backend.RMSNorm(ctx, x, normalizedShape, layer.AttnNormWeight, m.Config.RMSEpsilon)
	if err != nil {
		return nil, nil, fmt.Errorf("attention norm: %w", err)
	}

	// Attention with RoPE
	attnOut, newCache, err := m.forwardAttention(ctx, normed, layer, cache, batch, seq, posOffset, backend)
	normed.Free() // Free pre-attention normed
	if err != nil {
		return nil, nil, fmt.Errorf("attention: %w", err)
	}

	// Residual
	h, err := backend.Add(ctx, x, attnOut)
	attnOut.Free() // Free attention output
	if err != nil {
		return nil, nil, fmt.Errorf("attention residual: %w", err)
	}

	// Pre-MLP RMSNorm
	normedMLP, err := backend.RMSNorm(ctx, h, normalizedShape, layer.MLPNormWeight, m.Config.RMSEpsilon)
	if err != nil {
		return nil, nil, fmt.Errorf("mlp norm: %w", err)
	}

	// SwiGLU MLP: down(silu(gate(x)) * up(x))
	mlpOut, err := m.forwardMLP(ctx, normedMLP, layer, backend)
	normedMLP.Free() // Free pre-MLP normed
	if err != nil {
		return nil, nil, fmt.Errorf("mlp: %w", err)
	}

	// Residual
	out, err := backend.Add(ctx, h, mlpOut)
	h.Free()      // Free post-attention hidden state
	mlpOut.Free() // Free MLP output
	if err != nil {
		return nil, nil, fmt.Errorf("mlp residual: %w", err)
	}

	return out, newCache, nil
}

func (m *Model) forwardAttention(ctx context.Context, x *tendo.Tensor, layer *Layer, cache *KVCache, batch, seq, posOffset int, backend Backend) (*tendo.Tensor, *KVCache, error) {
	// Project to Q, K, V - these own their storage
	qProj, err := layer.QProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("q projection: %w", err)
	}
	kProj, err := layer.KProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("k projection: %w", err)
	}
	vProj, err := layer.VProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("v projection: %w", err)
	}

	// Reshape for multi-head attention (creates views, don't free)
	// Q: [batch, seq, dim] -> [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
	// K, V: [batch, seq, kv_dim] -> [batch, seq, num_kv_heads, head_dim] -> [batch, num_kv_heads, seq, head_dim]
	q, err := tendo.NewReshape(batch, seq, layer.NumHeads, layer.HeadDim).Process(ctx, qProj)
	if err != nil {
		return nil, nil, fmt.Errorf("reshape q: %w", err)
	}
	q, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, q)
	if err != nil {
		return nil, nil, fmt.Errorf("permute q: %w", err)
	}

	k, err := tendo.NewReshape(batch, seq, layer.NumKVHeads, layer.HeadDim).Process(ctx, kProj)
	if err != nil {
		return nil, nil, fmt.Errorf("reshape k: %w", err)
	}
	k, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, k)
	if err != nil {
		return nil, nil, fmt.Errorf("permute k: %w", err)
	}

	v, err := tendo.NewReshape(batch, seq, layer.NumKVHeads, layer.HeadDim).Process(ctx, vProj)
	if err != nil {
		return nil, nil, fmt.Errorf("reshape v: %w", err)
	}
	v, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, v)
	if err != nil {
		return nil, nil, fmt.Errorf("permute v: %w", err)
	}

	// Apply RoPE to Q and K - RoPE allocates new storage
	qRoped, kRoped, err := m.RoPE.Apply(ctx, q, k, posOffset, backend)
	qProj.Free() // Done with Q projection storage
	kProj.Free() // Done with K projection storage (before cache concat)
	if err != nil {
		vProj.Free()
		return nil, nil, fmt.Errorf("rope: %w", err)
	}

	// Handle KV cache
	if cache != nil {
		kCached, err := backend.Cat(ctx, []*tendo.Tensor{cache.K, kRoped}, 2)
		kRoped.Free() // Free pre-concat K
		if err != nil {
			vProj.Free()
			return nil, nil, fmt.Errorf("cat cached k: %w", err)
		}
		kRoped = kCached

		vCached, err := backend.Cat(ctx, []*tendo.Tensor{cache.V, v}, 2)
		vProj.Free() // Free V projection storage
		if err != nil {
			kRoped.Free()
			return nil, nil, fmt.Errorf("cat cached v: %w", err)
		}
		v = vCached
	}
	newCache := &KVCache{K: kRoped, V: v}

	// GQA: repeat K, V heads to match Q heads (creates views)
	kExpanded := kRoped
	vExpanded := v
	if layer.NumKVHeads < layer.NumHeads {
		kExpanded, err = repeatKVHeads(ctx, kRoped, layer.NumHeads/layer.NumKVHeads)
		if err != nil {
			return nil, nil, fmt.Errorf("repeat k heads: %w", err)
		}
		vExpanded, err = repeatKVHeads(ctx, v, layer.NumHeads/layer.NumKVHeads)
		if err != nil {
			return nil, nil, fmt.Errorf("repeat v heads: %w", err)
		}
	}

	// Scaled dot-product attention
	// scores = Q @ K.T / sqrt(head_dim)
	// kT is a view (transpose doesn't copy)
	kT, err := tendo.NewTranspose(-2, -1).Process(ctx, kExpanded)
	if err != nil {
		return nil, nil, fmt.Errorf("transpose k: %w", err)
	}

	scores, err := backend.MatMul(ctx, qRoped, kT)
	qRoped.Free() // Done with Q after matmul
	if err != nil {
		return nil, nil, fmt.Errorf("q @ k^T: %w", err)
	}

	scale := 1.0 / float32(math.Sqrt(float64(layer.HeadDim)))
	scaleTensor, err := backend.Full(scale, 1)
	if err != nil {
		scores.Free()
		return nil, nil, fmt.Errorf("create scale: %w", err)
	}
	scoresScaled, err := backend.Mul(ctx, scores, scaleTensor)
	scores.Free()
	scaleTensor.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("scale scores: %w", err)
	}

	// Causal mask
	scoresMasked, err := applyCausalMask(ctx, scoresScaled, backend)
	scoresScaled.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("causal mask: %w", err)
	}

	// Softmax
	attnWeights, err := backend.Softmax(ctx, scoresMasked, -1)
	scoresMasked.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("softmax: %w", err)
	}

	// Attention output
	attnMatmul, err := backend.MatMul(ctx, attnWeights, vExpanded)
	attnWeights.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("weights @ v: %w", err)
	}

	// Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, dim]
	// Permute and Reshape create views
	attnOut, err := tendo.NewPermute(0, 2, 1, 3).Process(ctx, attnMatmul)
	if err != nil {
		return nil, nil, fmt.Errorf("permute output: %w", err)
	}
	attnOut, err = tendo.NewReshape(batch, -1, m.Config.Dim).Process(ctx, attnOut)
	if err != nil {
		return nil, nil, fmt.Errorf("reshape output: %w", err)
	}

	// Output projection
	out, err := layer.OProj.Forward(ctx, attnOut, backend)
	attnMatmul.Free() // Free attention matmul result (attnOut is view of this)
	if err != nil {
		return nil, nil, fmt.Errorf("o projection: %w", err)
	}

	return out, newCache, nil
}

func (m *Model) forwardMLP(ctx context.Context, x *tendo.Tensor, layer *Layer, backend Backend) (*tendo.Tensor, error) {
	// SwiGLU: down(silu(gate(x)) * up(x))
	gateProj, err := layer.GateProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, fmt.Errorf("gate projection: %w", err)
	}
	gate, err := backend.SiLU(ctx, gateProj)
	gateProj.Free() // Free pre-activation
	if err != nil {
		return nil, fmt.Errorf("silu: %w", err)
	}

	up, err := layer.UpProj.Forward(ctx, x, backend)
	if err != nil {
		gate.Free()
		return nil, fmt.Errorf("up projection: %w", err)
	}

	h, err := backend.Mul(ctx, gate, up)
	gate.Free()
	up.Free()
	if err != nil {
		return nil, fmt.Errorf("gate * up: %w", err)
	}

	out, err := layer.DownProj.Forward(ctx, h, backend)
	h.Free()
	if err != nil {
		return nil, fmt.Errorf("down projection: %w", err)
	}

	return out, nil
}

// repeatKVHeads repeats K/V heads for GQA.
// Input: [batch, num_kv_heads, seq, head_dim]
// Output: [batch, num_heads, seq, head_dim] where num_heads = num_kv_heads * repeatFactor
func repeatKVHeads(ctx context.Context, x *tendo.Tensor, repeatFactor int) (*tendo.Tensor, error) {
	if repeatFactor == 1 {
		return x, nil
	}

	batch := x.Size(0)
	numKVHeads := x.Size(1)
	seq := x.Size(2)
	headDim := x.Size(3)

	// Expand: [batch, kv_heads, seq, head_dim] -> [batch, kv_heads, 1, seq, head_dim]
	x, err := tendo.NewReshape(batch, numKVHeads, 1, seq, headDim).Process(ctx, x)
	if err != nil {
		return nil, err
	}

	// Expand the middle dimension by repeating
	// [batch, kv_heads, 1, seq, head_dim] -> [batch, kv_heads, repeat, seq, head_dim]
	x, err = tendo.NewExpand(batch, numKVHeads, repeatFactor, seq, headDim).Process(ctx, x)
	if err != nil {
		return nil, err
	}

	// Reshape to merge: [batch, kv_heads * repeat, seq, head_dim]
	return tendo.NewReshape(batch, numKVHeads*repeatFactor, seq, headDim).Process(ctx, x)
}

func applyCausalMask(ctx context.Context, scores *tendo.Tensor, backend Backend) (*tendo.Tensor, error) {
	batch := scores.Size(0)
	heads := scores.Size(1)
	seqQ := scores.Size(-2)
	seqK := scores.Size(-1)

	ones, err := backend.Ones(seqQ, seqK)
	if err != nil {
		return nil, err
	}

	offset := seqK - seqQ
	mask2D, err := backend.Tril(ctx, ones, offset)
	ones.Free()
	if err != nil {
		return nil, err
	}

	// Expand mask to match scores shape: [seqQ, seqK] -> [batch, heads, seqQ, seqK]
	mask4D, err := tendo.NewReshape(1, 1, seqQ, seqK).Process(ctx, mask2D)
	if err != nil {
		mask2D.Free()
		return nil, err
	}
	maskExpanded, err := tendo.NewExpand(batch, heads, seqQ, seqK).Process(ctx, mask4D)
	if err != nil {
		mask2D.Free()
		return nil, err
	}
	// Force contiguous to materialize the expanded mask (creates new storage)
	mask := maskExpanded.Contiguous()
	// Now safe to free the original storage
	mask2D.Free()

	// Create negInf tensor matching scores shape
	negInf, err := backend.Full(float32(math.Inf(-1)), batch, heads, seqQ, seqK)
	if err != nil {
		mask.Free()
		return nil, err
	}

	result, err := backend.Where(ctx, mask, scores, negInf)
	mask.Free()
	negInf.Free()
	return result, err
}
