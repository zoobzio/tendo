package nn

import (
	"context"
	"fmt"
	"math"

	"github.com/zoobzio/tendo"
)

// AttentionBackend defines the operations needed for attention.
type AttentionBackend interface {
	LinearBackend
	Softmax(ctx context.Context, t *tendo.Tensor, dim int) (*tendo.Tensor, error)
	Mul(ctx context.Context, a, b *tendo.Tensor) (*tendo.Tensor, error)
	Full(value float32, shape ...int) (*tendo.Tensor, error)
	Cat(ctx context.Context, tensors []*tendo.Tensor, dim int) (*tendo.Tensor, error)
}

// KVCache holds cached key and value tensors for autoregressive generation.
type KVCache struct {
	K *tendo.Tensor // [batch, heads, cached_seq, head_dim]
	V *tendo.Tensor // [batch, heads, cached_seq, head_dim]
}

// Attention represents multi-head self-attention.
type Attention struct {
	QProj *Linear // query projection
	KProj *Linear // key projection
	VProj *Linear // value projection
	OProj *Linear // output projection

	NumHeads int
	HeadDim  int
	Scale    float32 // 1/sqrt(head_dim)
}

// AttentionConfig configures an Attention layer.
type AttentionConfig struct {
	Dim      int  // model dimension
	NumHeads int  // number of attention heads
	Bias     bool // whether projections have bias
}

// NewAttention creates an Attention layer from weight tensors.
// Weights should have shapes:
//   - qWeight, kWeight, vWeight: [dim, dim]
//   - oWeight: [dim, dim]
//   - biases (optional): [dim]
func NewAttention(cfg AttentionConfig, qWeight, kWeight, vWeight, oWeight *tendo.Tensor, qBias, kBias, vBias, oBias *tendo.Tensor) (*Attention, error) {
	if cfg.Dim%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("nn.Attention: dim %d not divisible by num_heads %d", cfg.Dim, cfg.NumHeads)
	}

	headDim := cfg.Dim / cfg.NumHeads

	qProj, err := NewLinear(qWeight, qBias)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: q projection: %w", err)
	}
	kProj, err := NewLinear(kWeight, kBias)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: k projection: %w", err)
	}
	vProj, err := NewLinear(vWeight, vBias)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: v projection: %w", err)
	}
	oProj, err := NewLinear(oWeight, oBias)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: o projection: %w", err)
	}

	return &Attention{
		QProj:    qProj,
		KProj:    kProj,
		VProj:    vProj,
		OProj:    oProj,
		NumHeads: cfg.NumHeads,
		HeadDim:  headDim,
		Scale:    1.0 / float32(math.Sqrt(float64(headDim))),
	}, nil
}

// AttentionOutput contains the attention output and optional updated cache.
type AttentionOutput struct {
	Hidden  *tendo.Tensor
	KVCache *KVCache
}

// Forward computes multi-head self-attention.
// Input shape: [batch, seq, dim]
// Output shape: [batch, seq, dim]
// If cache is provided, keys/values are appended and the updated cache is returned.
// If causal is true, applies causal masking (each position can only attend to earlier positions).
func (a *Attention) Forward(ctx context.Context, x *tendo.Tensor, cache *KVCache, causal bool, backend AttentionBackend) (*AttentionOutput, error) {
	batch := x.Size(0)
	seq := x.Size(1)

	// Project to Q, K, V: [batch, seq, dim]
	q, err := a.QProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: q projection: %w", err)
	}
	k, err := a.KProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: k projection: %w", err)
	}
	v, err := a.VProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: v projection: %w", err)
	}

	// Reshape to [batch, seq, heads, head_dim]
	q, err = tendo.NewReshape(batch, seq, a.NumHeads, a.HeadDim).Process(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: reshape q: %w", err)
	}
	k, err = tendo.NewReshape(batch, seq, a.NumHeads, a.HeadDim).Process(ctx, k)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: reshape k: %w", err)
	}
	v, err = tendo.NewReshape(batch, seq, a.NumHeads, a.HeadDim).Process(ctx, v)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: reshape v: %w", err)
	}

	// Permute to [batch, heads, seq, head_dim]
	q, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: permute q: %w", err)
	}
	k, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, k)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: permute k: %w", err)
	}
	v, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, v)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: permute v: %w", err)
	}

	// Handle KV cache
	var newCache *KVCache
	if cache != nil {
		// Concatenate with cached K, V along sequence dimension (dim 2)
		// cache.K/V come first (past), then new k/v
		k, err = backend.Cat(ctx, []*tendo.Tensor{cache.K, k}, 2)
		if err != nil {
			return nil, fmt.Errorf("nn.Attention: cat cached k: %w", err)
		}
		v, err = backend.Cat(ctx, []*tendo.Tensor{cache.V, v}, 2)
		if err != nil {
			return nil, fmt.Errorf("nn.Attention: cat cached v: %w", err)
		}
	}
	// Store new cache
	newCache = &KVCache{K: k, V: v}

	// Compute attention scores: Q @ K^T -> [batch, heads, seq_q, seq_k]
	kT, err := tendo.NewTranspose(-2, -1).Process(ctx, k)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: transpose k: %w", err)
	}

	scores, err := backend.MatMul(ctx, q, kT)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: q @ k^T: %w", err)
	}

	// Scale scores
	scaleTensor, err := backend.Full(a.Scale, 1)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: create scale tensor: %w", err)
	}
	scores, err = backend.Mul(ctx, scores, scaleTensor)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: scale scores: %w", err)
	}

	// Apply causal mask if needed
	if causal {
		scores, err = applyCausalMask(ctx, scores, backend)
		if err != nil {
			return nil, fmt.Errorf("nn.Attention: causal mask: %w", err)
		}
	}

	// Softmax over last dimension (keys)
	attnWeights, err := backend.Softmax(ctx, scores, -1)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: softmax: %w", err)
	}

	// Attention output: weights @ V -> [batch, heads, seq, head_dim]
	attnOut, err := backend.MatMul(ctx, attnWeights, v)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: weights @ v: %w", err)
	}

	// Permute back to [batch, seq, heads, head_dim]
	attnOut, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, attnOut)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: permute output: %w", err)
	}

	// Reshape to [batch, seq, dim]
	dim := a.NumHeads * a.HeadDim
	attnOut, err = tendo.NewReshape(batch, -1, dim).Process(ctx, attnOut)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: reshape output: %w", err)
	}

	// Output projection
	out, err := a.OProj.Forward(ctx, attnOut, backend)
	if err != nil {
		return nil, fmt.Errorf("nn.Attention: output projection: %w", err)
	}

	return &AttentionOutput{
		Hidden:  out,
		KVCache: newCache,
	}, nil
}

// CausalMaskBackend defines operations needed for causal masking.
type CausalMaskBackend interface {
	Tril(ctx context.Context, t *tendo.Tensor, k int) (*tendo.Tensor, error)
	Where(ctx context.Context, condition, x, y *tendo.Tensor) (*tendo.Tensor, error)
	Ones(shape ...int) (*tendo.Tensor, error)
	Full(value float32, shape ...int) (*tendo.Tensor, error)
}

// applyCausalMask applies a causal (lower triangular) mask to attention scores.
// Positions that should be masked get -inf (so softmax gives 0).
func applyCausalMask(ctx context.Context, scores *tendo.Tensor, backend interface{}) (*tendo.Tensor, error) {
	maskBackend, ok := backend.(CausalMaskBackend)
	if !ok {
		return nil, fmt.Errorf("backend does not support causal masking operations")
	}

	// scores shape: [batch, heads, seq_q, seq_k]
	seqQ := scores.Size(-2)
	seqK := scores.Size(-1)

	// Create lower triangular mask [seq_q, seq_k]
	ones, err := maskBackend.Ones(seqQ, seqK)
	if err != nil {
		return nil, err
	}

	// For causal: each query position can attend to keys at positions <= query position
	// When seq_q < seq_k (with KV cache), we need offset: k = seq_k - seq_q
	offset := seqK - seqQ
	mask, err := maskBackend.Tril(ctx, ones, offset)
	if err != nil {
		return nil, err
	}

	// Create -inf tensor for masked positions
	negInf, err := maskBackend.Full(float32(math.Inf(-1)), 1)
	if err != nil {
		return nil, err
	}

	// Where mask==1, keep scores; where mask==0, use -inf
	return maskBackend.Where(ctx, mask, scores, negInf)
}
