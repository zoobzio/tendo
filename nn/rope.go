package nn

import (
	"context"
	"fmt"
	"math"

	"github.com/zoobzio/tendo"
)

// RoPE implements Rotary Position Embeddings.
// It precomputes sin/cos frequencies for efficient application.
type RoPE struct {
	Dim       int
	MaxSeqLen int
	Base      float32

	// Precomputed frequencies: [max_seq, dim/2]
	CosCache *tendo.Tensor
	SinCache *tendo.Tensor
}

// NewRoPE creates a RoPE instance with precomputed frequencies.
func NewRoPE(dim, maxSeqLen int, base float32) (*RoPE, error) {
	if dim%2 != 0 {
		return nil, fmt.Errorf("nn.RoPE: dim must be even, got %d", dim)
	}

	// Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
	halfDim := dim / 2
	invFreq := make([]float32, halfDim)
	for i := 0; i < halfDim; i++ {
		invFreq[i] = 1.0 / float32(math.Pow(float64(base), float64(2*i)/float64(dim)))
	}

	// Compute position indices: [0, 1, 2, ..., maxSeqLen-1]
	// Then compute freqs[pos, i] = pos * invFreq[i]
	cosData := make([]float32, maxSeqLen*halfDim)
	sinData := make([]float32, maxSeqLen*halfDim)

	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < halfDim; i++ {
			freq := float32(pos) * invFreq[i]
			cosData[pos*halfDim+i] = float32(math.Cos(float64(freq)))
			sinData[pos*halfDim+i] = float32(math.Sin(float64(freq)))
		}
	}

	cosCache, err := tendo.FromSlice(cosData, maxSeqLen, halfDim)
	if err != nil {
		return nil, fmt.Errorf("nn.RoPE: create cos cache: %w", err)
	}
	sinCache, err := tendo.FromSlice(sinData, maxSeqLen, halfDim)
	if err != nil {
		return nil, fmt.Errorf("nn.RoPE: create sin cache: %w", err)
	}

	return &RoPE{
		Dim:       dim,
		MaxSeqLen: maxSeqLen,
		Base:      base,
		CosCache:  cosCache,
		SinCache:  sinCache,
	}, nil
}

// RoPEBackend defines operations needed for RoPE application.
type RoPEBackend interface {
	Mul(ctx context.Context, a, b *tendo.Tensor) (*tendo.Tensor, error)
	Add(ctx context.Context, a, b *tendo.Tensor) (*tendo.Tensor, error)
	Sub(ctx context.Context, a, b *tendo.Tensor) (*tendo.Tensor, error)
	Cat(ctx context.Context, tensors []*tendo.Tensor, dim int) (*tendo.Tensor, error)
}

// StorageBackend defines what's needed to copy tensors to a device.
type StorageBackend interface {
	CopyFrom(t *tendo.Tensor) (*tendo.Tensor, error)
}

// ToDevice copies the RoPE caches to the target device.
// Returns a new RoPE with caches on the target device.
func (r *RoPE) ToDevice(backend StorageBackend) (*RoPE, error) {
	cosCache, err := backend.CopyFrom(r.CosCache)
	if err != nil {
		return nil, fmt.Errorf("nn.RoPE: copy cos cache: %w", err)
	}
	sinCache, err := backend.CopyFrom(r.SinCache)
	if err != nil {
		return nil, fmt.Errorf("nn.RoPE: copy sin cache: %w", err)
	}
	return &RoPE{
		Dim:       r.Dim,
		MaxSeqLen: r.MaxSeqLen,
		Base:      r.Base,
		CosCache:  cosCache,
		SinCache:  sinCache,
	}, nil
}

// Apply applies rotary embeddings to query and key tensors.
// q, k shape: [batch, heads, seq, head_dim]
// posOffset: starting position (for KV cache scenarios)
// Returns rotated q, k with same shapes.
func (r *RoPE) Apply(ctx context.Context, q, k *tendo.Tensor, posOffset int, backend RoPEBackend) (*tendo.Tensor, *tendo.Tensor, error) {
	seq := q.Size(2)
	headDim := q.Size(3)

	if headDim != r.Dim {
		return nil, nil, fmt.Errorf("nn.RoPE: head_dim %d != rope dim %d", headDim, r.Dim)
	}
	if posOffset+seq > r.MaxSeqLen {
		return nil, nil, fmt.Errorf("nn.RoPE: position %d + seq %d exceeds max %d", posOffset, seq, r.MaxSeqLen)
	}

	// Get cos/sin for this sequence range: [seq, dim/2]
	cos, err := tendo.NewNarrow(0, posOffset, seq).Process(ctx, r.CosCache)
	if err != nil {
		return nil, nil, fmt.Errorf("nn.RoPE: narrow cos: %w", err)
	}
	sin, err := tendo.NewNarrow(0, posOffset, seq).Process(ctx, r.SinCache)
	if err != nil {
		return nil, nil, fmt.Errorf("nn.RoPE: narrow sin: %w", err)
	}

	// Apply rotation to q and k
	qRot, err := r.rotateHalf(ctx, q, cos, sin, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("nn.RoPE: rotate q: %w", err)
	}
	kRot, err := r.rotateHalf(ctx, k, cos, sin, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("nn.RoPE: rotate k: %w", err)
	}

	return qRot, kRot, nil
}

// rotateHalf applies the rotation to a tensor.
// x shape: [batch, heads, seq, head_dim]
// cos, sin shape: [seq, head_dim/2]
// Formula: x_rot = [x0*cos - x1*sin, x0*sin + x1*cos]
// where x0, x1 are the first and second halves of head_dim
func (r *RoPE) rotateHalf(ctx context.Context, x *tendo.Tensor, cos, sin *tendo.Tensor, backend RoPEBackend) (*tendo.Tensor, error) {
	halfDim := r.Dim / 2

	// Split x into two halves along last dimension
	// x: [batch, heads, seq, head_dim] -> x0, x1: [batch, heads, seq, head_dim/2]
	x0, err := tendo.NewNarrow(3, 0, halfDim).Process(ctx, x)
	if err != nil {
		return nil, err
	}
	x1, err := tendo.NewNarrow(3, halfDim, halfDim).Process(ctx, x)
	if err != nil {
		return nil, err
	}

	// Reshape cos/sin for broadcasting: [seq, half] -> [1, 1, seq, half]
	seq := cos.Size(0)
	cos, err = tendo.NewReshape(1, 1, seq, halfDim).Process(ctx, cos)
	if err != nil {
		return nil, err
	}
	sin, err = tendo.NewReshape(1, 1, seq, halfDim).Process(ctx, sin)
	if err != nil {
		return nil, err
	}

	// Compute rotations:
	// out0 = x0 * cos - x1 * sin
	// out1 = x0 * sin + x1 * cos
	x0Cos, err := backend.Mul(ctx, x0, cos)
	if err != nil {
		return nil, err
	}
	x1Sin, err := backend.Mul(ctx, x1, sin)
	if err != nil {
		return nil, err
	}
	out0, err := backend.Sub(ctx, x0Cos, x1Sin)
	if err != nil {
		return nil, err
	}

	x0Sin, err := backend.Mul(ctx, x0, sin)
	if err != nil {
		return nil, err
	}
	x1Cos, err := backend.Mul(ctx, x1, cos)
	if err != nil {
		return nil, err
	}
	out1, err := backend.Add(ctx, x0Sin, x1Cos)
	if err != nil {
		return nil, err
	}

	// Concatenate halves back: [batch, heads, seq, head_dim]
	return backend.Cat(ctx, []*tendo.Tensor{out0, out1}, 3)
}
