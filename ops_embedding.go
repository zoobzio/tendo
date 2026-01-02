package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// Embedding is a chainable operator that performs embedding lookup.
// Weight shape: [vocab_size, embed_dim]
// Input (indices) shape: arbitrary (e.g., [batch, seq_len])
// Output shape: input.shape + [embed_dim]
//
// The input tensor to Process should contain integer indices (stored as float32).
// Each index looks up the corresponding row in the weight matrix.
type Embedding struct {
	backend  EmbeddingOps
	weight   *Tensor
	identity pipz.Identity
}

// NewEmbedding creates an Embedding operator.
// weight is the embedding table with shape [vocab_size, embed_dim].
func NewEmbedding(backend EmbeddingOps, weight *Tensor) *Embedding {
	return &Embedding{
		identity: IdentityEmbedding,
		backend:  backend,
		weight:   weight,
	}
}

// Process performs embedding lookup.
// The input tensor contains indices to look up in the weight matrix.
func (e *Embedding) Process(ctx context.Context, indices *Tensor) (*Tensor, error) {
	result, err := e.backend.Embedding(ctx, e.weight, indices)
	if err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}

	emitWithTrace(ctx, OpEmbedding,
		KeyWeight.Field(e.weight),
		KeyIndices.Field(indices),
		KeyOutput.Field(result),
	)

	propagateTape(indices, result, "embedding", map[string]*Tensor{"weight": e.weight, "indices": indices})

	return result, nil
}

// Identity returns the operator identity.
func (e *Embedding) Identity() pipz.Identity { return e.identity }

// Schema returns the operator schema node.
func (e *Embedding) Schema() pipz.Node {
	return pipz.Node{Identity: e.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (e *Embedding) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Embedding)(nil)
