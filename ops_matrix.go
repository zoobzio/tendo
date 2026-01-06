package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// MatMul is a chainable operator that performs matrix multiplication.
// Supports 2D matrices and batched matrix multiplication.
type MatMul struct {
	backend  MatrixOps
	other    *Tensor
	identity pipz.Identity
}

// NewMatMul creates a MatMul operator.
func NewMatMul(backend MatrixOps, other *Tensor) *MatMul {
	return &MatMul{
		identity: IdentityMatMul,
		backend:  backend,
		other:    other,
	}
}

// Process performs matrix multiplication.
func (m *MatMul) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	result, err := m.backend.MatMul(ctx, t, m.other)
	if err != nil {
		return nil, fmt.Errorf("matmul: %w", err)
	}

	emitWithTrace(ctx, OpMatMul,
		KeyInputA.Field(t),
		KeyInputB.Field(m.other),
		KeyOutput.Field(result),
		KeyShape.Field(result.Shape()),
	)


	return result, nil
}

// Identity returns the operator identity.
func (m *MatMul) Identity() pipz.Identity { return m.identity }

// Schema returns the operator schema node.
func (m *MatMul) Schema() pipz.Node {
	return pipz.Node{Identity: m.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (m *MatMul) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*MatMul)(nil)

// Transpose is a chainable operator that transposes two dimensions.
// This is a backend-agnostic shape operation (creates a view).
type Transpose struct {
	identity pipz.Identity
	dim0     int
	dim1     int
}

// NewTranspose creates a Transpose operator.
func NewTranspose(dim0, dim1 int) *Transpose {
	return &Transpose{
		identity: IdentityTranspose,
		dim0:     dim0,
		dim1:     dim1,
	}
}

// Process transposes the specified dimensions.
func (tr *Transpose) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	newShape, err := TransposeShape(t.Shape(), tr.dim0, tr.dim1)
	if err != nil {
		return nil, err
	}

	newStride, err := TransposeStride(t.Stride(), tr.dim0, tr.dim1)
	if err != nil {
		return nil, err
	}

	// Create a view (shares storage)
	result := &Tensor{
		storage: t.storage,
		shape:   newShape,
		stride:  newStride,
		offset:  t.offset,
	}

	emitWithTrace(ctx, OpTranspose,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyShape.Field(newShape),
		KeyDim0.Field(tr.dim0),
		KeyDim1.Field(tr.dim1),
	)


	return result, nil
}

// Identity returns the operator identity.
func (tr *Transpose) Identity() pipz.Identity { return tr.identity }

// Schema returns the operator schema node.
func (tr *Transpose) Schema() pipz.Node {
	return pipz.Node{Identity: tr.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (tr *Transpose) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Transpose)(nil)

// T is a chainable operator that transposes the last two dimensions.
// Equivalent to Transpose(-2, -1). Backend-agnostic.
type T struct {
	identity pipz.Identity
}

// NewT creates a T operator.
func NewT() *T {
	return &T{
		identity: IdentityT,
	}
}

// Process transposes the last two dimensions.
func (t *T) Process(ctx context.Context, in *Tensor) (*Tensor, error) {
	if in.Dim() < 2 {
		return nil, fmt.Errorf("T requires at least 2 dimensions, got %d", in.Dim())
	}

	dim0 := in.Dim() - 2
	dim1 := in.Dim() - 1

	newShape, err := TransposeShape(in.Shape(), dim0, dim1)
	if err != nil {
		return nil, err
	}
	newStride, err := TransposeStride(in.Stride(), dim0, dim1)
	if err != nil {
		return nil, err
	}

	result := &Tensor{
		storage: in.storage,
		shape:   newShape,
		stride:  newStride,
		offset:  in.offset,
	}

	emitWithTrace(ctx, OpTranspose,
		KeyInput.Field(in),
		KeyOutput.Field(result),
		KeyShape.Field(newShape),
		KeyDim0.Field(dim0),
		KeyDim1.Field(dim1),
	)


	return result, nil
}

// Identity returns the operator identity.
func (t *T) Identity() pipz.Identity { return t.identity }

// Schema returns the operator schema node.
func (t *T) Schema() pipz.Node {
	return pipz.Node{Identity: t.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (t *T) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*T)(nil)
