package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// Reshape is a chainable operator that reshapes a tensor.
// The total number of elements must remain the same.
// A single dimension can be -1, which will be inferred.
// Backend-agnostic: creates a view or copies if non-contiguous.
type Reshape struct {
	identity pipz.Identity
	shape    []int
}

// NewReshape creates a Reshape operator.
func NewReshape(shape ...int) *Reshape {
	return &Reshape{
		identity: IdentityReshape,
		shape:    shape,
	}
}

// Process reshapes the tensor.
func (r *Reshape) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	newShape, err := InferShape(t.Numel(), r.shape)
	if err != nil {
		return nil, fmt.Errorf("reshape: %w", err)
	}

	if Numel(newShape) != t.Numel() {
		return nil, &ShapeError{
			Op:      "reshape",
			ShapeA:  t.Shape(),
			ShapeB:  newShape,
			Message: fmt.Sprintf("cannot reshape %v to %v: element count mismatch", t.Shape(), newShape),
		}
	}

	// If contiguous, create a view
	if t.IsContiguous() {
		result := &Tensor{
			storage: t.storage,
			shape:   newShape,
			stride:  ComputeStrides(newShape),
			offset:  t.offset,
		}

		emitWithTrace(ctx, OpReshape,
			KeyInput.Field(t),
			KeyOutput.Field(result),
			KeyShape.Field(newShape),
		)

		propagateTape(t, result, "reshape", nil)

		return result, nil
	}

	// Non-contiguous: must copy data
	result := t.Contiguous()
	result.shape = newShape
	result.stride = ComputeStrides(newShape)

	emitWithTrace(ctx, OpReshape,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyShape.Field(newShape),
	)

	propagateTape(t, result, "reshape", nil)

	return result, nil
}

// Identity returns the operator identity.
func (r *Reshape) Identity() pipz.Identity { return r.identity }

// Schema returns the operator schema.
func (r *Reshape) Schema() pipz.Node { return pipz.Node{Identity: r.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (r *Reshape) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Reshape)(nil)

// View is a chainable operator that creates a view with the given shape.
// The tensor must be contiguous. Use Reshape for non-contiguous tensors.
// Backend-agnostic.
type View struct {
	identity pipz.Identity
	shape    []int
}

// NewView creates a View operator.
func NewView(shape ...int) *View {
	return &View{
		identity: IdentityView,
		shape:    shape,
	}
}

// Process creates a view of the tensor.
func (v *View) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	if !t.IsContiguous() {
		return nil, &ShapeError{
			Op:      "view",
			ShapeA:  t.Shape(),
			Message: "view requires contiguous tensor",
		}
	}

	newShape, err := InferShape(t.Numel(), v.shape)
	if err != nil {
		return nil, fmt.Errorf("view: %w", err)
	}

	if Numel(newShape) != t.Numel() {
		return nil, &ShapeError{
			Op:      "view",
			ShapeA:  t.Shape(),
			ShapeB:  newShape,
			Message: fmt.Sprintf("cannot view %v as %v: element count mismatch", t.Shape(), newShape),
		}
	}

	result := &Tensor{
		storage: t.storage,
		shape:   newShape,
		stride:  ComputeStrides(newShape),
		offset:  t.offset,
	}

	emitWithTrace(ctx, OpReshape,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyShape.Field(newShape),
	)

	propagateTape(t, result, "view", nil)

	return result, nil
}

// Identity returns the operator identity.
func (v *View) Identity() pipz.Identity { return v.identity }

// Schema returns the operator schema.
func (v *View) Schema() pipz.Node { return pipz.Node{Identity: v.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (v *View) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*View)(nil)

// Squeeze is a chainable operator that removes dimensions of size 1.
// If dim is specified, only that dimension is squeezed (if size 1).
// Backend-agnostic.
type Squeeze struct {
	dim      *int // nil means squeeze all dimensions of size 1
	identity pipz.Identity
}

// NewSqueeze creates a Squeeze operator that squeezes all size-1 dimensions.
func NewSqueeze() *Squeeze {
	return &Squeeze{
		identity: IdentitySqueeze,
		dim:      nil,
	}
}

// NewSqueezeDim creates a Squeeze operator that squeezes a specific dimension.
func NewSqueezeDim(dim int) *Squeeze {
	return &Squeeze{
		identity: IdentitySqueeze,
		dim:      &dim,
	}
}

// Process squeezes the tensor.
func (s *Squeeze) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	shape := t.Shape()
	stride := t.Stride()

	var newShape, newStride []int

	if s.dim == nil {
		// Squeeze all dimensions of size 1
		for i, sz := range shape {
			if sz != 1 {
				newShape = append(newShape, sz)
				newStride = append(newStride, stride[i])
			}
		}
	} else {
		// Squeeze specific dimension
		d := *s.dim
		if d < 0 {
			d = len(shape) + d
		}
		if d < 0 || d >= len(shape) {
			return nil, &ShapeError{Op: "squeeze", Message: "dimension out of range"}
		}

		for i, sz := range shape {
			if i == d && sz == 1 {
				continue // skip this dimension
			}
			newShape = append(newShape, sz)
			newStride = append(newStride, stride[i])
		}
	}

	// Handle scalar case
	if len(newShape) == 0 {
		newShape = []int{}
		newStride = []int{}
	}

	result := &Tensor{
		storage: t.storage,
		shape:   newShape,
		stride:  newStride,
		offset:  t.offset,
	}

	emitWithTrace(ctx, OpSqueeze,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyShape.Field(newShape),
	)

	propagateTape(t, result, "squeeze", nil)

	return result, nil
}

// Identity returns the operator identity.
func (s *Squeeze) Identity() pipz.Identity { return s.identity }

// Schema returns the operator schema.
func (s *Squeeze) Schema() pipz.Node { return pipz.Node{Identity: s.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (s *Squeeze) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Squeeze)(nil)

// Unsqueeze is a chainable operator that inserts a dimension of size 1.
// Backend-agnostic.
type Unsqueeze struct {
	identity pipz.Identity
	dim      int
}

// NewUnsqueeze creates an Unsqueeze operator.
func NewUnsqueeze(dim int) *Unsqueeze {
	return &Unsqueeze{
		identity: IdentityUnsqueeze,
		dim:      dim,
	}
}

// Process unsqueezes the tensor.
func (u *Unsqueeze) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	shape := t.Shape()
	stride := t.Stride()
	dim := u.dim

	// Normalize dimension (can be 0 to len(shape) inclusive)
	if dim < 0 {
		dim = len(shape) + dim + 1
	}
	if dim < 0 || dim > len(shape) {
		return nil, &ShapeError{Op: "unsqueeze", Message: "dimension out of range"}
	}

	newShape := make([]int, len(shape)+1)
	newStride := make([]int, len(shape)+1)

	// Compute stride for new dimension
	newDimStride := 1
	if dim < len(shape) && len(stride) > 0 {
		newDimStride = stride[dim] * shape[dim]
	} else if len(stride) > 0 {
		newDimStride = 1
	}

	for i := 0; i < dim; i++ {
		newShape[i] = shape[i]
		newStride[i] = stride[i]
	}
	newShape[dim] = 1
	newStride[dim] = newDimStride
	for i := dim; i < len(shape); i++ {
		newShape[i+1] = shape[i]
		newStride[i+1] = stride[i]
	}

	result := &Tensor{
		storage: t.storage,
		shape:   newShape,
		stride:  newStride,
		offset:  t.offset,
	}

	emitWithTrace(ctx, OpUnsqueeze,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyShape.Field(newShape),
	)

	propagateTape(t, result, "unsqueeze", nil)

	return result, nil
}

// Identity returns the operator identity.
func (u *Unsqueeze) Identity() pipz.Identity { return u.identity }

// Schema returns the operator schema.
func (u *Unsqueeze) Schema() pipz.Node { return pipz.Node{Identity: u.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (u *Unsqueeze) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Unsqueeze)(nil)

// Flatten is a chainable operator that flattens dimensions from startDim to endDim.
// Backend-agnostic.
type Flatten struct {
	identity pipz.Identity
	startDim int
	endDim   int
}

// NewFlatten creates a Flatten operator.
func NewFlatten(startDim, endDim int) *Flatten {
	return &Flatten{
		identity: IdentityFlatten,
		startDim: startDim,
		endDim:   endDim,
	}
}

// Process flattens the tensor.
func (f *Flatten) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	shape := t.Shape()
	ndim := len(shape)
	startDim := f.startDim
	endDim := f.endDim

	// Normalize dimensions
	if startDim < 0 {
		startDim = ndim + startDim
	}
	if endDim < 0 {
		endDim = ndim + endDim
	}
	if startDim < 0 || startDim >= ndim || endDim < 0 || endDim >= ndim || startDim > endDim {
		return nil, &ShapeError{Op: "flatten", Message: "invalid dimension range"}
	}

	// Compute flattened size
	flatSize := 1
	for i := startDim; i <= endDim; i++ {
		flatSize *= shape[i]
	}

	// Build new shape
	newShape := make([]int, 0, ndim-(endDim-startDim))
	for i := 0; i < startDim; i++ {
		newShape = append(newShape, shape[i])
	}
	newShape = append(newShape, flatSize)
	for i := endDim + 1; i < ndim; i++ {
		newShape = append(newShape, shape[i])
	}

	// Must be contiguous for flatten
	src := t
	if !t.IsContiguous() {
		src = t.Contiguous()
	}

	result := &Tensor{
		storage: src.storage,
		shape:   newShape,
		stride:  ComputeStrides(newShape),
		offset:  src.offset,
	}

	emitWithTrace(ctx, OpReshape,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyShape.Field(newShape),
	)

	propagateTape(t, result, "flatten", nil)

	return result, nil
}

// Identity returns the operator identity.
func (f *Flatten) Identity() pipz.Identity { return f.identity }

// Schema returns the operator schema.
func (f *Flatten) Schema() pipz.Node { return pipz.Node{Identity: f.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (f *Flatten) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Flatten)(nil)

// Slice is a chainable operator that extracts a slice along a dimension.
// start is inclusive, end is exclusive. Backend-agnostic.
type Slice struct {
	identity pipz.Identity
	dim      int
	start    int
	end      int
}

// NewSlice creates a Slice operator.
func NewSlice(dim, start, end int) *Slice {
	return &Slice{
		identity: IdentitySlice,
		dim:      dim,
		start:    start,
		end:      end,
	}
}

// Process slices the tensor.
func (sl *Slice) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	shape := t.Shape()
	stride := t.Stride()
	dim := sl.dim
	start := sl.start
	end := sl.end

	// Normalize dimension
	if dim < 0 {
		dim = len(shape) + dim
	}
	if dim < 0 || dim >= len(shape) {
		return nil, &ShapeError{Op: "slice", Message: "dimension out of range"}
	}

	dimSize := shape[dim]

	// Normalize start/end
	if start < 0 {
		start = dimSize + start
	}
	if end < 0 {
		end = dimSize + end
	}
	if start < 0 {
		start = 0
	}
	if end > dimSize {
		end = dimSize
	}
	if start >= end {
		return nil, &ShapeError{Op: "slice", Message: "empty slice"}
	}

	// Compute new shape and offset
	newShape := make([]int, len(shape))
	copy(newShape, shape)
	newShape[dim] = end - start

	newOffset := t.offset + start*stride[dim]

	result := &Tensor{
		storage: t.storage,
		shape:   newShape,
		stride:  stride, // stride unchanged
		offset:  newOffset,
	}

	emitWithTrace(ctx, OpSlice,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyDim.Field(dim),
		KeyStart.Field(start),
		KeyEnd.Field(end),
	)

	propagateTape(t, result, "slice", nil)

	return result, nil
}

// Identity returns the operator identity.
func (sl *Slice) Identity() pipz.Identity { return sl.identity }

// Schema returns the operator schema.
func (sl *Slice) Schema() pipz.Node { return pipz.Node{Identity: sl.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (sl *Slice) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Slice)(nil)

// Narrow is an alias for Slice. Backend-agnostic.
type Narrow struct {
	identity pipz.Identity
	dim      int
	start    int
	length   int
}

// NewNarrow creates a Narrow operator.
func NewNarrow(dim, start, length int) *Narrow {
	return &Narrow{
		identity: IdentityNarrow,
		dim:      dim,
		start:    start,
		length:   length,
	}
}

// Process narrows the tensor.
func (n *Narrow) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	slice := NewSlice(n.dim, n.start, n.start+n.length)
	return slice.Process(ctx, t)
}

// Identity returns the operator identity.
func (n *Narrow) Identity() pipz.Identity { return n.identity }

// Schema returns the operator schema.
func (n *Narrow) Schema() pipz.Node { return pipz.Node{Identity: n.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (n *Narrow) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Narrow)(nil)

// Expand is a chainable operator that expands singleton dimensions.
// The tensor is not copied; stride is set to 0 for expanded dimensions.
// Backend-agnostic.
type Expand struct {
	identity pipz.Identity
	shape    []int
}

// NewExpand creates an Expand operator.
func NewExpand(shape ...int) *Expand {
	return &Expand{
		identity: IdentityExpand,
		shape:    shape,
	}
}

// Process expands the tensor.
func (e *Expand) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	srcShape := t.Shape()
	srcStride := t.Stride()

	// Shapes must be broadcastable
	if len(e.shape) < len(srcShape) {
		return nil, &ShapeError{
			Op:      "expand",
			ShapeA:  srcShape,
			ShapeB:  e.shape,
			Message: "expanded shape must have at least as many dimensions",
		}
	}

	// Pad source shape/stride with leading 1s
	offset := len(e.shape) - len(srcShape)
	paddedShape := make([]int, len(e.shape))
	paddedStride := make([]int, len(e.shape))
	for i := 0; i < offset; i++ {
		paddedShape[i] = 1
		paddedStride[i] = 0
	}
	for i := 0; i < len(srcShape); i++ {
		paddedShape[offset+i] = srcShape[i]
		paddedStride[offset+i] = srcStride[i]
	}

	// Compute new stride
	newStride := make([]int, len(e.shape))
	for i := 0; i < len(e.shape); i++ {
		switch paddedShape[i] {
		case e.shape[i]:
			newStride[i] = paddedStride[i]
		case 1:
			newStride[i] = 0 // broadcast
		default:
			return nil, &ShapeError{
				Op:      "expand",
				ShapeA:  srcShape,
				ShapeB:  e.shape,
				Message: fmt.Sprintf("cannot expand dimension %d from %d to %d", i, paddedShape[i], e.shape[i]),
			}
		}
	}

	result := &Tensor{
		storage: t.storage,
		shape:   e.shape,
		stride:  newStride,
		offset:  t.offset,
	}

	emitWithTrace(ctx, OpExpand,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyShape.Field(e.shape),
	)

	propagateTape(t, result, "expand", nil)

	return result, nil
}

// Identity returns the operator identity.
func (e *Expand) Identity() pipz.Identity { return e.identity }

// Schema returns the operator schema.
func (e *Expand) Schema() pipz.Node { return pipz.Node{Identity: e.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (e *Expand) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Expand)(nil)

// Permute is a chainable operator that permutes dimensions.
// Backend-agnostic.
type Permute struct {
	identity pipz.Identity
	dims     []int
}

// NewPermute creates a Permute operator.
func NewPermute(dims ...int) *Permute {
	return &Permute{
		identity: IdentityPermute,
		dims:     dims,
	}
}

// Process permutes the tensor dimensions.
func (p *Permute) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	shape := t.Shape()
	stride := t.Stride()

	if len(p.dims) != len(shape) {
		return nil, &ShapeError{
			Op:      "permute",
			Message: fmt.Sprintf("permute requires %d dimensions, got %d", len(shape), len(p.dims)),
		}
	}

	// Validate permutation
	seen := make([]bool, len(p.dims))
	for _, d := range p.dims {
		dim := d
		if dim < 0 {
			dim = len(shape) + dim
		}
		if dim < 0 || dim >= len(shape) || seen[dim] {
			return nil, &ShapeError{Op: "permute", Message: "invalid permutation"}
		}
		seen[dim] = true
	}

	newShape := make([]int, len(shape))
	newStride := make([]int, len(stride))
	for i, d := range p.dims {
		dim := d
		if dim < 0 {
			dim = len(shape) + dim
		}
		newShape[i] = shape[dim]
		newStride[i] = stride[dim]
	}

	result := &Tensor{
		storage: t.storage,
		shape:   newShape,
		stride:  newStride,
		offset:  t.offset,
	}

	emitWithTrace(ctx, OpPermute,
		KeyInput.Field(t),
		KeyOutput.Field(result),
		KeyShape.Field(newShape),
		KeyPermutation.Field(p.dims),
	)

	propagateTape(t, result, "permute", nil)

	return result, nil
}

// Identity returns the operator identity.
func (p *Permute) Identity() pipz.Identity { return p.identity }

// Schema returns the operator schema.
func (p *Permute) Schema() pipz.Node { return pipz.Node{Identity: p.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (p *Permute) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Permute)(nil)

// Cat is a chainable operator that concatenates tensors along a dimension.
// All tensors must have the same shape except in the concatenation dimension.
type Cat struct {
	identity pipz.Identity
	backend  ShapeOps
	tensors  []*Tensor
	dim      int
}

// NewCat creates a Cat operator.
// If backend is nil, uses CPU-only fallback implementation.
func NewCat(backend ShapeOps, tensors []*Tensor, dim int) *Cat {
	return &Cat{
		identity: IdentityCat,
		backend:  backend,
		tensors:  tensors,
		dim:      dim,
	}
}

// Process concatenates tensors.
func (c *Cat) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	// Prepend the input tensor to the list
	allTensors := make([]*Tensor, 0, len(c.tensors)+1)
	allTensors = append(allTensors, t)
	allTensors = append(allTensors, c.tensors...)

	if len(allTensors) == 0 {
		return nil, &ShapeError{Op: "cat", Message: "cat requires at least one tensor"}
	}

	// If backend is available, delegate to it
	if c.backend != nil {
		result, err := c.backend.Cat(ctx, allTensors, c.dim)
		if err != nil {
			return nil, fmt.Errorf("cat: %w", err)
		}

		emitWithTrace(ctx, OpCat,
			KeyInputs.Field(allTensors),
			KeyOutput.Field(result),
			KeyDim.Field(c.dim),
		)

		propagateTape(t, result, "cat", nil)

		return result, nil
	}

	// CPU-only fallback implementation
	shape := allTensors[0].Shape()
	ndim := len(shape)
	dim := c.dim

	// Normalize dimension
	if dim < 0 {
		dim = ndim + dim
	}
	if dim < 0 || dim >= ndim {
		return nil, &ShapeError{Op: "cat", Message: "dimension out of range"}
	}

	// Validate all tensors have compatible shapes
	totalSize := 0
	for i, tensor := range allTensors {
		tShape := tensor.Shape()
		if len(tShape) != ndim {
			return nil, &ShapeError{
				Op:      "cat",
				Message: fmt.Sprintf("tensor %d has %d dimensions, expected %d", i, len(tShape), ndim),
			}
		}
		for d := 0; d < ndim; d++ {
			if d == dim {
				totalSize += tShape[d]
			} else if tShape[d] != shape[d] {
				return nil, &ShapeError{
					Op:      "cat",
					Message: fmt.Sprintf("tensor %d has size %d at dim %d, expected %d", i, tShape[d], d, shape[d]),
				}
			}
		}
	}

	// Compute output shape
	outShape := make([]int, ndim)
	copy(outShape, shape)
	outShape[dim] = totalSize

	// Allocate output storage
	outNumel := Numel(outShape)
	outStorage := NewCPUStorageFromSlice(make([]float32, outNumel), allTensors[0].DType())
	outData := outStorage.Data()

	// Copy data from each tensor
	outStrides := ComputeStrides(outShape)
	offset := 0
	for _, tensor := range allTensors {
		src := tensor.Contiguous()
		srcCPU, ok := src.storage.(CPUDataAccessor)
		if !ok {
			return nil, &DeviceError{Expected: CPU, Got: tensor.Device().Type}
		}
		srcData := srcCPU.Data()
		srcShape := src.Shape()

		// Copy this tensor's data into the output
		copyTensorData(outData, srcData, outShape, srcShape, outStrides, dim, offset)
		offset += srcShape[dim]
	}

	result := NewTensor(outStorage, outShape, nil)

	emitWithTrace(ctx, OpCat,
		KeyInputs.Field(allTensors),
		KeyOutput.Field(result),
		KeyDim.Field(dim),
	)

	propagateTape(t, result, "cat", nil)

	return result, nil
}

// Identity returns the operator identity.
func (c *Cat) Identity() pipz.Identity { return c.identity }

// Schema returns the operator schema.
func (c *Cat) Schema() pipz.Node { return pipz.Node{Identity: c.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (c *Cat) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Cat)(nil)

// copyTensorData copies src tensor data into dst at the given offset along dim.
func copyTensorData(dst, src []float32, dstShape, srcShape []int, dstStrides []int, dim, offset int) {
	srcNumel := Numel(srcShape)

	for i := 0; i < srcNumel; i++ {
		// Convert flat index to multi-dimensional coordinates
		coords := FlatToCoords(i, srcShape)

		// Adjust coordinate for concat dimension
		coords[dim] += offset

		// Convert back to flat index in destination
		dstIdx := CoordsToFlat(coords, dstStrides)

		dst[dstIdx] = src[i]
	}
}

// Stack is a chainable operator that stacks tensors along a new dimension.
// All tensors must have identical shapes.
type Stack struct {
	identity pipz.Identity
	backend  ShapeOps
	tensors  []*Tensor
	dim      int
}

// NewStack creates a Stack operator.
// If backend is nil, uses CPU-only fallback implementation.
func NewStack(backend ShapeOps, tensors []*Tensor, dim int) *Stack {
	return &Stack{
		identity: IdentityStack,
		backend:  backend,
		tensors:  tensors,
		dim:      dim,
	}
}

// Process stacks tensors.
func (s *Stack) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	// Prepend the input tensor to the list
	allTensors := make([]*Tensor, 0, len(s.tensors)+1)
	allTensors = append(allTensors, t)
	allTensors = append(allTensors, s.tensors...)

	if len(allTensors) == 0 {
		return nil, &ShapeError{Op: "stack", Message: "stack requires at least one tensor"}
	}

	// If backend is available, delegate to it
	if s.backend != nil {
		result, err := s.backend.Stack(ctx, allTensors, s.dim)
		if err != nil {
			return nil, fmt.Errorf("stack: %w", err)
		}

		emitWithTrace(ctx, OpStack,
			KeyInputs.Field(allTensors),
			KeyOutput.Field(result),
			KeyDim.Field(s.dim),
		)

		propagateTape(t, result, "stack", nil)

		return result, nil
	}

	// CPU-only fallback implementation
	shape := allTensors[0].Shape()
	ndim := len(shape)
	dim := s.dim

	// Normalize dimension (can be 0 to ndim inclusive for new dim)
	if dim < 0 {
		dim = ndim + dim + 1
	}
	if dim < 0 || dim > ndim {
		return nil, &ShapeError{Op: "stack", Message: "dimension out of range"}
	}

	// Validate all tensors have identical shapes
	for i, tensor := range allTensors {
		tShape := tensor.Shape()
		if len(tShape) != ndim {
			return nil, &ShapeError{
				Op:      "stack",
				Message: fmt.Sprintf("tensor %d has %d dimensions, expected %d", i, len(tShape), ndim),
			}
		}
		for d := 0; d < ndim; d++ {
			if tShape[d] != shape[d] {
				return nil, &ShapeError{
					Op:      "stack",
					Message: fmt.Sprintf("tensor %d has size %d at dim %d, expected %d", i, tShape[d], d, shape[d]),
				}
			}
		}
	}

	// Compute output shape: insert new dimension at 'dim'
	outShape := make([]int, ndim+1)
	for d := 0; d < dim; d++ {
		outShape[d] = shape[d]
	}
	outShape[dim] = len(allTensors)
	for d := dim; d < ndim; d++ {
		outShape[d+1] = shape[d]
	}

	// Allocate output storage
	outNumel := Numel(outShape)
	outStorage := NewCPUStorageFromSlice(make([]float32, outNumel), allTensors[0].DType())
	outData := outStorage.Data()

	// Copy data from each tensor
	outStrides := ComputeStrides(outShape)
	tensorNumel := Numel(shape)
	srcStrides := ComputeStrides(shape)

	for ti, tensor := range allTensors {
		src := tensor.Contiguous()
		srcCPU, ok := src.storage.(CPUDataAccessor)
		if !ok {
			return nil, &DeviceError{Expected: CPU, Got: tensor.Device().Type}
		}
		srcData := srcCPU.Data()

		for i := 0; i < tensorNumel; i++ {
			// Convert flat index to coordinates in source
			coords := FlatToCoords(i, shape)

			// Build output coordinates with new dimension inserted at dim
			outCoords := make([]int, ndim+1)
			copy(outCoords[:dim], coords[:dim])
			outCoords[dim] = ti
			copy(outCoords[dim+1:], coords[dim:])

			// Convert to flat indices
			srcIdx := CoordsToFlat(coords, srcStrides)
			dstIdx := CoordsToFlat(outCoords, outStrides)

			outData[dstIdx] = srcData[srcIdx]
		}
	}

	result := NewTensor(outStorage, outShape, nil)

	emitWithTrace(ctx, OpStack,
		KeyInputs.Field(allTensors),
		KeyOutput.Field(result),
		KeyDim.Field(dim),
	)

	propagateTape(t, result, "stack", nil)

	return result, nil
}

// Identity returns the operator identity.
func (s *Stack) Identity() pipz.Identity { return s.identity }

// Schema returns the operator schema.
func (s *Stack) Schema() pipz.Node { return pipz.Node{Identity: s.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (s *Stack) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Stack)(nil)
