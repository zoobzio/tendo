package tendo

import (
	"fmt"
)

// Numel returns the total number of elements for a given shape.
func Numel(shape []int) int {
	if len(shape) == 0 {
		return 1 // scalar
	}
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

// ComputeStrides computes the strides for a contiguous tensor with the given shape.
// Strides are in row-major (C) order.
func ComputeStrides(shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}

	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// IsContiguous returns true if the given shape and stride represent contiguous memory.
func IsContiguous(shape, stride []int) bool {
	if len(shape) != len(stride) {
		return false
	}
	if len(shape) == 0 {
		return true
	}

	expected := ComputeStrides(shape)
	for i := range stride {
		if stride[i] != expected[i] {
			return false
		}
	}
	return true
}

// InferShape infers the full shape when one dimension is -1.
// Returns an error if more than one -1 is present or if the shape is invalid.
func InferShape(numel int, shape []int) ([]int, error) {
	result := make([]int, len(shape))
	inferIdx := -1
	product := 1

	for i, s := range shape {
		if s == -1 {
			if inferIdx != -1 {
				return nil, fmt.Errorf("can only infer one dimension, found -1 at indices %d and %d", inferIdx, i)
			}
			inferIdx = i
		} else if s <= 0 {
			return nil, fmt.Errorf("invalid dimension %d at index %d", s, i)
		} else {
			result[i] = s
			product *= s
		}
	}

	if inferIdx == -1 {
		copy(result, shape)
		return result, nil
	}

	if numel%product != 0 {
		return nil, fmt.Errorf("cannot infer dimension: %d elements not divisible by %d", numel, product)
	}

	result[inferIdx] = numel / product
	return result, nil
}

// BroadcastShapes computes the broadcast shape of two shapes.
// Returns an error if the shapes are not broadcastable.
func BroadcastShapes(a, b []int) ([]int, error) {
	if !CanBroadcast(a, b) {
		return nil, fmt.Errorf("shapes %v and %v are not broadcastable", a, b)
	}

	// Result has the max number of dimensions
	maxDim := len(a)
	if len(b) > maxDim {
		maxDim = len(b)
	}

	result := make([]int, maxDim)

	// Work from the right
	for i := 0; i < maxDim; i++ {
		ai := len(a) - 1 - i
		bi := len(b) - 1 - i
		ri := maxDim - 1 - i

		var da, db = 1, 1
		if ai >= 0 {
			da = a[ai]
		}
		if bi >= 0 {
			db = b[bi]
		}

		if da == db {
			result[ri] = da
		} else if da == 1 {
			result[ri] = db
		} else if db == 1 {
			result[ri] = da
		}
		// CanBroadcast already verified this is valid
	}

	return result, nil
}

// CanBroadcast returns true if two shapes can be broadcast together.
func CanBroadcast(a, b []int) bool {
	// Work from the right
	for i := 0; i < len(a) || i < len(b); i++ {
		ai := len(a) - 1 - i
		bi := len(b) - 1 - i

		var da, db = 1, 1
		if ai >= 0 {
			da = a[ai]
		}
		if bi >= 0 {
			db = b[bi]
		}

		if da != db && da != 1 && db != 1 {
			return false
		}
	}
	return true
}

// ValidateMatMul validates shapes for matrix multiplication and returns the output shape.
// Supports 2D matrices and batched matrix multiplication.
func ValidateMatMul(a, b []int) ([]int, error) {
	if len(a) < 2 || len(b) < 2 {
		return nil, fmt.Errorf("matmul requires at least 2D tensors, got %dD and %dD", len(a), len(b))
	}

	// Get the matrix dimensions (last two dims)
	m, k1 := a[len(a)-2], a[len(a)-1]
	k2, n := b[len(b)-2], b[len(b)-1]

	if k1 != k2 {
		return nil, fmt.Errorf("matmul dimension mismatch: %v x %v (inner dims %d != %d)", a, b, k1, k2)
	}

	// Handle batch dimensions
	batchA := a[:len(a)-2]
	batchB := b[:len(b)-2]

	batchOut, err := BroadcastShapes(batchA, batchB)
	if err != nil {
		return nil, fmt.Errorf("matmul batch dimensions not broadcastable: %w", err)
	}

	result := make([]int, len(batchOut)+2)
	copy(result, batchOut)
	result[len(result)-2] = m
	result[len(result)-1] = n

	return result, nil
}

// ValidateElementwise validates shapes for element-wise operations and returns the output shape.
func ValidateElementwise(a, b []int) ([]int, error) {
	return BroadcastShapes(a, b)
}

// TransposeShape returns the shape after transposing two dimensions.
func TransposeShape(shape []int, dim0, dim1 int) ([]int, error) {
	dim0, dim1, err := normalizeDims(len(shape), dim0, dim1)
	if err != nil {
		return nil, err
	}

	result := make([]int, len(shape))
	copy(result, shape)
	result[dim0], result[dim1] = result[dim1], result[dim0]
	return result, nil
}

// TransposeStride returns the strides after transposing two dimensions.
func TransposeStride(stride []int, dim0, dim1 int) ([]int, error) {
	dim0, dim1, err := normalizeDims(len(stride), dim0, dim1)
	if err != nil {
		return nil, err
	}

	result := make([]int, len(stride))
	copy(result, stride)
	result[dim0], result[dim1] = result[dim1], result[dim0]
	return result, nil
}

// SqueezeShape removes a dimension of size 1 from the shape.
func SqueezeShape(shape []int, dim int) ([]int, error) {
	if dim < 0 {
		dim = len(shape) + dim
	}
	if dim < 0 || dim >= len(shape) {
		return nil, fmt.Errorf("dimension %d out of range for shape %v", dim, shape)
	}
	if shape[dim] != 1 {
		return nil, fmt.Errorf("cannot squeeze dimension %d with size %d (must be 1)", dim, shape[dim])
	}

	result := make([]int, 0, len(shape)-1)
	result = append(result, shape[:dim]...)
	result = append(result, shape[dim+1:]...)
	return result, nil
}

// UnsqueezeShape inserts a dimension of size 1 at the given position.
func UnsqueezeShape(shape []int, dim int) ([]int, error) {
	if dim < 0 {
		dim = len(shape) + dim + 1
	}
	if dim < 0 || dim > len(shape) {
		return nil, fmt.Errorf("dimension %d out of range for unsqueeze on shape %v", dim, shape)
	}

	result := make([]int, len(shape)+1)
	copy(result[:dim], shape[:dim])
	result[dim] = 1
	copy(result[dim+1:], shape[dim:])
	return result, nil
}

// normalizeDims normalizes two dimension indices and validates them.
func normalizeDims(ndim, dim0, dim1 int) (int, int, error) {
	if dim0 < 0 {
		dim0 = ndim + dim0
	}
	if dim1 < 0 {
		dim1 = ndim + dim1
	}
	if dim0 < 0 || dim0 >= ndim {
		return 0, 0, fmt.Errorf("dimension %d out of range for %d dimensions", dim0, ndim)
	}
	if dim1 < 0 || dim1 >= ndim {
		return 0, 0, fmt.Errorf("dimension %d out of range for %d dimensions", dim1, ndim)
	}
	return dim0, dim1, nil
}

// FlatToStrided converts a flat (contiguous) index to a strided storage index.
// This handles non-contiguous tensors by computing the physical storage location
// from a logical element position.
func FlatToStrided(flat int, shape, stride []int, offset int) int {
	idx := offset
	tmp := flat
	for d := len(shape) - 1; d >= 0; d-- {
		coord := tmp % shape[d]
		tmp /= shape[d]
		idx += coord * stride[d]
	}
	return idx
}

// FlatToCoords converts a flat index to multi-dimensional coordinates.
func FlatToCoords(flat int, shape []int) []int {
	coords := make([]int, len(shape))
	tmp := flat
	for d := len(shape) - 1; d >= 0; d-- {
		coords[d] = tmp % shape[d]
		tmp /= shape[d]
	}
	return coords
}

// CoordsToFlat converts multi-dimensional coordinates to a flat index using strides.
func CoordsToFlat(coords, strides []int) int {
	idx := 0
	for d := 0; d < len(coords); d++ {
		idx += coords[d] * strides[d]
	}
	return idx
}

// IterateStrided calls fn for each element, providing both the logical flat index
// and the physical strided storage index. Useful for iterating non-contiguous tensors.
func IterateStrided(numel int, shape, stride []int, offset int, fn func(flatIdx, stridedIdx int)) {
	for i := 0; i < numel; i++ {
		fn(i, FlatToStrided(i, shape, stride, offset))
	}
}

// FlatToReducedIndex maps an input flat index to an output index after removing
// specified dimensions. strides should be from ComputeStrides(shape) for contiguous data.
// skipDims is a set of dimensions to exclude from the output index.
func FlatToReducedIndex(flat int, shape, strides, outStrides []int, skipDims map[int]bool) int {
	outIdx := 0
	outDimIdx := 0
	tmp := flat
	for d := 0; d < len(shape); d++ {
		coord := tmp / strides[d]
		tmp %= strides[d]
		if !skipDims[d] {
			if outDimIdx < len(outStrides) {
				outIdx += coord * outStrides[outDimIdx]
			}
			outDimIdx++
		}
	}
	return outIdx
}

// ReduceShapeAndStrides computes the output shape and strides after removing dimensions.
func ReduceShapeAndStrides(shape []int, skipDims map[int]bool) (outShape, outStrides []int) {
	for i, s := range shape {
		if !skipDims[i] {
			outShape = append(outShape, s)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{}
	}
	outStrides = ComputeStrides(outShape)
	return
}

// SliceBaseIndex computes the base storage index for a slice along a dimension.
// sliceIdx is the index in the reduced shape (with dim removed), and the returned
// baseIdx is the starting position in storage for iterating along dim.
func SliceBaseIndex(sliceIdx int, shape, strides []int, dim int) int {
	baseIdx := 0
	tmp := sliceIdx
	// Build reduced shape (excluding dim)
	reducedIdx := 0
	for d := len(shape) - 1; d >= 0; d-- {
		if d == dim {
			continue
		}
		// Compute size of remaining reduced dimensions
		reducedSize := 1
		for dd := d + 1; dd < len(shape); dd++ {
			if dd != dim {
				reducedSize *= shape[dd]
			}
		}
		if reducedSize > 0 {
			coord := tmp / reducedSize
			tmp %= reducedSize
			baseIdx += coord * strides[d]
		}
		reducedIdx++
	}
	return baseIdx
}

// IterateSlices iterates over all slices along a dimension.
// For each slice, calls fn with the base storage index and dim stride.
// Use: for j := 0; j < shape[dim]; j++ { idx := baseIdx + j*dimStride }.
func IterateSlices(shape, strides []int, dim int, fn func(baseIdx, dimStride int)) {
	dimSize := shape[dim]
	numSlices := Numel(shape) / dimSize
	dimStride := strides[dim]

	for i := 0; i < numSlices; i++ {
		baseIdx := SliceBaseIndex(i, shape, strides, dim)
		fn(baseIdx, dimStride)
	}
}
