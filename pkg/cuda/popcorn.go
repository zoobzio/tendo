package cuda

/*
#cgo CFLAGS: -I/home/zoobzio/code/popcorn/include
#cgo LDFLAGS: -L/home/zoobzio/code/popcorn/lib -lpopcorn -lcudart -lstdc++

#include <popcorn.h>
#include <stdlib.h>
*/
import "C"

import (
	"unsafe"

	"github.com/zoobzio/tendo"
)

// popcornError converts a popcorn status to a Go error.
func popcornError(status C.popcornStatus_t) error {
	if status == C.POPCORN_SUCCESS {
		return nil
	}
	return &CUDAError{
		Code:    int(status),
		Message: C.GoString(C.popcornGetErrorString(status)),
	}
}

// -----------------------------------------------------------------------------
// Unary Operations
// -----------------------------------------------------------------------------

// PopcornAbs computes element-wise absolute value.
func PopcornAbs(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornAbs_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornExp computes element-wise exponential.
func PopcornExp(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornExp_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornLog computes element-wise natural logarithm.
func PopcornLog(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornLog_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornSign computes element-wise sign (-1, 0, or 1).
func PopcornSign(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSign_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornGelu computes GELU activation.
func PopcornGelu(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornGelu_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornLeakyRelu computes Leaky ReLU activation with given alpha.
func PopcornLeakyRelu(t *tendo.Tensor, alpha float32) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornLeakyRelu_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.float(alpha),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// -----------------------------------------------------------------------------
// Binary Operations
// -----------------------------------------------------------------------------

// PopcornDiv computes element-wise division.
// Note: Requires tensors to have the same shape (no broadcasting).
func PopcornDiv(a, b *tendo.Tensor) (*tendo.Tensor, error) {
	aStorage, ok := a.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: a.Device().Type}
	}
	bStorage, ok := b.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: b.Device().Type}
	}

	if a.Numel() != b.Numel() {
		return nil, &tendo.ShapeError{Op: "div", Message: "tensors must have same number of elements"}
	}

	numel := a.Numel()
	outStorage, err := NewStorage(numel, a.DType(), aStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornDiv_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(aStorage.Ptr())),
		(*C.float)(unsafe.Pointer(bStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, a.Shape(), nil), nil
}

// PopcornPowTensor computes element-wise power (a^b).
// Note: Requires tensors to have the same shape (no broadcasting).
func PopcornPowTensor(a, b *tendo.Tensor) (*tendo.Tensor, error) {
	aStorage, ok := a.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: a.Device().Type}
	}
	bStorage, ok := b.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: b.Device().Type}
	}

	if a.Numel() != b.Numel() {
		return nil, &tendo.ShapeError{Op: "pow", Message: "tensors must have same number of elements"}
	}

	numel := a.Numel()
	outStorage, err := NewStorage(numel, a.DType(), aStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornPow_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(aStorage.Ptr())),
		(*C.float)(unsafe.Pointer(bStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, a.Shape(), nil), nil
}

// -----------------------------------------------------------------------------
// Scalar Operations
// -----------------------------------------------------------------------------

// PopcornPowScalar computes element-wise power with scalar exponent.
func PopcornPowScalar(t *tendo.Tensor, exp float32) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornPowScalar_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.float(exp),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// -----------------------------------------------------------------------------
// Comparison / Selection Operations
// -----------------------------------------------------------------------------

// PopcornClamp clamps tensor values to [min, max] range.
func PopcornClamp(t *tendo.Tensor, minVal, maxVal float32) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornClamp_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.float(minVal),
		C.float(maxVal),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornWhere selects elements based on condition.
// out[i] = cond[i] > 0 ? a[i] : b[i]
// Note: All tensors must have the same shape (no broadcasting).
func PopcornWhere(cond, a, b *tendo.Tensor) (*tendo.Tensor, error) {
	condStorage, ok := cond.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: cond.Device().Type}
	}
	aStorage, ok := a.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: a.Device().Type}
	}
	bStorage, ok := b.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: b.Device().Type}
	}

	numel := cond.Numel()
	if a.Numel() != numel || b.Numel() != numel {
		return nil, &tendo.ShapeError{Op: "where", Message: "all tensors must have same number of elements"}
	}

	outStorage, err := NewStorage(numel, a.DType(), aStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornWhere_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(condStorage.Ptr())),
		(*C.float)(unsafe.Pointer(aStorage.Ptr())),
		(*C.float)(unsafe.Pointer(bStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, cond.Shape(), nil), nil
}

// -----------------------------------------------------------------------------
// Gather / Index Operations
// -----------------------------------------------------------------------------

// PopcornGather gathers values from input at indices.
// out[i] = in[i * stride + idx[i]]
func PopcornGather(input *tendo.Tensor, indices *tendo.Tensor) (*tendo.Tensor, error) {
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}
	indicesStorage, ok := indices.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: indices.Device().Type}
	}

	// input shape: [n, classes], indices shape: [n]
	n := indices.Numel()
	stride := input.Shape()[len(input.Shape())-1]

	outStorage, err := NewStorage(n, input.DType(), inputStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornGather_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		(*C.int64_t)(unsafe.Pointer(indicesStorage.Ptr())),
		C.int64_t(n),
		C.int64_t(stride),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, indices.Shape(), nil), nil
}

// -----------------------------------------------------------------------------
// Reduction Operations
// -----------------------------------------------------------------------------

// PopcornArgMax returns indices of max values along the last dimension.
func PopcornArgMax(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	shape := t.Shape()
	if len(shape) < 1 {
		return nil, &tendo.ShapeError{Op: "argmax", Message: "tensor must have at least 1 dimension"}
	}

	stride := shape[len(shape)-1]
	n := t.Numel() / stride

	// Output is int64 indices
	outStorage, err := NewStorage(n, tendo.Int64, cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornArgMax_f32(
		(*C.int64_t)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(n),
		C.int64_t(stride),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Output shape is input shape with last dim removed
	outShape := make([]int, len(shape)-1)
	copy(outShape, shape[:len(shape)-1])
	if len(outShape) == 0 {
		outShape = []int{1}
	}

	return tendo.NewTensor(outStorage, outShape, nil), nil
}

// PopcornArgMin returns indices of min values along the last dimension.
func PopcornArgMin(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	shape := t.Shape()
	if len(shape) < 1 {
		return nil, &tendo.ShapeError{Op: "argmin", Message: "tensor must have at least 1 dimension"}
	}

	stride := shape[len(shape)-1]
	n := t.Numel() / stride

	// Output is int64 indices
	outStorage, err := NewStorage(n, tendo.Int64, cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornArgMin_f32(
		(*C.int64_t)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(n),
		C.int64_t(stride),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Output shape is input shape with last dim removed
	outShape := make([]int, len(shape)-1)
	copy(outShape, shape[:len(shape)-1])
	if len(outShape) == 0 {
		outShape = []int{1}
	}

	return tendo.NewTensor(outStorage, outShape, nil), nil
}

// -----------------------------------------------------------------------------
// Normalization Operations
// -----------------------------------------------------------------------------

// PopcornLayerNorm applies layer normalization.
func PopcornLayerNorm(input, weight, bias *tendo.Tensor, normalizedShape []int, eps float32) (*tendo.Tensor, error) {
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	// Calculate norm_size from normalizedShape
	normSize := 1
	for _, d := range normalizedShape {
		normSize *= d
	}

	n := input.Numel() / normSize

	outStorage, err := NewStorage(input.Numel(), input.DType(), inputStorage.device)
	if err != nil {
		return nil, err
	}

	// Get weight and bias pointers (nullable)
	var weightPtr, biasPtr *C.float
	if weight != nil {
		weightStorage, ok := weight.Storage().(*Storage)
		if !ok {
			outStorage.Free()
			return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: weight.Device().Type}
		}
		weightPtr = (*C.float)(unsafe.Pointer(weightStorage.Ptr()))
	}
	if bias != nil {
		biasStorage, ok := bias.Storage().(*Storage)
		if !ok {
			outStorage.Free()
			return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: bias.Device().Type}
		}
		biasPtr = (*C.float)(unsafe.Pointer(biasStorage.Ptr()))
	}

	status := C.popcornLayerNorm_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		weightPtr,
		biasPtr,
		C.int64_t(n),
		C.int64_t(normSize),
		C.float(eps),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, input.Shape(), nil), nil
}

// -----------------------------------------------------------------------------
// Additional Unary Operations
// -----------------------------------------------------------------------------

// PopcornNeg computes element-wise negation.
func PopcornNeg(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornNeg_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornSqrt computes element-wise square root.
func PopcornSqrt(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSqrt_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornSquare computes element-wise square.
func PopcornSquare(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSquare_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// -----------------------------------------------------------------------------
// Additional Binary Operations
// -----------------------------------------------------------------------------

// PopcornAdd computes element-wise addition.
func PopcornAdd(a, b *tendo.Tensor) (*tendo.Tensor, error) {
	aStorage, ok := a.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: a.Device().Type}
	}
	bStorage, ok := b.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: b.Device().Type}
	}

	if a.Numel() != b.Numel() {
		return nil, &tendo.ShapeError{Op: "add", Message: "tensors must have same number of elements"}
	}

	numel := a.Numel()
	outStorage, err := NewStorage(numel, a.DType(), aStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornAdd_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(aStorage.Ptr())),
		(*C.float)(unsafe.Pointer(bStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, a.Shape(), nil), nil
}

// PopcornSub computes element-wise subtraction.
func PopcornSub(a, b *tendo.Tensor) (*tendo.Tensor, error) {
	aStorage, ok := a.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: a.Device().Type}
	}
	bStorage, ok := b.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: b.Device().Type}
	}

	if a.Numel() != b.Numel() {
		return nil, &tendo.ShapeError{Op: "sub", Message: "tensors must have same number of elements"}
	}

	numel := a.Numel()
	outStorage, err := NewStorage(numel, a.DType(), aStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSub_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(aStorage.Ptr())),
		(*C.float)(unsafe.Pointer(bStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, a.Shape(), nil), nil
}

// PopcornMul computes element-wise multiplication.
func PopcornMul(a, b *tendo.Tensor) (*tendo.Tensor, error) {
	aStorage, ok := a.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: a.Device().Type}
	}
	bStorage, ok := b.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: b.Device().Type}
	}

	if a.Numel() != b.Numel() {
		return nil, &tendo.ShapeError{Op: "mul", Message: "tensors must have same number of elements"}
	}

	numel := a.Numel()
	outStorage, err := NewStorage(numel, a.DType(), aStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornMul_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(aStorage.Ptr())),
		(*C.float)(unsafe.Pointer(bStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, a.Shape(), nil), nil
}

// -----------------------------------------------------------------------------
// Scalar Operations
// -----------------------------------------------------------------------------

// PopcornAddScalar adds a scalar to each element.
func PopcornAddScalar(t *tendo.Tensor, scalar float32) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornAddScalar_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.float(scalar),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornSubScalar subtracts a scalar from each element.
func PopcornSubScalar(t *tendo.Tensor, scalar float32) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSubScalar_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.float(scalar),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornMulScalar multiplies each element by a scalar.
func PopcornMulScalar(t *tendo.Tensor, scalar float32) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornMulScalar_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.float(scalar),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornDivScalar divides each element by a scalar.
func PopcornDivScalar(t *tendo.Tensor, scalar float32) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornDivScalar_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.float(scalar),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// -----------------------------------------------------------------------------
// Tensor Operations
// -----------------------------------------------------------------------------

// PopcornEmbedding performs embedding lookup.
func PopcornEmbedding(weight *tendo.Tensor, indices *tendo.Tensor) (*tendo.Tensor, error) {
	weightStorage, ok := weight.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: weight.Device().Type}
	}
	indicesStorage, ok := indices.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: indices.Device().Type}
	}

	// weight shape: [vocab_size, embed_dim]
	// indices shape: arbitrary, flattened
	// output shape: indices.shape + [embed_dim]
	weightShape := weight.Shape()
	if len(weightShape) != 2 {
		return nil, &tendo.ShapeError{Op: "embedding", Message: "weight must be 2D [vocab_size, embed_dim]"}
	}
	vocabSize := weightShape[0]
	embedDim := weightShape[1]
	n := indices.Numel()

	// Output shape: indices shape + embed_dim
	outShape := append(indices.Shape(), embedDim)
	outNumel := tendo.Numel(outShape)

	outStorage, err := NewStorage(outNumel, weight.DType(), weightStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornEmbedding_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(weightStorage.Ptr())),
		(*C.int64_t)(unsafe.Pointer(indicesStorage.Ptr())),
		C.int64_t(n),
		C.int64_t(embedDim),
		C.int64_t(vocabSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, outShape, nil), nil
}

// PopcornCat concatenates tensors along a dimension.
func PopcornCat(tensors []*tendo.Tensor, dim int) (*tendo.Tensor, error) {
	if len(tensors) == 0 {
		return nil, &tendo.ShapeError{Op: "cat", Message: "cat requires at least one tensor"}
	}

	// Validate all tensors are CUDA
	shape := tensors[0].Shape()
	ndim := len(shape)
	if dim < 0 {
		dim = ndim + dim
	}
	if dim < 0 || dim >= ndim {
		return nil, &tendo.ShapeError{Op: "cat", Message: "dimension out of range"}
	}

	// Collect storage pointers and sizes
	inputPtrs := make([]unsafe.Pointer, len(tensors))
	sizes := make([]C.int64_t, len(tensors))
	var device int
	totalCatSize := 0

	for i, t := range tensors {
		storage, ok := t.Storage().(*Storage)
		if !ok {
			return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
		}
		if i == 0 {
			device = storage.device
		}

		tShape := t.Shape()
		if len(tShape) != ndim {
			return nil, &tendo.ShapeError{Op: "cat", Message: "all tensors must have same number of dimensions"}
		}
		for d := 0; d < ndim; d++ {
			if d != dim && tShape[d] != shape[d] {
				return nil, &tendo.ShapeError{Op: "cat", Message: "tensors must match in non-cat dimensions"}
			}
		}

		// Make contiguous if needed
		src := t
		if !t.IsContiguous() {
			src = t.Contiguous()
		}
		srcStorage := src.Storage().(*Storage)

		inputPtrs[i] = unsafe.Pointer(srcStorage.Ptr())
		sizes[i] = C.int64_t(tShape[dim])
		totalCatSize += tShape[dim]
	}

	// Compute outer and inner sizes
	outerSize := 1
	for d := 0; d < dim; d++ {
		outerSize *= shape[d]
	}
	innerSize := 1
	for d := dim + 1; d < ndim; d++ {
		innerSize *= shape[d]
	}

	// Output shape
	outShape := make([]int, ndim)
	copy(outShape, shape)
	outShape[dim] = totalCatSize
	outNumel := tendo.Numel(outShape)

	outStorage, err := NewStorage(outNumel, tensors[0].DType(), device)
	if err != nil {
		return nil, err
	}

	// Create C array of pointers
	cInputPtrs := make([]*C.float, len(tensors))
	for i, ptr := range inputPtrs {
		cInputPtrs[i] = (*C.float)(ptr)
	}

	status := C.popcornCat_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(**C.float)(unsafe.Pointer(&cInputPtrs[0])),
		C.int64_t(len(tensors)),
		(*C.int64_t)(unsafe.Pointer(&sizes[0])),
		C.int64_t(outerSize),
		C.int64_t(innerSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, outShape, nil), nil
}

// PopcornStack stacks tensors along a new dimension.
func PopcornStack(tensors []*tendo.Tensor, dim int) (*tendo.Tensor, error) {
	if len(tensors) == 0 {
		return nil, &tendo.ShapeError{Op: "stack", Message: "stack requires at least one tensor"}
	}

	// Validate all tensors have identical shapes and are CUDA
	shape := tensors[0].Shape()
	ndim := len(shape)
	if dim < 0 {
		dim = ndim + dim + 1
	}
	if dim < 0 || dim > ndim {
		return nil, &tendo.ShapeError{Op: "stack", Message: "dimension out of range"}
	}

	inputPtrs := make([]unsafe.Pointer, len(tensors))
	var device int
	tensorSize := tensors[0].Numel()

	for i, t := range tensors {
		storage, ok := t.Storage().(*Storage)
		if !ok {
			return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
		}
		if i == 0 {
			device = storage.device
		}

		tShape := t.Shape()
		if len(tShape) != ndim {
			return nil, &tendo.ShapeError{Op: "stack", Message: "all tensors must have same number of dimensions"}
		}
		for d := 0; d < ndim; d++ {
			if tShape[d] != shape[d] {
				return nil, &tendo.ShapeError{Op: "stack", Message: "all tensors must have identical shapes"}
			}
		}

		// Make contiguous if needed
		src := t
		if !t.IsContiguous() {
			src = t.Contiguous()
		}
		srcStorage := src.Storage().(*Storage)
		inputPtrs[i] = unsafe.Pointer(srcStorage.Ptr())
	}

	// Output shape: insert new dimension at 'dim'
	outShape := make([]int, ndim+1)
	for d := 0; d < dim; d++ {
		outShape[d] = shape[d]
	}
	outShape[dim] = len(tensors)
	for d := dim; d < ndim; d++ {
		outShape[d+1] = shape[d]
	}
	outNumel := tendo.Numel(outShape)

	outStorage, err := NewStorage(outNumel, tensors[0].DType(), device)
	if err != nil {
		return nil, err
	}

	// Create C array of pointers
	cInputPtrs := make([]*C.float, len(tensors))
	for i, ptr := range inputPtrs {
		cInputPtrs[i] = (*C.float)(ptr)
	}

	status := C.popcornStack_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(**C.float)(unsafe.Pointer(&cInputPtrs[0])),
		C.int64_t(len(tensors)),
		C.int64_t(tensorSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, outShape, nil), nil
}

// -----------------------------------------------------------------------------
// Trigonometric Operations
// -----------------------------------------------------------------------------

// PopcornSin computes element-wise sine.
func PopcornSin(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSin_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornCos computes element-wise cosine.
func PopcornCos(t *tendo.Tensor) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	outStorage, err := NewStorage(numel, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornCos_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// PopcornTril returns lower triangular part of a 2D tensor.
func PopcornTril(t *tendo.Tensor, k int) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	shape := t.Shape()
	if len(shape) != 2 {
		return nil, &tendo.ShapeError{Op: "tril", Message: "input must be 2D"}
	}
	rows, cols := shape[0], shape[1]

	outStorage, err := NewStorage(t.Numel(), t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornTril_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(cudaStorage.Ptr())),
		C.int64_t(rows),
		C.int64_t(cols),
		C.int64_t(k),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, shape, nil), nil
}

// -----------------------------------------------------------------------------
// Backward Pass Operations
// -----------------------------------------------------------------------------

// PopcornGeluBackward computes GELU backward pass.
func PopcornGeluBackward(gradOut, input *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	numel := gradOut.Numel()
	outStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornGeluBackward_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, gradOut.Shape(), nil), nil
}

// PopcornLeakyReluBackward computes LeakyReLU backward pass.
func PopcornLeakyReluBackward(gradOut, input *tendo.Tensor, alpha float32) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	numel := gradOut.Numel()
	outStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornLeakyReluBackward_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		C.float(alpha),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, gradOut.Shape(), nil), nil
}

// PopcornLayerNormBackward computes LayerNorm backward pass.
// Returns (gradInput, gradWeight, gradBias). gradWeight and gradBias may be nil if weight was nil.
func PopcornLayerNormBackward(gradOut, input, mean, invstd, weight *tendo.Tensor, normSize int) (*tendo.Tensor, *tendo.Tensor, *tendo.Tensor, error) {
	gradOutStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, nil, nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, nil, nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}
	meanStorage, ok := mean.Storage().(*Storage)
	if !ok {
		return nil, nil, nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: mean.Device().Type}
	}
	invstdStorage, ok := invstd.Storage().(*Storage)
	if !ok {
		return nil, nil, nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: invstd.Device().Type}
	}

	n := input.Numel() / normSize

	// Allocate output gradient
	gradInStorage, err := NewStorage(input.Numel(), input.DType(), inputStorage.device)
	if err != nil {
		return nil, nil, nil, err
	}

	// Weight and bias gradients (if weight provided)
	var weightPtr *C.float
	var gradWeightStorage, gradBiasStorage *Storage
	var gradWeightPtr, gradBiasPtr *C.float

	if weight != nil {
		weightStorage, ok := weight.Storage().(*Storage)
		if !ok {
			gradInStorage.Free()
			return nil, nil, nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: weight.Device().Type}
		}
		weightPtr = (*C.float)(unsafe.Pointer(weightStorage.Ptr()))

		gradWeightStorage, err = NewStorage(normSize, weight.DType(), inputStorage.device)
		if err != nil {
			gradInStorage.Free()
			return nil, nil, nil, err
		}
		gradWeightPtr = (*C.float)(unsafe.Pointer(gradWeightStorage.Ptr()))

		gradBiasStorage, err = NewStorage(normSize, weight.DType(), inputStorage.device)
		if err != nil {
			gradInStorage.Free()
			gradWeightStorage.Free()
			return nil, nil, nil, err
		}
		gradBiasPtr = (*C.float)(unsafe.Pointer(gradBiasStorage.Ptr()))
	}

	status := C.popcornLayerNormBackward_f32(
		(*C.float)(unsafe.Pointer(gradInStorage.Ptr())),
		gradWeightPtr,
		gradBiasPtr,
		(*C.float)(unsafe.Pointer(gradOutStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		(*C.float)(unsafe.Pointer(meanStorage.Ptr())),
		(*C.float)(unsafe.Pointer(invstdStorage.Ptr())),
		weightPtr,
		C.int64_t(n),
		C.int64_t(normSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		gradInStorage.Free()
		if gradWeightStorage != nil {
			gradWeightStorage.Free()
		}
		if gradBiasStorage != nil {
			gradBiasStorage.Free()
		}
		return nil, nil, nil, err
	}

	gradIn := tendo.NewTensor(gradInStorage, input.Shape(), nil)
	var gradWeight, gradBias *tendo.Tensor
	if gradWeightStorage != nil {
		gradWeight = tendo.NewTensor(gradWeightStorage, []int{normSize}, nil)
		gradBias = tendo.NewTensor(gradBiasStorage, []int{normSize}, nil)
	}

	return gradIn, gradWeight, gradBias, nil
}

// PopcornEmbeddingBackward accumulates gradients into embedding table.
func PopcornEmbeddingBackward(gradOut, indices *tendo.Tensor, vocabSize, embedDim int) (*tendo.Tensor, error) {
	gradOutStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	indicesStorage, ok := indices.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: indices.Device().Type}
	}

	n := indices.Numel()

	// Allocate gradient for embedding table
	gradWeightStorage, err := NewStorage(vocabSize*embedDim, gradOut.DType(), gradOutStorage.device)
	if err != nil {
		return nil, err
	}
	// Zero the storage first
	if err := gradWeightStorage.Zero(); err != nil {
		gradWeightStorage.Free()
		return nil, err
	}

	status := C.popcornEmbeddingBackward_f32(
		(*C.float)(unsafe.Pointer(gradWeightStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradOutStorage.Ptr())),
		(*C.int64_t)(unsafe.Pointer(indicesStorage.Ptr())),
		C.int64_t(n),
		C.int64_t(embedDim),
		C.int64_t(vocabSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		gradWeightStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(gradWeightStorage, []int{vocabSize, embedDim}, nil), nil
}

// PopcornReluBackward computes ReLU backward pass.
func PopcornReluBackward(gradOut, input *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	numel := gradOut.Numel()
	outStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornReluBackward_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, gradOut.Shape(), nil), nil
}

// PopcornSigmoidBackward computes Sigmoid backward pass.
// Takes the sigmoid output (not input) as that's what the kernel expects.
func PopcornSigmoidBackward(gradOut, sigmoidOut *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	outStorage, ok := sigmoidOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: sigmoidOut.Device().Type}
	}

	numel := gradOut.Numel()
	gradInStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSigmoidBackward_f32(
		(*C.float)(unsafe.Pointer(gradInStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		gradInStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(gradInStorage, gradOut.Shape(), nil), nil
}

// PopcornTanhBackward computes Tanh backward pass.
// Takes the tanh output (not input) as that's what the kernel expects.
func PopcornTanhBackward(gradOut, tanhOut *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	outStorage, ok := tanhOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: tanhOut.Device().Type}
	}

	numel := gradOut.Numel()
	gradInStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornTanhBackward_f32(
		(*C.float)(unsafe.Pointer(gradInStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		gradInStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(gradInStorage, gradOut.Shape(), nil), nil
}

// PopcornSiluBackward computes SiLU backward pass.
func PopcornSiluBackward(gradOut, input *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	numel := gradOut.Numel()
	outStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSiluBackward_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, gradOut.Shape(), nil), nil
}

// PopcornExpBackward computes Exp backward pass.
// Takes the exp output (not input) as that's what the kernel expects.
func PopcornExpBackward(gradOut, expOut *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	outStorage, ok := expOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: expOut.Device().Type}
	}

	numel := gradOut.Numel()
	gradInStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornExpBackward_f32(
		(*C.float)(unsafe.Pointer(gradInStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		gradInStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(gradInStorage, gradOut.Shape(), nil), nil
}

// PopcornLogBackward computes Log backward pass.
func PopcornLogBackward(gradOut, input *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	numel := gradOut.Numel()
	outStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornLogBackward_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, gradOut.Shape(), nil), nil
}

// PopcornSqrtBackward computes Sqrt backward pass.
// Takes the sqrt output (not input) as that's what the kernel expects.
func PopcornSqrtBackward(gradOut, sqrtOut *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	outStorage, ok := sqrtOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: sqrtOut.Device().Type}
	}

	numel := gradOut.Numel()
	gradInStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSqrtBackward_f32(
		(*C.float)(unsafe.Pointer(gradInStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		gradInStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(gradInStorage, gradOut.Shape(), nil), nil
}

// PopcornSinBackward computes Sin backward pass.
func PopcornSinBackward(gradOut, input *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	numel := gradOut.Numel()
	outStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSinBackward_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, gradOut.Shape(), nil), nil
}

// PopcornCosBackward computes Cos backward pass.
func PopcornCosBackward(gradOut, input *tendo.Tensor) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	numel := gradOut.Numel()
	outStorage, err := NewStorage(numel, gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornCosBackward_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		C.int64_t(numel),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, gradOut.Shape(), nil), nil
}

// PopcornSoftmaxBackward computes Softmax backward pass.
func PopcornSoftmaxBackward(gradOut, softmaxOut *tendo.Tensor, dim int) (*tendo.Tensor, error) {
	gradStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	outStorage, ok := softmaxOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: softmaxOut.Device().Type}
	}

	shape := gradOut.Shape()
	ndim := len(shape)
	if dim < 0 {
		dim = ndim + dim
	}

	// Compute batch (product of dims before dim) and dim size
	batch := 1
	for d := 0; d < dim; d++ {
		batch *= shape[d]
	}
	// Also include dims after dim in batch
	for d := dim + 1; d < ndim; d++ {
		batch *= shape[d]
	}
	dimSize := shape[dim]

	gradInStorage, err := NewStorage(gradOut.Numel(), gradOut.DType(), gradStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornSoftmaxBackward_f32(
		(*C.float)(unsafe.Pointer(gradInStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		C.int64_t(batch),
		C.int64_t(dimSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		gradInStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(gradInStorage, shape, nil), nil
}

// PopcornCrossEntropyBackward computes fused CrossEntropy backward pass.
// Returns gradient w.r.t. logits: scale * (softmax - one_hot(target)).
func PopcornCrossEntropyBackward(softmaxOut, targets *tendo.Tensor, scale float32) (*tendo.Tensor, error) {
	softmaxStorage, ok := softmaxOut.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: softmaxOut.Device().Type}
	}
	targetsStorage, ok := targets.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: targets.Device().Type}
	}

	shape := softmaxOut.Shape()
	if len(shape) != 2 {
		return nil, &tendo.ShapeError{Op: "cross_entropy_backward", Message: "softmax must be 2D [batch, classes]"}
	}
	batch := shape[0]
	classes := shape[1]

	gradInStorage, err := NewStorage(softmaxOut.Numel(), softmaxOut.DType(), softmaxStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornCrossEntropyBackward_f32(
		(*C.float)(unsafe.Pointer(gradInStorage.Ptr())),
		(*C.float)(unsafe.Pointer(softmaxStorage.Ptr())),
		(*C.int64_t)(unsafe.Pointer(targetsStorage.Ptr())),
		C.int64_t(batch),
		C.int64_t(classes),
		C.float(scale),
		nil,
	)
	if err := popcornError(status); err != nil {
		gradInStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(gradInStorage, shape, nil), nil
}

// PopcornRMSNormBackward computes RMSNorm backward pass.
// Returns (gradInput, gradWeight). gradWeight may be nil if weight was nil.
func PopcornRMSNormBackward(gradOut, input, rrms, weight *tendo.Tensor, normSize int) (*tendo.Tensor, *tendo.Tensor, error) {
	gradOutStorage, ok := gradOut.Storage().(*Storage)
	if !ok {
		return nil, nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: gradOut.Device().Type}
	}
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}
	rrmsStorage, ok := rrms.Storage().(*Storage)
	if !ok {
		return nil, nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: rrms.Device().Type}
	}

	n := input.Numel() / normSize

	// Allocate output gradient
	gradInStorage, err := NewStorage(input.Numel(), input.DType(), inputStorage.device)
	if err != nil {
		return nil, nil, err
	}

	// Weight gradient (if weight provided)
	var weightPtr *C.float
	var gradWeightStorage *Storage
	var gradWeightPtr *C.float

	if weight != nil {
		weightStorage, ok := weight.Storage().(*Storage)
		if !ok {
			gradInStorage.Free()
			return nil, nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: weight.Device().Type}
		}
		weightPtr = (*C.float)(unsafe.Pointer(weightStorage.Ptr()))

		gradWeightStorage, err = NewStorage(normSize, weight.DType(), inputStorage.device)
		if err != nil {
			gradInStorage.Free()
			return nil, nil, err
		}
		gradWeightPtr = (*C.float)(unsafe.Pointer(gradWeightStorage.Ptr()))
	}

	status := C.popcornRMSNormBackward_f32(
		(*C.float)(unsafe.Pointer(gradInStorage.Ptr())),
		gradWeightPtr,
		(*C.float)(unsafe.Pointer(gradOutStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		(*C.float)(unsafe.Pointer(rrmsStorage.Ptr())),
		weightPtr,
		C.int64_t(n),
		C.int64_t(normSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		gradInStorage.Free()
		if gradWeightStorage != nil {
			gradWeightStorage.Free()
		}
		return nil, nil, err
	}

	gradIn := tendo.NewTensor(gradInStorage, input.Shape(), nil)
	var gradWeight *tendo.Tensor
	if gradWeightStorage != nil {
		gradWeight = tendo.NewTensor(gradWeightStorage, []int{normSize}, nil)
	}

	return gradIn, gradWeight, nil
}

// PopcornScatter writes values to indexed positions.
func PopcornScatter(out, in, indices *tendo.Tensor, stride int) error {
	outStorage, ok := out.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: out.Device().Type}
	}
	inStorage, ok := in.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: in.Device().Type}
	}
	indicesStorage, ok := indices.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: indices.Device().Type}
	}

	n := in.Numel()

	status := C.popcornScatter_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inStorage.Ptr())),
		(*C.int64_t)(unsafe.Pointer(indicesStorage.Ptr())),
		C.int64_t(n),
		C.int64_t(stride),
		nil,
	)
	return popcornError(status)
}

// PopcornScatterAdd accumulates values at indexed positions.
func PopcornScatterAdd(out, in, indices *tendo.Tensor, stride int) error {
	outStorage, ok := out.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: out.Device().Type}
	}
	inStorage, ok := in.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: in.Device().Type}
	}
	indicesStorage, ok := indices.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: indices.Device().Type}
	}

	n := in.Numel()

	status := C.popcornScatterAdd_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inStorage.Ptr())),
		(*C.int64_t)(unsafe.Pointer(indicesStorage.Ptr())),
		C.int64_t(n),
		C.int64_t(stride),
		nil,
	)
	return popcornError(status)
}

// PopcornSplit splits a tensor into multiple outputs (inverse of Cat).
func PopcornSplit(input *tendo.Tensor, sizes []int, dim int) ([]*tendo.Tensor, error) {
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	shape := input.Shape()
	ndim := len(shape)
	if dim < 0 {
		dim = ndim + dim
	}

	// Compute outer and inner sizes
	outerSize := 1
	for d := 0; d < dim; d++ {
		outerSize *= shape[d]
	}
	innerSize := 1
	for d := dim + 1; d < ndim; d++ {
		innerSize *= shape[d]
	}

	// Allocate output tensors
	outputs := make([]*tendo.Tensor, len(sizes))
	outputPtrs := make([]unsafe.Pointer, len(sizes))
	cSizes := make([]C.int64_t, len(sizes))

	for i, size := range sizes {
		outShape := make([]int, ndim)
		copy(outShape, shape)
		outShape[dim] = size

		outStorage, err := NewStorage(tendo.Numel(outShape), input.DType(), inputStorage.device)
		if err != nil {
			// Free already allocated
			for j := 0; j < i; j++ {
				outputs[j].Free()
			}
			return nil, err
		}
		outputs[i] = tendo.NewTensor(outStorage, outShape, nil)
		outputPtrs[i] = unsafe.Pointer(outStorage.Ptr())
		cSizes[i] = C.int64_t(size)
	}

	// Create C array of pointers
	cOutputPtrs := make([]*C.float, len(outputs))
	for i, ptr := range outputPtrs {
		cOutputPtrs[i] = (*C.float)(ptr)
	}

	status := C.popcornSplit_f32(
		(**C.float)(unsafe.Pointer(&cOutputPtrs[0])),
		C.int64_t(len(outputs)),
		(*C.int64_t)(unsafe.Pointer(&cSizes[0])),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		C.int64_t(outerSize),
		C.int64_t(innerSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		for _, t := range outputs {
			t.Free()
		}
		return nil, err
	}

	return outputs, nil
}

// PopcornUnstack splits a tensor along first dimension (inverse of Stack).
func PopcornUnstack(input *tendo.Tensor) ([]*tendo.Tensor, error) {
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	shape := input.Shape()
	if len(shape) < 1 {
		return nil, &tendo.ShapeError{Op: "unstack", Message: "input must have at least 1 dimension"}
	}

	numOutputs := shape[0]
	tensorSize := input.Numel() / numOutputs
	outShape := shape[1:]
	if len(outShape) == 0 {
		outShape = []int{1}
	}

	// Allocate output tensors
	outputs := make([]*tendo.Tensor, numOutputs)
	outputPtrs := make([]unsafe.Pointer, numOutputs)

	for i := 0; i < numOutputs; i++ {
		outStorage, err := NewStorage(tensorSize, input.DType(), inputStorage.device)
		if err != nil {
			for j := 0; j < i; j++ {
				outputs[j].Free()
			}
			return nil, err
		}
		outputs[i] = tendo.NewTensor(outStorage, outShape, nil)
		outputPtrs[i] = unsafe.Pointer(outStorage.Ptr())
	}

	// Create C array of pointers
	cOutputPtrs := make([]*C.float, numOutputs)
	for i, ptr := range outputPtrs {
		cOutputPtrs[i] = (*C.float)(ptr)
	}

	status := C.popcornUnstack_f32(
		(**C.float)(unsafe.Pointer(&cOutputPtrs[0])),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		C.int64_t(numOutputs),
		C.int64_t(tensorSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		for _, t := range outputs {
			t.Free()
		}
		return nil, err
	}

	return outputs, nil
}

// -----------------------------------------------------------------------------
// Optimizer Operations
// -----------------------------------------------------------------------------

// PopcornAdamW performs a fused AdamW optimizer step.
// Updates param, m, v tensors in-place.
// biasCorrection1 = 1 - beta1^t, biasCorrection2 = 1 - beta2^t (precomputed by caller).
func PopcornAdamW(
	param, grad, m, v *tendo.Tensor,
	lr, beta1, beta2, epsilon, weightDecay, biasCorrection1, biasCorrection2 float32,
) error {
	paramStorage, ok := param.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: param.Device().Type}
	}
	gradStorage, ok := grad.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: grad.Device().Type}
	}
	mStorage, ok := m.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: m.Device().Type}
	}
	vStorage, ok := v.Storage().(*Storage)
	if !ok {
		return &tendo.DeviceError{Expected: tendo.CUDA, Got: v.Device().Type}
	}

	numel := param.Numel()

	status := C.popcornAdamW_f32(
		(*C.float)(unsafe.Pointer(paramStorage.Ptr())),
		(*C.float)(unsafe.Pointer(gradStorage.Ptr())),
		(*C.float)(unsafe.Pointer(mStorage.Ptr())),
		(*C.float)(unsafe.Pointer(vStorage.Ptr())),
		C.float(lr),
		C.float(beta1),
		C.float(beta2),
		C.float(epsilon),
		C.float(weightDecay),
		C.float(biasCorrection1),
		C.float(biasCorrection2),
		C.int64_t(numel),
		nil,
	)
	return popcornError(status)
}
