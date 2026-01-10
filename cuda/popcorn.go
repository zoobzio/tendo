//go:build cuda

package cuda

/*
#cgo CFLAGS: -I/usr/include
#cgo LDFLAGS: -L/usr/lib -lpopcorn -lcudart -lstdc++

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
	msg := C.GoString(C.popcornGetErrorString(status))
	if status == C.POPCORN_ERROR_CUDA {
		// Get more detailed CUDA error string
		cudaMsg := C.GoString(C.popcornGetLastCudaErrorString())
		if cudaMsg != "" {
			msg = msg + ": " + cudaMsg
		}
	}
	return &CUDAError{
		Code:    int(status),
		Message: msg,
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

// PopcornRMSNorm applies RMS normalization.
func PopcornRMSNorm(input, weight *tendo.Tensor, normalizedShape []int, eps float32) (*tendo.Tensor, error) {
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

	// Get weight pointer (nullable)
	var weightPtr *C.float
	if weight != nil {
		weightStorage, ok := weight.Storage().(*Storage)
		if !ok {
			outStorage.Free()
			return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: weight.Device().Type}
		}
		weightPtr = (*C.float)(unsafe.Pointer(weightStorage.Ptr()))
	}

	status := C.popcornRMSNorm_f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(inputStorage.Ptr())),
		weightPtr,
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
// Quantized MatMul Operations
// -----------------------------------------------------------------------------

// PopcornDequantizeMatmul performs fused dequantize + matmul with per-channel scales.
// Computes: out = x @ dequantize(qweight).T
// x: [M, K] activations, qweight: [N, K] int8, scale: [N], out: [M, N]
func PopcornDequantizeMatmul(x, qweight, scale *tendo.Tensor) (*tendo.Tensor, error) {
	xStorage, ok := x.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: x.Device().Type}
	}
	qweightStorage, ok := qweight.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: qweight.Device().Type}
	}
	scaleStorage, ok := scale.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: scale.Device().Type}
	}

	xShape := x.Shape()
	qShape := qweight.Shape()

	// x: [M, K], qweight: [N, K]
	M := xShape[0]
	K := xShape[1]
	N := qShape[0]

	outStorage, err := NewStorage(M*N, x.DType(), xStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornDequantizeMatmul_i8f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(xStorage.Ptr())),
		(*C.int8_t)(unsafe.Pointer(qweightStorage.Ptr())),
		(*C.float)(unsafe.Pointer(scaleStorage.Ptr())),
		C.int64_t(M),
		C.int64_t(N),
		C.int64_t(K),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, []int{M, N}, nil), nil
}

// PopcornDequantizeMatmulGrouped performs fused dequantize + matmul with per-group scales.
// Computes: out = x @ dequantize(qweight).T
// x: [M, K], qweight: [N, K] int8, scale: [N, num_groups], out: [M, N]
func PopcornDequantizeMatmulGrouped(x, qweight, scale *tendo.Tensor, groupSize int) (*tendo.Tensor, error) {
	xStorage, ok := x.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: x.Device().Type}
	}
	qweightStorage, ok := qweight.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: qweight.Device().Type}
	}
	scaleStorage, ok := scale.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: scale.Device().Type}
	}

	xShape := x.Shape()
	qShape := qweight.Shape()

	M := xShape[0]
	K := xShape[1]
	N := qShape[0]

	outStorage, err := NewStorage(M*N, x.DType(), xStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornDequantizeMatmulGrouped_i8f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(xStorage.Ptr())),
		(*C.int8_t)(unsafe.Pointer(qweightStorage.Ptr())),
		(*C.float)(unsafe.Pointer(scaleStorage.Ptr())),
		C.int64_t(M),
		C.int64_t(N),
		C.int64_t(K),
		C.int64_t(groupSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, []int{M, N}, nil), nil
}

// PopcornDequantizeMatmulBatched performs batched fused dequantize + matmul.
// Computes: out[b] = x[b] @ dequantize(qweight).T for each batch
// x: [B, M, K], qweight: [N, K] (shared), scale: [N], out: [B, M, N]
func PopcornDequantizeMatmulBatched(x, qweight, scale *tendo.Tensor) (*tendo.Tensor, error) {
	xStorage, ok := x.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: x.Device().Type}
	}
	qweightStorage, ok := qweight.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: qweight.Device().Type}
	}
	scaleStorage, ok := scale.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: scale.Device().Type}
	}

	xShape := x.Shape()
	qShape := qweight.Shape()

	B := xShape[0]
	M := xShape[1]
	K := xShape[2]
	N := qShape[0]

	outStorage, err := NewStorage(B*M*N, x.DType(), xStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornDequantizeMatmulBatched_i8f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(xStorage.Ptr())),
		(*C.int8_t)(unsafe.Pointer(qweightStorage.Ptr())),
		(*C.float)(unsafe.Pointer(scaleStorage.Ptr())),
		C.int64_t(B),
		C.int64_t(M),
		C.int64_t(N),
		C.int64_t(K),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, []int{B, M, N}, nil), nil
}

// PopcornDequantizeMatmulInt4Grouped performs INT4 dequantize + matmul with asymmetric quantization.
// Computes: out = x @ dequantize(qweight).T
// qweight is packed: 2 int4 values per byte
// x: [M, K], qweight: [N, K/2] packed, scale: [N, groups], zero: [N, groups], out: [M, N]
func PopcornDequantizeMatmulInt4Grouped(x, qweight, scale, zero *tendo.Tensor, groupSize int) (*tendo.Tensor, error) {
	xStorage, ok := x.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: x.Device().Type}
	}
	qweightStorage, ok := qweight.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: qweight.Device().Type}
	}
	scaleStorage, ok := scale.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: scale.Device().Type}
	}
	zeroStorage, ok := zero.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: zero.Device().Type}
	}

	xShape := x.Shape()
	qShape := qweight.Shape()

	M := xShape[0]
	K := xShape[1]        // original K (not packed)
	N := qShape[0]

	outStorage, err := NewStorage(M*N, x.DType(), xStorage.device)
	if err != nil {
		return nil, err
	}

	status := C.popcornDequantizeMatmulGrouped_i4f32(
		(*C.float)(unsafe.Pointer(outStorage.Ptr())),
		(*C.float)(unsafe.Pointer(xStorage.Ptr())),
		(*C.uint8_t)(unsafe.Pointer(qweightStorage.Ptr())),
		(*C.float)(unsafe.Pointer(scaleStorage.Ptr())),
		(*C.float)(unsafe.Pointer(zeroStorage.Ptr())),
		C.int64_t(M),
		C.int64_t(N),
		C.int64_t(K),
		C.int64_t(groupSize),
		nil,
	)
	if err := popcornError(status); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, []int{M, N}, nil), nil
}
