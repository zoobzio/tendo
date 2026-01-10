//go:build cuda

package cuda

/*
#cgo CFLAGS: -I/opt/cuda/targets/x86_64-linux/include
#cgo LDFLAGS: -L/opt/cuda/targets/x86_64-linux/lib -lcudart -lcublas

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>

const char* getCudaErrorString(cudaError_t err) {
    return cudaGetErrorString(err);
}
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// cudaError converts a CUDA error code to a Go error.
func cudaError(err C.cudaError_t) error {
	if err == C.cudaSuccess {
		return nil
	}
	return &CUDAError{
		Code:    int(err),
		Message: C.GoString(C.getCudaErrorString(err)),
	}
}

// cublasError converts a cuBLAS status to a Go error.
func cublasError(status C.cublasStatus_t) error {
	if status == C.CUBLAS_STATUS_SUCCESS {
		return nil
	}
	messages := map[C.cublasStatus_t]string{
		C.CUBLAS_STATUS_NOT_INITIALIZED:  "cuBLAS not initialized",
		C.CUBLAS_STATUS_ALLOC_FAILED:     "cuBLAS allocation failed",
		C.CUBLAS_STATUS_INVALID_VALUE:    "cuBLAS invalid value",
		C.CUBLAS_STATUS_ARCH_MISMATCH:    "cuBLAS architecture mismatch",
		C.CUBLAS_STATUS_MAPPING_ERROR:    "cuBLAS mapping error",
		C.CUBLAS_STATUS_EXECUTION_FAILED: "cuBLAS execution failed",
		C.CUBLAS_STATUS_INTERNAL_ERROR:   "cuBLAS internal error",
	}
	msg, ok := messages[status]
	if !ok {
		msg = fmt.Sprintf("cuBLAS error %d", int(status))
	}
	return &CUDAError{Code: int(status), Message: msg}
}

// cudaSetDevice sets the current CUDA device.
func cudaSetDevice(device int) error {
	return cudaError(C.cudaSetDevice(C.int(device)))
}

// cudaGetDevice returns the current CUDA device.
func cudaGetDevice() (int, error) {
	var device C.int
	err := C.cudaGetDevice(&device)
	return int(device), cudaError(err)
}

// cudaMalloc allocates memory on the current CUDA device.
func cudaMalloc(size int) (uintptr, error) {
	var ptr unsafe.Pointer
	err := C.cudaMalloc(&ptr, C.size_t(size))
	if err != C.cudaSuccess {
		return 0, cudaError(err)
	}
	return uintptr(ptr), nil
}

// cudaFree frees memory on the current CUDA device.
func cudaFree(ptr uintptr) error {
	return cudaError(C.cudaFree(unsafe.Pointer(ptr)))
}

// cudaMemcpyKind specifies the direction of a memory copy.
type cudaMemcpyKind int

const (
	cudaMemcpyHostToHost     cudaMemcpyKind = C.cudaMemcpyHostToHost
	cudaMemcpyHostToDevice   cudaMemcpyKind = C.cudaMemcpyHostToDevice
	cudaMemcpyDeviceToHost   cudaMemcpyKind = C.cudaMemcpyDeviceToHost
	cudaMemcpyDeviceToDevice cudaMemcpyKind = C.cudaMemcpyDeviceToDevice
)

// cudaMemcpy copies data between host and device.
func cudaMemcpy(dst, src unsafe.Pointer, size int, kind cudaMemcpyKind) error {
	return cudaError(C.cudaMemcpy(
		dst,
		src,
		C.size_t(size),
		uint32(kind),
	))
}

// cudaMemcpyPtr copies data between host and device using uintptr (for device pointers).
// IMPORTANT: Only use this for device-to-device copies where both pointers are device memory.
func cudaMemcpyPtr(dst, src uintptr, size int, kind cudaMemcpyKind) error {
	return cudaError(C.cudaMemcpy(
		unsafe.Pointer(dst),
		unsafe.Pointer(src),
		C.size_t(size),
		uint32(kind),
	))
}

// cudaMemset sets device memory to a value.
func cudaMemset(ptr uintptr, value int, size int) error {
	return cudaError(C.cudaMemset(unsafe.Pointer(ptr), C.int(value), C.size_t(size)))
}

// cudaDeviceSynchronize blocks until all device operations complete.
func cudaDeviceSynchronize() error {
	return cudaError(C.cudaDeviceSynchronize())
}

// cudaGetDeviceCount returns the number of CUDA devices.
func cudaGetDeviceCount() (int, error) {
	var count C.int
	err := C.cudaGetDeviceCount(&count)
	return int(count), cudaError(err)
}

// cublasHandle represents a cuBLAS context.
type cublasHandle uintptr

// cublasCreate creates a cuBLAS handle.
func cublasCreate() (cublasHandle, error) {
	var handle C.cublasHandle_t
	status := C.cublasCreate(&handle)
	if status != C.CUBLAS_STATUS_SUCCESS {
		return 0, cublasError(status)
	}
	return cublasHandle(uintptr(unsafe.Pointer(handle))), nil
}

// cublasDestroy destroys a cuBLAS handle.
func cublasDestroy(handle cublasHandle) error {
	return cublasError(C.cublasDestroy(C.cublasHandle_t(unsafe.Pointer(handle))))
}

// cublasOperation specifies the operation on a matrix.
type cublasOperation int

const (
	cublasOpN cublasOperation = C.CUBLAS_OP_N // no transpose
	cublasOpT cublasOperation = C.CUBLAS_OP_T // transpose
	cublasOpC cublasOperation = C.CUBLAS_OP_C // conjugate transpose
)

// cublasSgemm performs single-precision matrix multiplication: C = alpha*op(A)*op(B) + beta*C
func cublasSgemm(handle cublasHandle,
	transA, transB cublasOperation,
	m, n, k int,
	alpha float32,
	A uintptr, lda int,
	B uintptr, ldb int,
	beta float32,
	cPtr uintptr, ldc int) error {

	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cublasSgemm(
		C.cublasHandle_t(unsafe.Pointer(handle)),
		C.cublasOperation_t(transA),
		C.cublasOperation_t(transB),
		C.int(m), C.int(n), C.int(k),
		&cAlpha,
		(*C.float)(unsafe.Pointer(A)), C.int(lda),
		(*C.float)(unsafe.Pointer(B)), C.int(ldb),
		&cBeta,
		(*C.float)(unsafe.Pointer(cPtr)), C.int(ldc),
	)
	return cublasError(status)
}

// cublasSgemmStridedBatched performs batched single-precision matrix multiplication.
// C[i] = alpha*op(A[i])*op(B[i]) + beta*C[i] for i in [0, batchCount)
func cublasSgemmStridedBatched(handle cublasHandle,
	transA, transB cublasOperation,
	m, n, k int,
	alpha float32,
	A uintptr, lda int, strideA int64,
	B uintptr, ldb int, strideB int64,
	beta float32,
	cPtr uintptr, ldc int, strideC int64,
	batchCount int) error {

	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cublasSgemmStridedBatched(
		C.cublasHandle_t(unsafe.Pointer(handle)),
		C.cublasOperation_t(transA),
		C.cublasOperation_t(transB),
		C.int(m), C.int(n), C.int(k),
		&cAlpha,
		(*C.float)(unsafe.Pointer(A)), C.int(lda), C.longlong(strideA),
		(*C.float)(unsafe.Pointer(B)), C.int(ldb), C.longlong(strideB),
		&cBeta,
		(*C.float)(unsafe.Pointer(cPtr)), C.int(ldc), C.longlong(strideC),
		C.int(batchCount),
	)
	return cublasError(status)
}

// Global cuBLAS handle (lazily initialized)
var (
	globalCublasHandle    cublasHandle
	globalCublasHandleErr error
	globalCublasOnce      sync.Once
)

// getCublasHandle returns the global cuBLAS handle, initializing if needed.
// Thread-safe via sync.Once.
func getCublasHandle() (cublasHandle, error) {
	globalCublasOnce.Do(func() {
		globalCublasHandle, globalCublasHandleErr = cublasCreate()
	})
	return globalCublasHandle, globalCublasHandleErr
}
