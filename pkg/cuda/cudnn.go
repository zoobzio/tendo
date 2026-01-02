package cuda

/*
#cgo CFLAGS: -I/opt/cuda/targets/x86_64-linux/include
#cgo LDFLAGS: -L/opt/cuda/targets/x86_64-linux/lib -lcudnn

#include <cudnn.h>
#include <stdlib.h>

const char* getCudnnErrorString(cudnnStatus_t status) {
    return cudnnGetErrorString(status);
}
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// cudnnHandle represents a cuDNN context.
type cudnnHandle uintptr

// cudnnTensorDescriptor describes a tensor's layout.
type cudnnTensorDescriptor uintptr

// cudnnActivationDescriptor describes an activation operation.
type cudnnActivationDescriptor uintptr

// cudnnActivationMode specifies the activation function type.
type cudnnActivationMode int

const (
	CudnnActivationSigmoid     cudnnActivationMode = C.CUDNN_ACTIVATION_SIGMOID
	CudnnActivationRelu        cudnnActivationMode = C.CUDNN_ACTIVATION_RELU
	CudnnActivationTanh        cudnnActivationMode = C.CUDNN_ACTIVATION_TANH
	CudnnActivationClippedRelu cudnnActivationMode = C.CUDNN_ACTIVATION_CLIPPED_RELU
	CudnnActivationElu         cudnnActivationMode = C.CUDNN_ACTIVATION_ELU
	CudnnActivationSwish       cudnnActivationMode = C.CUDNN_ACTIVATION_SWISH
)

// cudnnSoftmaxAlgorithm specifies the softmax algorithm.
type cudnnSoftmaxAlgorithm int

const (
	cudnnSoftmaxFast     cudnnSoftmaxAlgorithm = C.CUDNN_SOFTMAX_FAST
	cudnnSoftmaxAccurate cudnnSoftmaxAlgorithm = C.CUDNN_SOFTMAX_ACCURATE
	cudnnSoftmaxLog      cudnnSoftmaxAlgorithm = C.CUDNN_SOFTMAX_LOG
)

// cudnnSoftmaxMode specifies where softmax is computed.
type cudnnSoftmaxMode int

const (
	cudnnSoftmaxModeInstance cudnnSoftmaxMode = C.CUDNN_SOFTMAX_MODE_INSTANCE
	cudnnSoftmaxModeChannel  cudnnSoftmaxMode = C.CUDNN_SOFTMAX_MODE_CHANNEL
)

// cudnnDataType specifies the data type.
type cudnnDataType int

const (
	cudnnDataFloat cudnnDataType = C.CUDNN_DATA_FLOAT
)

// cudnnNanPropagation specifies NaN handling.
type cudnnNanPropagation int

const (
	cudnnNotPropagateNan cudnnNanPropagation = C.CUDNN_NOT_PROPAGATE_NAN
	cudnnPropagateNan    cudnnNanPropagation = C.CUDNN_PROPAGATE_NAN
)

// cudnnError converts a cuDNN status to a Go error.
func cudnnError(status C.cudnnStatus_t) error {
	if status == C.CUDNN_STATUS_SUCCESS {
		return nil
	}
	return &CUDAError{
		Code:    int(status),
		Message: fmt.Sprintf("cuDNN error: %s", C.GoString(C.getCudnnErrorString(status))),
	}
}

// cudnnCreate creates a cuDNN handle.
func cudnnCreate() (cudnnHandle, error) {
	var handle C.cudnnHandle_t
	status := C.cudnnCreate(&handle)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return cudnnHandle(uintptr(unsafe.Pointer(handle))), nil
}

// cudnnDestroy destroys a cuDNN handle.
func cudnnDestroy(handle cudnnHandle) error {
	return cudnnError(C.cudnnDestroy(C.cudnnHandle_t(unsafe.Pointer(handle))))
}

// cudnnCreateTensorDescriptor creates a tensor descriptor.
func cudnnCreateTensorDescriptor() (cudnnTensorDescriptor, error) {
	var desc C.cudnnTensorDescriptor_t
	status := C.cudnnCreateTensorDescriptor(&desc)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return cudnnTensorDescriptor(uintptr(unsafe.Pointer(desc))), nil
}

// cudnnDestroyTensorDescriptor destroys a tensor descriptor.
func cudnnDestroyTensorDescriptor(desc cudnnTensorDescriptor) error {
	return cudnnError(C.cudnnDestroyTensorDescriptor(C.cudnnTensorDescriptor_t(unsafe.Pointer(desc))))
}

// cudnnSetTensor4dDescriptor sets a 4D tensor descriptor.
func cudnnSetTensor4dDescriptor(desc cudnnTensorDescriptor, n, c, h, w int) error {
	status := C.cudnnSetTensor4dDescriptor(
		C.cudnnTensorDescriptor_t(unsafe.Pointer(desc)),
		C.CUDNN_TENSOR_NCHW,
		C.CUDNN_DATA_FLOAT,
		C.int(n), C.int(c), C.int(h), C.int(w),
	)
	return cudnnError(status)
}

// cudnnCreateActivationDescriptor creates an activation descriptor.
func cudnnCreateActivationDescriptor() (cudnnActivationDescriptor, error) {
	var desc C.cudnnActivationDescriptor_t
	status := C.cudnnCreateActivationDescriptor(&desc)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return cudnnActivationDescriptor(uintptr(unsafe.Pointer(desc))), nil
}

// cudnnDestroyActivationDescriptor destroys an activation descriptor.
func cudnnDestroyActivationDescriptor(desc cudnnActivationDescriptor) error {
	return cudnnError(C.cudnnDestroyActivationDescriptor(C.cudnnActivationDescriptor_t(unsafe.Pointer(desc))))
}

// cudnnSetActivationDescriptor configures an activation descriptor.
func cudnnSetActivationDescriptor(desc cudnnActivationDescriptor, mode cudnnActivationMode, coef float64) error {
	status := C.cudnnSetActivationDescriptor(
		C.cudnnActivationDescriptor_t(unsafe.Pointer(desc)),
		C.cudnnActivationMode_t(mode),
		C.CUDNN_NOT_PROPAGATE_NAN,
		C.double(coef),
	)
	return cudnnError(status)
}

// cudnnActivationForward applies an activation function.
func cudnnActivationForward(
	handle cudnnHandle,
	activationDesc cudnnActivationDescriptor,
	alpha float32,
	xDesc cudnnTensorDescriptor,
	x uintptr,
	beta float32,
	yDesc cudnnTensorDescriptor,
	y uintptr,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cudnnActivationForward(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnActivationDescriptor_t(unsafe.Pointer(activationDesc)),
		unsafe.Pointer(&cAlpha),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		unsafe.Pointer(x),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		unsafe.Pointer(y),
	)
	return cudnnError(status)
}

// cudnnSoftmaxForward applies softmax.
func cudnnSoftmaxForward(
	handle cudnnHandle,
	algo cudnnSoftmaxAlgorithm,
	mode cudnnSoftmaxMode,
	alpha float32,
	xDesc cudnnTensorDescriptor,
	x uintptr,
	beta float32,
	yDesc cudnnTensorDescriptor,
	y uintptr,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cudnnSoftmaxForward(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnSoftmaxAlgorithm_t(algo),
		C.cudnnSoftmaxMode_t(mode),
		unsafe.Pointer(&cAlpha),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		unsafe.Pointer(x),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		unsafe.Pointer(y),
	)
	return cudnnError(status)
}

// cudnnFilterDescriptor describes a convolution filter.
type cudnnFilterDescriptor uintptr

// cudnnConvolutionDescriptor describes convolution parameters.
type cudnnConvolutionDescriptor uintptr

// cudnnConvolutionMode specifies the convolution mode.
type cudnnConvolutionMode int

const (
	cudnnConvolution      cudnnConvolutionMode = C.CUDNN_CONVOLUTION
	cudnnCrossCorrelation cudnnConvolutionMode = C.CUDNN_CROSS_CORRELATION
)

// cudnnConvolutionFwdAlgo specifies the forward convolution algorithm.
type cudnnConvolutionFwdAlgo int

const (
	cudnnConvolutionFwdAlgoImplicitGemm        cudnnConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
	cudnnConvolutionFwdAlgoImplicitPrecompGemm cudnnConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	cudnnConvolutionFwdAlgoGemm                cudnnConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM
)

// cudnnCreateFilterDescriptor creates a filter descriptor.
func cudnnCreateFilterDescriptor() (cudnnFilterDescriptor, error) {
	var desc C.cudnnFilterDescriptor_t
	status := C.cudnnCreateFilterDescriptor(&desc)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return cudnnFilterDescriptor(uintptr(unsafe.Pointer(desc))), nil
}

// cudnnDestroyFilterDescriptor destroys a filter descriptor.
func cudnnDestroyFilterDescriptor(desc cudnnFilterDescriptor) error {
	return cudnnError(C.cudnnDestroyFilterDescriptor(C.cudnnFilterDescriptor_t(unsafe.Pointer(desc))))
}

// cudnnSetFilter4dDescriptor sets a 4D filter descriptor.
// k = output channels, c = input channels, h = height, w = width
func cudnnSetFilter4dDescriptor(desc cudnnFilterDescriptor, k, c, h, w int) error {
	status := C.cudnnSetFilter4dDescriptor(
		C.cudnnFilterDescriptor_t(unsafe.Pointer(desc)),
		C.CUDNN_DATA_FLOAT,
		C.CUDNN_TENSOR_NCHW,
		C.int(k), C.int(c), C.int(h), C.int(w),
	)
	return cudnnError(status)
}

// cudnnCreateConvolutionDescriptor creates a convolution descriptor.
func cudnnCreateConvolutionDescriptor() (cudnnConvolutionDescriptor, error) {
	var desc C.cudnnConvolutionDescriptor_t
	status := C.cudnnCreateConvolutionDescriptor(&desc)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return cudnnConvolutionDescriptor(uintptr(unsafe.Pointer(desc))), nil
}

// cudnnDestroyConvolutionDescriptor destroys a convolution descriptor.
func cudnnDestroyConvolutionDescriptor(desc cudnnConvolutionDescriptor) error {
	return cudnnError(C.cudnnDestroyConvolutionDescriptor(C.cudnnConvolutionDescriptor_t(unsafe.Pointer(desc))))
}

// cudnnSetConvolution2dDescriptor sets 2D convolution parameters.
func cudnnSetConvolution2dDescriptor(desc cudnnConvolutionDescriptor, padH, padW, strideH, strideW, dilationH, dilationW int, mode cudnnConvolutionMode) error {
	status := C.cudnnSetConvolution2dDescriptor(
		C.cudnnConvolutionDescriptor_t(unsafe.Pointer(desc)),
		C.int(padH), C.int(padW),
		C.int(strideH), C.int(strideW),
		C.int(dilationH), C.int(dilationW),
		C.cudnnConvolutionMode_t(mode),
		C.CUDNN_DATA_FLOAT,
	)
	return cudnnError(status)
}

// cudnnGetConvolution2dForwardOutputDim computes the output dimensions.
func cudnnGetConvolution2dForwardOutputDim(
	convDesc cudnnConvolutionDescriptor,
	inputDesc cudnnTensorDescriptor,
	filterDesc cudnnFilterDescriptor,
) (n, c, h, w int, err error) {
	var cn, cc, ch, cw C.int
	status := C.cudnnGetConvolution2dForwardOutputDim(
		C.cudnnConvolutionDescriptor_t(unsafe.Pointer(convDesc)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(inputDesc)),
		C.cudnnFilterDescriptor_t(unsafe.Pointer(filterDesc)),
		&cn, &cc, &ch, &cw,
	)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, 0, 0, 0, cudnnError(status)
	}
	return int(cn), int(cc), int(ch), int(cw), nil
}

// cudnnConvolutionForward performs forward convolution.
func cudnnConvolutionForward(
	handle cudnnHandle,
	alpha float32,
	xDesc cudnnTensorDescriptor,
	x uintptr,
	wDesc cudnnFilterDescriptor,
	w uintptr,
	convDesc cudnnConvolutionDescriptor,
	algo cudnnConvolutionFwdAlgo,
	workspace uintptr,
	workspaceSize int,
	beta float32,
	yDesc cudnnTensorDescriptor,
	y uintptr,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cudnnConvolutionForward(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		unsafe.Pointer(&cAlpha),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		unsafe.Pointer(x),
		C.cudnnFilterDescriptor_t(unsafe.Pointer(wDesc)),
		unsafe.Pointer(w),
		C.cudnnConvolutionDescriptor_t(unsafe.Pointer(convDesc)),
		C.cudnnConvolutionFwdAlgo_t(algo),
		unsafe.Pointer(workspace),
		C.size_t(workspaceSize),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		unsafe.Pointer(y),
	)
	return cudnnError(status)
}

// cudnnGetConvolutionForwardWorkspaceSize returns the required workspace size.
func cudnnGetConvolutionForwardWorkspaceSize(
	handle cudnnHandle,
	xDesc cudnnTensorDescriptor,
	wDesc cudnnFilterDescriptor,
	convDesc cudnnConvolutionDescriptor,
	yDesc cudnnTensorDescriptor,
	algo cudnnConvolutionFwdAlgo,
) (int, error) {
	var size C.size_t
	status := C.cudnnGetConvolutionForwardWorkspaceSize(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		C.cudnnFilterDescriptor_t(unsafe.Pointer(wDesc)),
		C.cudnnConvolutionDescriptor_t(unsafe.Pointer(convDesc)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		C.cudnnConvolutionFwdAlgo_t(algo),
		&size,
	)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return int(size), nil
}

// cudnnDropoutDescriptor describes dropout parameters.
type cudnnDropoutDescriptor uintptr

// cudnnCreateDropoutDescriptor creates a dropout descriptor.
func cudnnCreateDropoutDescriptor() (cudnnDropoutDescriptor, error) {
	var desc C.cudnnDropoutDescriptor_t
	status := C.cudnnCreateDropoutDescriptor(&desc)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return cudnnDropoutDescriptor(uintptr(unsafe.Pointer(desc))), nil
}

// cudnnDestroyDropoutDescriptor destroys a dropout descriptor.
func cudnnDestroyDropoutDescriptor(desc cudnnDropoutDescriptor) error {
	return cudnnError(C.cudnnDestroyDropoutDescriptor(C.cudnnDropoutDescriptor_t(unsafe.Pointer(desc))))
}

// cudnnDropoutGetStatesSize returns the size needed for dropout states.
func cudnnDropoutGetStatesSize(handle cudnnHandle) (int, error) {
	var size C.size_t
	status := C.cudnnDropoutGetStatesSize(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		&size,
	)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return int(size), nil
}

// cudnnDropoutGetReserveSpaceSize returns the size needed for reserve space.
func cudnnDropoutGetReserveSpaceSize(xDesc cudnnTensorDescriptor) (int, error) {
	var size C.size_t
	status := C.cudnnDropoutGetReserveSpaceSize(
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		&size,
	)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return int(size), nil
}

// cudnnSetDropoutDescriptor configures a dropout descriptor.
func cudnnSetDropoutDescriptor(
	desc cudnnDropoutDescriptor,
	handle cudnnHandle,
	dropout float32,
	states uintptr,
	statesSizeInBytes int,
	seed uint64,
) error {
	status := C.cudnnSetDropoutDescriptor(
		C.cudnnDropoutDescriptor_t(unsafe.Pointer(desc)),
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.float(dropout),
		unsafe.Pointer(states),
		C.size_t(statesSizeInBytes),
		C.ulonglong(seed),
	)
	return cudnnError(status)
}

// cudnnDropoutForward applies dropout.
func cudnnDropoutForward(
	handle cudnnHandle,
	dropoutDesc cudnnDropoutDescriptor,
	xDesc cudnnTensorDescriptor,
	x uintptr,
	yDesc cudnnTensorDescriptor,
	y uintptr,
	reserveSpace uintptr,
	reserveSpaceSizeInBytes int,
) error {
	status := C.cudnnDropoutForward(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnDropoutDescriptor_t(unsafe.Pointer(dropoutDesc)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		unsafe.Pointer(x),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		unsafe.Pointer(y),
		unsafe.Pointer(reserveSpace),
		C.size_t(reserveSpaceSizeInBytes),
	)
	return cudnnError(status)
}

// Global cuDNN handle (lazily initialized)
var (
	globalCudnnHandle    cudnnHandle
	globalCudnnHandleErr error
	globalCudnnOnce      sync.Once
)

// getCudnnHandle returns the global cuDNN handle, initializing if needed.
// Thread-safe via sync.Once.
func getCudnnHandle() (cudnnHandle, error) {
	globalCudnnOnce.Do(func() {
		globalCudnnHandle, globalCudnnHandleErr = cudnnCreate()
	})
	return globalCudnnHandle, globalCudnnHandleErr
}

// cudnnPoolingDescriptor describes pooling parameters.
type cudnnPoolingDescriptor uintptr

// cudnnPoolingMode specifies the pooling operation type.
type cudnnPoolingMode int

const (
	cudnnPoolingMax                       cudnnPoolingMode = C.CUDNN_POOLING_MAX
	cudnnPoolingAverageCountIncludePading cudnnPoolingMode = C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
	cudnnPoolingAverageCountExcludePading cudnnPoolingMode = C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
)

// cudnnCreatePoolingDescriptor creates a pooling descriptor.
func cudnnCreatePoolingDescriptor() (cudnnPoolingDescriptor, error) {
	var desc C.cudnnPoolingDescriptor_t
	status := C.cudnnCreatePoolingDescriptor(&desc)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return cudnnPoolingDescriptor(uintptr(unsafe.Pointer(desc))), nil
}

// cudnnDestroyPoolingDescriptor destroys a pooling descriptor.
func cudnnDestroyPoolingDescriptor(desc cudnnPoolingDescriptor) error {
	return cudnnError(C.cudnnDestroyPoolingDescriptor(C.cudnnPoolingDescriptor_t(unsafe.Pointer(desc))))
}

// cudnnSetPooling2dDescriptor sets 2D pooling parameters.
func cudnnSetPooling2dDescriptor(desc cudnnPoolingDescriptor, mode cudnnPoolingMode, windowH, windowW, padH, padW, strideH, strideW int) error {
	status := C.cudnnSetPooling2dDescriptor(
		C.cudnnPoolingDescriptor_t(unsafe.Pointer(desc)),
		C.cudnnPoolingMode_t(mode),
		C.CUDNN_NOT_PROPAGATE_NAN,
		C.int(windowH), C.int(windowW),
		C.int(padH), C.int(padW),
		C.int(strideH), C.int(strideW),
	)
	return cudnnError(status)
}

// cudnnGetPooling2dForwardOutputDim computes pooling output dimensions.
func cudnnGetPooling2dForwardOutputDim(
	poolingDesc cudnnPoolingDescriptor,
	inputDesc cudnnTensorDescriptor,
) (n, c, h, w int, err error) {
	var cn, cc, ch, cw C.int
	status := C.cudnnGetPooling2dForwardOutputDim(
		C.cudnnPoolingDescriptor_t(unsafe.Pointer(poolingDesc)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(inputDesc)),
		&cn, &cc, &ch, &cw,
	)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, 0, 0, 0, cudnnError(status)
	}
	return int(cn), int(cc), int(ch), int(cw), nil
}

// cudnnPoolingForward applies pooling.
func cudnnPoolingForward(
	handle cudnnHandle,
	poolingDesc cudnnPoolingDescriptor,
	alpha float32,
	xDesc cudnnTensorDescriptor,
	x uintptr,
	beta float32,
	yDesc cudnnTensorDescriptor,
	y uintptr,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cudnnPoolingForward(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnPoolingDescriptor_t(unsafe.Pointer(poolingDesc)),
		unsafe.Pointer(&cAlpha),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		unsafe.Pointer(x),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		unsafe.Pointer(y),
	)
	return cudnnError(status)
}

// cudnnBatchNormMode specifies the batch normalization mode.
type cudnnBatchNormMode int

const (
	cudnnBatchNormPerActivation cudnnBatchNormMode = C.CUDNN_BATCHNORM_PER_ACTIVATION
	cudnnBatchNormSpatial       cudnnBatchNormMode = C.CUDNN_BATCHNORM_SPATIAL
)

// cudnnDeriveBNTensorDescriptor derives a batch normalization tensor descriptor.
func cudnnDeriveBNTensorDescriptor(derivedDesc, xDesc cudnnTensorDescriptor, mode cudnnBatchNormMode) error {
	status := C.cudnnDeriveBNTensorDescriptor(
		C.cudnnTensorDescriptor_t(unsafe.Pointer(derivedDesc)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		C.cudnnBatchNormMode_t(mode),
	)
	return cudnnError(status)
}

// cudnnBatchNormalizationForwardInference applies batch normalization in inference mode.
func cudnnBatchNormalizationForwardInference(
	handle cudnnHandle,
	mode cudnnBatchNormMode,
	alpha, beta float32,
	xDesc cudnnTensorDescriptor,
	x uintptr,
	yDesc cudnnTensorDescriptor,
	y uintptr,
	bnScaleBiasMeanVarDesc cudnnTensorDescriptor,
	bnScale uintptr,
	bnBias uintptr,
	estimatedMean uintptr,
	estimatedVariance uintptr,
	epsilon float64,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cudnnBatchNormalizationForwardInference(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnBatchNormMode_t(mode),
		unsafe.Pointer(&cAlpha),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		unsafe.Pointer(x),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		unsafe.Pointer(y),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(bnScaleBiasMeanVarDesc)),
		unsafe.Pointer(bnScale),
		unsafe.Pointer(bnBias),
		unsafe.Pointer(estimatedMean),
		unsafe.Pointer(estimatedVariance),
		C.double(epsilon),
	)
	return cudnnError(status)
}

// cudnnBatchNormalizationForwardTraining applies batch normalization in training mode.
// Computes batch statistics and updates running mean/variance with exponential moving average.
// Also outputs saveMean and saveInvVariance for use in backward pass.
func cudnnBatchNormalizationForwardTraining(
	handle cudnnHandle,
	mode cudnnBatchNormMode,
	alpha, beta float32,
	xDesc cudnnTensorDescriptor,
	x uintptr,
	yDesc cudnnTensorDescriptor,
	y uintptr,
	bnScaleBiasMeanVarDesc cudnnTensorDescriptor,
	bnScale uintptr,
	bnBias uintptr,
	exponentialAverageFactor float64,
	runningMean uintptr,
	runningVariance uintptr,
	epsilon float64,
	saveMean uintptr,
	saveInvVariance uintptr,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cudnnBatchNormalizationForwardTraining(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnBatchNormMode_t(mode),
		unsafe.Pointer(&cAlpha),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		unsafe.Pointer(x),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		unsafe.Pointer(y),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(bnScaleBiasMeanVarDesc)),
		unsafe.Pointer(bnScale),
		unsafe.Pointer(bnBias),
		C.double(exponentialAverageFactor),
		unsafe.Pointer(runningMean),
		unsafe.Pointer(runningVariance),
		C.double(epsilon),
		unsafe.Pointer(saveMean),
		unsafe.Pointer(saveInvVariance),
	)
	return cudnnError(status)
}

// cudnnReduceTensorDescriptor describes a reduce tensor operation.
type cudnnReduceTensorDescriptor uintptr

// cudnnReduceTensorOp specifies the reduction operation.
type cudnnReduceTensorOp int

const (
	cudnnReduceTensorAdd          cudnnReduceTensorOp = C.CUDNN_REDUCE_TENSOR_ADD
	cudnnReduceTensorMul          cudnnReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MUL
	cudnnReduceTensorMin          cudnnReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MIN
	cudnnReduceTensorMax          cudnnReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MAX
	cudnnReduceTensorAmax         cudnnReduceTensorOp = C.CUDNN_REDUCE_TENSOR_AMAX
	cudnnReduceTensorAvg          cudnnReduceTensorOp = C.CUDNN_REDUCE_TENSOR_AVG
	cudnnReduceTensorNorm1        cudnnReduceTensorOp = C.CUDNN_REDUCE_TENSOR_NORM1
	cudnnReduceTensorNorm2        cudnnReduceTensorOp = C.CUDNN_REDUCE_TENSOR_NORM2
	cudnnReduceTensorMulNoZeros   cudnnReduceTensorOp = C.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS
)

// cudnnCreateReduceTensorDescriptor creates a reduce tensor descriptor.
func cudnnCreateReduceTensorDescriptor() (cudnnReduceTensorDescriptor, error) {
	var desc C.cudnnReduceTensorDescriptor_t
	status := C.cudnnCreateReduceTensorDescriptor(&desc)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return cudnnReduceTensorDescriptor(uintptr(unsafe.Pointer(desc))), nil
}

// cudnnDestroyReduceTensorDescriptor destroys a reduce tensor descriptor.
func cudnnDestroyReduceTensorDescriptor(desc cudnnReduceTensorDescriptor) error {
	return cudnnError(C.cudnnDestroyReduceTensorDescriptor(C.cudnnReduceTensorDescriptor_t(unsafe.Pointer(desc))))
}

// cudnnSetReduceTensorDescriptor configures a reduce tensor descriptor.
func cudnnSetReduceTensorDescriptor(desc cudnnReduceTensorDescriptor, op cudnnReduceTensorOp) error {
	status := C.cudnnSetReduceTensorDescriptor(
		C.cudnnReduceTensorDescriptor_t(unsafe.Pointer(desc)),
		C.cudnnReduceTensorOp_t(op),
		C.CUDNN_DATA_FLOAT,
		C.CUDNN_NOT_PROPAGATE_NAN,
		C.CUDNN_REDUCE_TENSOR_NO_INDICES,
		C.CUDNN_32BIT_INDICES,
	)
	return cudnnError(status)
}

// cudnnGetReductionWorkspaceSize returns the workspace size needed for reduction.
func cudnnGetReductionWorkspaceSize(
	handle cudnnHandle,
	reduceDesc cudnnReduceTensorDescriptor,
	aDesc, cDesc cudnnTensorDescriptor,
) (int, error) {
	var size C.size_t
	status := C.cudnnGetReductionWorkspaceSize(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnReduceTensorDescriptor_t(unsafe.Pointer(reduceDesc)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(aDesc)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(cDesc)),
		&size,
	)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return int(size), nil
}

// cudnnReduceTensor performs tensor reduction.
func cudnnReduceTensor(
	handle cudnnHandle,
	reduceDesc cudnnReduceTensorDescriptor,
	workspace uintptr,
	workspaceSize int,
	alpha float32,
	aDesc cudnnTensorDescriptor,
	a uintptr,
	beta float32,
	cDesc cudnnTensorDescriptor,
	c uintptr,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cudnnReduceTensor(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnReduceTensorDescriptor_t(unsafe.Pointer(reduceDesc)),
		nil, 0, // indices (not used)
		unsafe.Pointer(workspace),
		C.size_t(workspaceSize),
		unsafe.Pointer(&cAlpha),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(aDesc)),
		unsafe.Pointer(a),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(cDesc)),
		unsafe.Pointer(c),
	)
	return cudnnError(status)
}

// cudnnSetTensorNdDescriptor sets an N-dimensional tensor descriptor.
func cudnnSetTensorNdDescriptor(desc cudnnTensorDescriptor, dims, strides []int) error {
	if len(dims) != len(strides) {
		return fmt.Errorf("dims and strides must have same length")
	}

	cDims := make([]C.int, len(dims))
	cStrides := make([]C.int, len(strides))
	for i := range dims {
		cDims[i] = C.int(dims[i])
		cStrides[i] = C.int(strides[i])
	}

	status := C.cudnnSetTensorNdDescriptor(
		C.cudnnTensorDescriptor_t(unsafe.Pointer(desc)),
		C.CUDNN_DATA_FLOAT,
		C.int(len(dims)),
		&cDims[0],
		&cStrides[0],
	)
	return cudnnError(status)
}

// cudnnOpTensorDescriptor describes an op tensor operation.
type cudnnOpTensorDescriptor uintptr

// cudnnOpTensorOp specifies the elementwise operation.
type cudnnOpTensorOp int

const (
	cudnnOpTensorAdd  cudnnOpTensorOp = C.CUDNN_OP_TENSOR_ADD
	cudnnOpTensorMul  cudnnOpTensorOp = C.CUDNN_OP_TENSOR_MUL
	cudnnOpTensorMin  cudnnOpTensorOp = C.CUDNN_OP_TENSOR_MIN
	cudnnOpTensorMax  cudnnOpTensorOp = C.CUDNN_OP_TENSOR_MAX
	cudnnOpTensorSqrt cudnnOpTensorOp = C.CUDNN_OP_TENSOR_SQRT
	cudnnOpTensorNot  cudnnOpTensorOp = C.CUDNN_OP_TENSOR_NOT
)

// cudnnCreateOpTensorDescriptor creates an op tensor descriptor.
func cudnnCreateOpTensorDescriptor() (cudnnOpTensorDescriptor, error) {
	var desc C.cudnnOpTensorDescriptor_t
	status := C.cudnnCreateOpTensorDescriptor(&desc)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return cudnnOpTensorDescriptor(uintptr(unsafe.Pointer(desc))), nil
}

// cudnnDestroyOpTensorDescriptor destroys an op tensor descriptor.
func cudnnDestroyOpTensorDescriptor(desc cudnnOpTensorDescriptor) error {
	return cudnnError(C.cudnnDestroyOpTensorDescriptor(C.cudnnOpTensorDescriptor_t(unsafe.Pointer(desc))))
}

// cudnnSetOpTensorDescriptor configures an op tensor descriptor.
func cudnnSetOpTensorDescriptor(desc cudnnOpTensorDescriptor, op cudnnOpTensorOp) error {
	status := C.cudnnSetOpTensorDescriptor(
		C.cudnnOpTensorDescriptor_t(unsafe.Pointer(desc)),
		C.cudnnOpTensorOp_t(op),
		C.CUDNN_DATA_FLOAT,
		C.CUDNN_NOT_PROPAGATE_NAN,
	)
	return cudnnError(status)
}

// cudnnOpTensor performs elementwise tensor operations.
// C = op(alpha1 * A, alpha2 * B) + beta * C
func cudnnOpTensor(
	handle cudnnHandle,
	opDesc cudnnOpTensorDescriptor,
	alpha1 float32,
	aDesc cudnnTensorDescriptor,
	a uintptr,
	alpha2 float32,
	bDesc cudnnTensorDescriptor,
	b uintptr,
	beta float32,
	cDesc cudnnTensorDescriptor,
	c uintptr,
) error {
	cAlpha1 := C.float(alpha1)
	cAlpha2 := C.float(alpha2)
	cBeta := C.float(beta)

	status := C.cudnnOpTensor(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnOpTensorDescriptor_t(unsafe.Pointer(opDesc)),
		unsafe.Pointer(&cAlpha1),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(aDesc)),
		unsafe.Pointer(a),
		unsafe.Pointer(&cAlpha2),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(bDesc)),
		unsafe.Pointer(b),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(cDesc)),
		unsafe.Pointer(c),
	)
	return cudnnError(status)
}

// cudnnScaleTensor scales all elements of a tensor: y = alpha * y
func cudnnScaleTensor(
	handle cudnnHandle,
	yDesc cudnnTensorDescriptor,
	y uintptr,
	alpha float32,
) error {
	cAlpha := C.float(alpha)
	status := C.cudnnScaleTensor(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		unsafe.Pointer(y),
		unsafe.Pointer(&cAlpha),
	)
	return cudnnError(status)
}

// cudnnAddTensor adds a tensor to another: C = alpha * A + beta * C
// A is broadcast to match C's dimensions. Useful for bias addition.
func cudnnAddTensor(
	handle cudnnHandle,
	alpha float32,
	aDesc cudnnTensorDescriptor,
	a uintptr,
	beta float32,
	cDesc cudnnTensorDescriptor,
	c uintptr,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)
	status := C.cudnnAddTensor(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		unsafe.Pointer(&cAlpha),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(aDesc)),
		unsafe.Pointer(a),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(cDesc)),
		unsafe.Pointer(c),
	)
	return cudnnError(status)
}

// cudnnTransformTensor copies data between tensors with optional scaling.
// y = alpha * x + beta * y
// Useful for format conversion or making tensors contiguous.
func cudnnTransformTensor(
	handle cudnnHandle,
	alpha float32,
	xDesc cudnnTensorDescriptor,
	x uintptr,
	beta float32,
	yDesc cudnnTensorDescriptor,
	y uintptr,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)
	status := C.cudnnTransformTensor(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		unsafe.Pointer(&cAlpha),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(xDesc)),
		unsafe.Pointer(x),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(yDesc)),
		unsafe.Pointer(y),
	)
	return cudnnError(status)
}

// cudnnConvolutionBwdDataAlgo specifies the backward data convolution algorithm.
type cudnnConvolutionBwdDataAlgo int

const (
	cudnnConvolutionBwdDataAlgo0 cudnnConvolutionBwdDataAlgo = C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
	cudnnConvolutionBwdDataAlgo1 cudnnConvolutionBwdDataAlgo = C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
)

// cudnnGetConvolutionBackwardDataWorkspaceSize returns the required workspace size for backward data convolution.
func cudnnGetConvolutionBackwardDataWorkspaceSize(
	handle cudnnHandle,
	wDesc cudnnFilterDescriptor,
	dyDesc cudnnTensorDescriptor,
	convDesc cudnnConvolutionDescriptor,
	dxDesc cudnnTensorDescriptor,
	algo cudnnConvolutionBwdDataAlgo,
) (int, error) {
	var size C.size_t
	status := C.cudnnGetConvolutionBackwardDataWorkspaceSize(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		C.cudnnFilterDescriptor_t(unsafe.Pointer(wDesc)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(dyDesc)),
		C.cudnnConvolutionDescriptor_t(unsafe.Pointer(convDesc)),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(dxDesc)),
		C.cudnnConvolutionBwdDataAlgo_t(algo),
		&size,
	)
	if status != C.CUDNN_STATUS_SUCCESS {
		return 0, cudnnError(status)
	}
	return int(size), nil
}

// cudnnConvolutionBackwardData performs backward data convolution (transposed convolution).
// This computes dx from dy and w, which is equivalent to ConvTranspose2d for inference.
func cudnnConvolutionBackwardData(
	handle cudnnHandle,
	alpha float32,
	wDesc cudnnFilterDescriptor,
	w uintptr,
	dyDesc cudnnTensorDescriptor,
	dy uintptr,
	convDesc cudnnConvolutionDescriptor,
	algo cudnnConvolutionBwdDataAlgo,
	workspace uintptr,
	workspaceSize int,
	beta float32,
	dxDesc cudnnTensorDescriptor,
	dx uintptr,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cudnnConvolutionBackwardData(
		C.cudnnHandle_t(unsafe.Pointer(handle)),
		unsafe.Pointer(&cAlpha),
		C.cudnnFilterDescriptor_t(unsafe.Pointer(wDesc)),
		unsafe.Pointer(w),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(dyDesc)),
		unsafe.Pointer(dy),
		C.cudnnConvolutionDescriptor_t(unsafe.Pointer(convDesc)),
		C.cudnnConvolutionBwdDataAlgo_t(algo),
		unsafe.Pointer(workspace),
		C.size_t(workspaceSize),
		unsafe.Pointer(&cBeta),
		C.cudnnTensorDescriptor_t(unsafe.Pointer(dxDesc)),
		unsafe.Pointer(dx),
	)
	return cudnnError(status)
}
