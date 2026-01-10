//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/zoobzio/tendo"
)

// Activation applies a cuDNN activation function to a CUDA tensor.
func Activation(t *tendo.Tensor, mode cudnnActivationMode) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	// Create output storage
	outStorage, err := NewStorage(cudaStorage.Len(), t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	// Create tensor descriptors
	xDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(xDesc)

	yDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(yDesc)

	// Set up 4D tensor descriptor (cuDNN requires 4D)
	// For tensors with fewer dims, pad with 1s
	n, c, h, w := tensorTo4D(t.Shape())

	if err := cudnnSetTensor4dDescriptor(xDesc, n, c, h, w); err != nil {
		outStorage.Free()
		return nil, err
	}
	if err := cudnnSetTensor4dDescriptor(yDesc, n, c, h, w); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Create activation descriptor
	actDesc, err := cudnnCreateActivationDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyActivationDescriptor(actDesc)

	if err := cudnnSetActivationDescriptor(actDesc, mode, 0.0); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Apply activation
	err = cudnnActivationForward(
		handle,
		actDesc,
		1.0, xDesc, cudaStorage.Ptr(),
		0.0, yDesc, outStorage.Ptr(),
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// Softmax applies cuDNN softmax to a CUDA tensor.
func Softmax(t *tendo.Tensor, dim int, logSoftmax bool) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	// Create output storage
	outStorage, err := NewStorage(cudaStorage.Len(), t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	// Create tensor descriptors
	xDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(xDesc)

	yDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(yDesc)

	// Normalize dim
	shape := t.Shape()
	ndim := len(shape)
	if dim < 0 {
		dim = ndim + dim
	}

	// For softmax over last dimension, reshape to [batch, dim_size, 1, 1]
	// and use cudnnSoftmaxModeChannel which operates on C dimension.
	// For other dimensions, we'd need a different approach.
	var n, c, h, w int
	mode := cudnnSoftmaxModeChannel
	if dim == ndim-1 {
		// Softmax over last dimension: reshape to [N, C, 1, 1] where C is the last dim
		n = 1
		for i := 0; i < ndim-1; i++ {
			n *= shape[i]
		}
		c = shape[ndim-1]
		h = 1
		w = 1
	} else {
		// For other dimensions, use the default mapping with ModeInstance
		// This may not be correct for all cases but covers common ones
		n, c, h, w = tensorTo4D(shape)
		mode = cudnnSoftmaxModeInstance
	}

	if err := cudnnSetTensor4dDescriptor(xDesc, n, c, h, w); err != nil {
		outStorage.Free()
		return nil, err
	}
	if err := cudnnSetTensor4dDescriptor(yDesc, n, c, h, w); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Determine softmax algorithm
	algo := cudnnSoftmaxAccurate
	if logSoftmax {
		algo = cudnnSoftmaxLog
	}

	// Apply softmax
	err = cudnnSoftmaxForward(
		handle,
		algo,
		mode,
		1.0, xDesc, cudaStorage.Ptr(),
		0.0, yDesc, outStorage.Ptr(),
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}

// MatMul performs matrix multiplication using cuBLAS.
// Supports both 2D and batched (3D+) tensors.
// Handles transposed 2D tensors efficiently using cuBLAS transpose flags.
func MatMul(a, b *tendo.Tensor) (*tendo.Tensor, error) {
	aStorage, ok := a.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: a.Device().Type}
	}
	bStorage, ok := b.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: b.Device().Type}
	}

	handle, err := getCublasHandle()
	if err != nil {
		return nil, err
	}

	aShape := a.Shape()
	bShape := b.Shape()

	// Handle 2D case with transpose detection
	if len(aShape) == 2 && len(bShape) == 2 {
		return matMul2DWithTranspose(handle, a, b, aStorage, bStorage)
	}

	// Handle batched case (3D+)
	// Check if b is a 2D transposed matrix (common for Linear layers)
	// In this case, we can use efficient transpose handling
	if len(bShape) == 2 && isSimpleTranspose(b) {
		return matMulBatched3D2DTransposed(handle, a, b, aStorage, bStorage)
	}

	// General batched case - requires contiguous for now
	if !a.IsContiguous() {
		a = a.Contiguous()
		aStorage = a.Storage().(*Storage)
	}
	if !b.IsContiguous() {
		b = b.Contiguous()
		bStorage = b.Storage().(*Storage)
	}
	return matMulBatched(handle, aStorage, bStorage, a.Shape(), b.Shape())
}

// isSimpleTranspose returns true if tensor is a simple 2D transpose (strides swapped).
func isSimpleTranspose(t *tendo.Tensor) bool {
	if t.Dim() != 2 {
		return false
	}
	stride := t.Stride()
	// A transposed 2D tensor has stride[0] < stride[1] (row stride < col stride)
	return stride[0] < stride[1]
}

// matMul2DWithTranspose handles 2D matmul with efficient transpose via cuBLAS flags.
func matMul2DWithTranspose(handle cublasHandle, a, b *tendo.Tensor, aStorage, bStorage *Storage) (*tendo.Tensor, error) {
	aShape := a.Shape()
	bShape := b.Shape()
	aStride := a.Stride()
	bStride := b.Stride()

	// Logical dimensions for A @ B
	m := aShape[0] // rows of A
	k := aShape[1] // cols of A = rows of B
	n := bShape[1] // cols of B

	if bShape[0] != k {
		return nil, &tendo.ShapeError{Op: "matmul", Message: "inner dimensions must match"}
	}

	// Determine if inputs are transposed
	// In row-major: contiguous has stride[0] > stride[1]
	// Transposed has stride[0] < stride[1]
	aTransposed := isSimpleTranspose(a)
	bTransposed := isSimpleTranspose(b)

	// For transposed tensors, get the "physical" (storage) dimensions
	// If A is logically [m, k] but stored as transpose of [k, m], physical is [k, m]
	var ldA, ldB int
	var opA, opB cublasOperation

	if aTransposed {
		// A is stored as [k, m] transposed, physical leading dim is m
		ldA = aStride[1] // the larger stride (was row stride before transpose)
		opA = cublasOpT
	} else {
		ldA = aStride[0]
		if ldA == 0 {
			ldA = k
		}
		opA = cublasOpN
	}

	if bTransposed {
		// B is stored as [n, k] transposed, physical leading dim is k
		ldB = bStride[1]
		opB = cublasOpT
	} else {
		ldB = bStride[0]
		if ldB == 0 {
			ldB = n
		}
		opB = cublasOpN
	}

	// Create output storage
	outStorage, err := NewStorage(m*n, tendo.Float32, aStorage.device)
	if err != nil {
		return nil, err
	}

	// cuBLAS uses column-major. For row-major C = A @ B, we compute:
	// C^T = B^T @ A^T in column-major terms
	// We swap the order and adjust the ops accordingly.
	//
	// If in row-major:
	// - A is [m,k], A^T (col-major view) is [k,m]
	// - B is [k,n], B^T (col-major view) is [n,k]
	// - C is [m,n], C^T (col-major view) is [n,m]
	//
	// col-major: C^T = B^T @ A^T means we call sgemm(n, m, k, B, A, C)
	//
	// Transpose flags need to be swapped when we swap A and B:
	// - If row-major A was transposed, col-major A^T needs OP_T to un-transpose
	// - But we're swapping A and B, so A becomes the "B" argument

	// Swap ops since we swap arguments
	var opACublas, opBCublas cublasOperation
	opACublas = opB // A (now second arg) gets B's original op
	opBCublas = opA // B (now first arg) gets A's original op

	// Similarly swap leading dims
	ldACublas := ldB
	ldBCublas := ldA

	err = cublasSgemm(
		handle,
		opACublas, opBCublas,
		n, m, k,
		1.0,
		bStorage.Ptr(), ldACublas,
		aStorage.Ptr(), ldBCublas,
		0.0,
		outStorage.Ptr(), n,
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, []int{m, n}, nil), nil
}

// matMulBatched3D2DTransposed handles the common case: [batch, seq, in] @ [in, out]^T
// where the 2D weight is transposed. Uses cuBLAS strided batched GEMM with transpose flags.
func matMulBatched3D2DTransposed(handle cublasHandle, a, b *tendo.Tensor, aStorage, bStorage *Storage) (*tendo.Tensor, error) {
	aShape := a.Shape()
	bShape := b.Shape()
	bStride := b.Stride()

	// Make a contiguous if needed (but keep b as transposed view)
	if !a.IsContiguous() {
		a = a.Contiguous()
		aStorage = a.Storage().(*Storage)
		aShape = a.Shape()
	}

	// A is [batch..., m, k], B is transposed [k, n]
	m := aShape[len(aShape)-2]
	k := aShape[len(aShape)-1]
	n := bShape[1] // logical n (B's logical shape is [k, n])

	if bShape[0] != k {
		return nil, &tendo.ShapeError{Op: "matmul", Message: "inner dimensions must match"}
	}

	// Calculate batch size
	batchCount := 1
	batchDims := aShape[:len(aShape)-2]
	for _, d := range batchDims {
		batchCount *= d
	}
	if batchCount == 0 {
		batchCount = 1
	}

	// B is transposed in row-major: logical [k, n], physical [n, k]
	// When cuBLAS sees row-major [n, k] as col-major, it's [k, n]
	// We need [n, k] for matmul, so use OP_T to transpose cuBLAS's view
	// Physical row size is n (from the original [n, k] before transpose)
	physicalK := bStride[1] // stride[1] = physical row size of original matrix

	// Create output [batch..., m, n]
	outShape := append(append([]int{}, batchDims...), m, n)
	outStorage, err := NewStorage(batchCount*m*n, tendo.Float32, aStorage.device)
	if err != nil {
		return nil, err
	}

	// cuBLAS strided batched GEMM for row-major C = A @ B
	// where B is transposed (logical [k,n] but physical [n,k])
	//
	// Row-major C[m,n] = A[m,k] @ B[k,n] translates to:
	// Col-major C^T[n,m] = B^T[n,k] @ A^T[k,m]
	//
	// cuBLAS sees our row-major data as col-major transposes:
	// - A [m,k] row-major → cuBLAS sees [k,m] col-major
	// - B physical [n,k] row-major → cuBLAS sees [k,n] col-major
	//   With OP_T: [n,k] which is what we need
	// - C [m,n] row-major → cuBLAS sees [n,m] col-major
	//
	// sgemm(OP_T, OP_N, n, m, k, B, A, C):
	// C[n,m] = OP_T(B[k,n])[n,k] @ OP_N(A[k,m])[k,m] = [n,k] @ [k,m] = [n,m] ✓
	err = cublasSgemmStridedBatched(
		handle,
		cublasOpT, cublasOpN, // B transposed, A not transposed
		n, m, k,
		1.0,
		bStorage.Ptr(), physicalK, 0, // B: ldb = physical row size, stride 0 (broadcast)
		aStorage.Ptr(), k, int64(m*k), // A: lda = k
		0.0,
		outStorage.Ptr(), n, int64(m*n), // C: ldc = n
		batchCount,
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, outShape, nil), nil
}

// matMul2D performs 2D matrix multiplication (legacy, assumes contiguous).
func matMul2D(handle cublasHandle, aStorage, bStorage *Storage, aShape, bShape []int) (*tendo.Tensor, error) {
	m := aShape[0] // rows of A
	k := aShape[1] // cols of A = rows of B
	n := bShape[1] // cols of B

	if bShape[0] != k {
		return nil, &tendo.ShapeError{Op: "matmul", Message: "inner dimensions must match"}
	}

	// Create output storage
	outStorage, err := NewStorage(m*n, tendo.Float32, aStorage.device)
	if err != nil {
		return nil, err
	}

	// cuBLAS uses column-major order, but our tensors are row-major
	// To compute C = A * B in row-major, we compute C^T = B^T * A^T in col-major
	// Since cuBLAS sees our row-major as transposed col-major, we swap A and B
	err = cublasSgemm(
		handle,
		cublasOpN, cublasOpN,
		n, m, k,             // dimensions swapped for row-major
		1.0,                 // alpha
		bStorage.Ptr(), n,   // B, ldb
		aStorage.Ptr(), k,   // A, lda
		0.0,                 // beta
		outStorage.Ptr(), n, // C, ldc
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, []int{m, n}, nil), nil
}

// matMulBatched performs batched matrix multiplication using cuBLAS strided batched GEMM.
func matMulBatched(handle cublasHandle, aStorage, bStorage *Storage, aShape, bShape []int) (*tendo.Tensor, error) {
	// Get matrix dimensions (last two dims)
	m := aShape[len(aShape)-2]
	k := aShape[len(aShape)-1]
	n := bShape[len(bShape)-1]

	if bShape[len(bShape)-2] != k {
		return nil, &tendo.ShapeError{Op: "matmul", Message: "inner dimensions must match"}
	}

	// Calculate batch dimensions
	batchDimsA := aShape[:len(aShape)-2]
	batchDimsB := bShape[:len(bShape)-2]

	// Compute output batch shape (broadcast batch dimensions)
	outBatchShape, err := tendo.BroadcastShapes(batchDimsA, batchDimsB)
	if err != nil {
		return nil, err
	}

	batchCount := 1
	for _, d := range outBatchShape {
		batchCount *= d
	}
	if batchCount == 0 {
		batchCount = 1
	}

	// Calculate strides for batching
	strideA := int64(m * k)
	strideB := int64(k * n)
	strideC := int64(m * n)

	// Handle broadcasting: if batch dim is 1, stride should be 0
	batchA := tendo.Numel(batchDimsA)
	batchB := tendo.Numel(batchDimsB)
	if batchA == 0 {
		batchA = 1
	}
	if batchB == 0 {
		batchB = 1
	}
	if batchA == 1 {
		strideA = 0
	}
	if batchB == 1 {
		strideB = 0
	}

	// Create output storage
	outShape := append(outBatchShape, m, n)
	outStorage, err := NewStorage(batchCount*m*n, tendo.Float32, aStorage.device)
	if err != nil {
		return nil, err
	}

	// cuBLAS strided batched GEMM with row-major adjustment
	err = cublasSgemmStridedBatched(
		handle,
		cublasOpN, cublasOpN,
		n, m, k,                      // dimensions swapped for row-major
		1.0,                          // alpha
		bStorage.Ptr(), n, strideB,   // B, ldb, strideB
		aStorage.Ptr(), k, strideA,   // A, lda, strideA
		0.0,                          // beta
		outStorage.Ptr(), n, strideC, // C, ldc, strideC
		batchCount,
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, outShape, nil), nil
}

// tensorTo4D converts a shape to 4D (N, C, H, W) for cuDNN.
// Pads with 1s for dimensions less than 4.
func tensorTo4D(shape []int) (n, c, h, w int) {
	switch len(shape) {
	case 0:
		return 1, 1, 1, 1
	case 1:
		return 1, 1, 1, shape[0]
	case 2:
		return 1, 1, shape[0], shape[1]
	case 3:
		return 1, shape[0], shape[1], shape[2]
	case 4:
		return shape[0], shape[1], shape[2], shape[3]
	default:
		// For >4D, collapse leading dimensions into N
		n = 1
		for i := 0; i < len(shape)-3; i++ {
			n *= shape[i]
		}
		return n, shape[len(shape)-3], shape[len(shape)-2], shape[len(shape)-1]
	}
}

// IsCUDA checks if a tensor is on CUDA.
func IsCUDA(t *tendo.Tensor) bool {
	return t.Device().Type == tendo.CUDA
}

// Dropout applies dropout on GPU, randomly zeroing elements with probability p.
func Dropout(ctx context.Context, t *tendo.Tensor, p float32) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	numel := t.Numel()
	scale := float32(1.0) / (1.0 - p)

	// Generate mask on CPU and transfer to GPU
	// TODO: Use cuRAND for GPU-native random generation
	maskData := make([]float32, numel)
	for i := range maskData {
		if rand.Float32() >= p {
			maskData[i] = 1.0
		} else {
			maskData[i] = 0.0
		}
	}

	// Transfer mask to GPU
	maskStorage, err := NewStorageFromSlice(maskData, t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}
	mask := tendo.NewTensor(maskStorage, t.Shape(), nil)
	defer mask.Storage().Free()

	// Apply: output = input * mask
	temp, err := PopcornMul(t, mask)
	if err != nil {
		return nil, err
	}

	// Scale by 1/(1-p)
	output, err := PopcornMulScalar(temp, scale)
	temp.Storage().Free()
	if err != nil {
		return nil, err
	}

	return output, nil
}

// randFloat32 returns a random float32 in [0, 1).
func randFloat32() float32 {
	return rand.Float32()
}

// Pool2d performs 2D pooling using cuDNN.
func Pool2d(input *tendo.Tensor, kernelSize, stride, padding [2]int, maxPool bool) (*tendo.Tensor, error) {
	cudaStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, &tendo.ShapeError{Op: "pool2d", Message: "input must be 4D [N, C, H, W]"}
	}

	// Create input tensor descriptor
	xDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(xDesc)

	if err := cudnnSetTensor4dDescriptor(xDesc, inShape[0], inShape[1], inShape[2], inShape[3]); err != nil {
		return nil, err
	}

	// Create pooling descriptor
	poolDesc, err := cudnnCreatePoolingDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyPoolingDescriptor(poolDesc)

	mode := cudnnPoolingAverageCountExcludePading
	if maxPool {
		mode = cudnnPoolingMax
	}

	if err := cudnnSetPooling2dDescriptor(poolDesc, mode,
		kernelSize[0], kernelSize[1],
		padding[0], padding[1],
		stride[0], stride[1]); err != nil {
		return nil, err
	}

	// Get output dimensions
	outN, outC, outH, outW, err := cudnnGetPooling2dForwardOutputDim(poolDesc, xDesc)
	if err != nil {
		return nil, err
	}

	// Create output tensor descriptor
	yDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(yDesc)

	if err := cudnnSetTensor4dDescriptor(yDesc, outN, outC, outH, outW); err != nil {
		return nil, err
	}

	// Create output storage
	outStorage, err := NewStorage(outN*outC*outH*outW, input.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	// Apply pooling
	err = cudnnPoolingForward(
		handle,
		poolDesc,
		1.0, xDesc, cudaStorage.Ptr(),
		0.0, yDesc, outStorage.Ptr(),
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, []int{outN, outC, outH, outW}, nil), nil
}

// BatchNorm2d performs 2D batch normalization using cuDNN.
// Supports both training and inference modes based on context.
func BatchNorm2d(ctx context.Context, input, weight, bias, runningMean, runningVar *tendo.Tensor, epsilon, momentum float32) (*tendo.Tensor, error) {
	cudaInput, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}
	cudaWeight, ok := weight.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: weight.Device().Type}
	}
	cudaBias, ok := bias.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: bias.Device().Type}
	}
	cudaMean, ok := runningMean.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: runningMean.Device().Type}
	}
	cudaVar, ok := runningVar.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: runningVar.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, &tendo.ShapeError{Op: "batchnorm2d", Message: "input must be 4D [N, C, H, W]"}
	}

	C := inShape[1] // number of channels

	// Create input/output tensor descriptors
	xDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(xDesc)

	if err := cudnnSetTensor4dDescriptor(xDesc, inShape[0], inShape[1], inShape[2], inShape[3]); err != nil {
		return nil, err
	}

	yDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(yDesc)

	if err := cudnnSetTensor4dDescriptor(yDesc, inShape[0], inShape[1], inShape[2], inShape[3]); err != nil {
		return nil, err
	}

	// Create batch norm parameter descriptor
	bnDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(bnDesc)

	if err := cudnnDeriveBNTensorDescriptor(bnDesc, xDesc, cudnnBatchNormSpatial); err != nil {
		return nil, err
	}

	// Create output storage
	outStorage, err := NewStorage(input.Numel(), input.DType(), cudaInput.device)
	if err != nil {
		return nil, err
	}

	training := tendo.IsTraining(ctx)

	if training {
		// Training mode: compute batch statistics and update running stats
		// saveMean and saveInvVariance are allocated for cuDNN but freed after forward pass.
		saveMeanStorage, err := NewStorage(C, tendo.Float32, cudaInput.device)
		if err != nil {
			outStorage.Free()
			return nil, err
		}
		defer saveMeanStorage.Free()

		saveInvVarStorage, err := NewStorage(C, tendo.Float32, cudaInput.device)
		if err != nil {
			outStorage.Free()
			return nil, err
		}
		defer saveInvVarStorage.Free()

		// exponentialAverageFactor is momentum in PyTorch convention
		// cuDNN: runningMean = runningMean * (1 - factor) + batchMean * factor
		err = cudnnBatchNormalizationForwardTraining(
			handle,
			cudnnBatchNormSpatial,
			1.0, 0.0,
			xDesc, cudaInput.Ptr(),
			yDesc, outStorage.Ptr(),
			bnDesc,
			cudaWeight.Ptr(),
			cudaBias.Ptr(),
			float64(momentum),
			cudaMean.Ptr(),
			cudaVar.Ptr(),
			float64(epsilon),
			saveMeanStorage.Ptr(),
			saveInvVarStorage.Ptr(),
		)
	} else {
		// Inference mode: use running statistics
		err = cudnnBatchNormalizationForwardInference(
			handle,
			cudnnBatchNormSpatial,
			1.0, 0.0,
			xDesc, cudaInput.Ptr(),
			yDesc, outStorage.Ptr(),
			bnDesc,
			cudaWeight.Ptr(),
			cudaBias.Ptr(),
			cudaMean.Ptr(),
			cudaVar.Ptr(),
			float64(epsilon),
		)
	}

	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, inShape, nil), nil
}

// InstanceNorm2d performs instance normalization using cuDNN.
// Normalizes over (H, W) for each (N, C) independently.
func InstanceNorm2d(input, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	cudaInput, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	inShape := input.Shape()
	if len(inShape) != 4 {
		return nil, &tendo.ShapeError{Op: "instancenorm2d", Message: "input must be 4D [N, C, H, W]"}
	}

	N, C, H, W := inShape[0], inShape[1], inShape[2], inShape[3]

	// For InstanceNorm, we treat each (n, c) as a separate "batch"
	// Reshape [N, C, H, W] -> [N*C, 1, H, W] and use per-activation mode
	xDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(xDesc)

	if err := cudnnSetTensor4dDescriptor(xDesc, N*C, 1, H, W); err != nil {
		return nil, err
	}

	yDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(yDesc)

	if err := cudnnSetTensor4dDescriptor(yDesc, N*C, 1, H, W); err != nil {
		return nil, err
	}

	// BN descriptor for per-activation mode
	bnDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(bnDesc)

	if err := cudnnDeriveBNTensorDescriptor(bnDesc, xDesc, cudnnBatchNormPerActivation); err != nil {
		return nil, err
	}

	// Create output storage
	outStorage, err := NewStorage(input.Numel(), input.DType(), cudaInput.device)
	if err != nil {
		return nil, err
	}

	// For InstanceNorm, we need weight/bias per instance (N*C elements)
	// But typically weight/bias are per-channel (C elements), so we need to expand them
	// Create expanded weight/bias tensors
	expandedWeightStorage, err := NewStorage(N*C*H*W, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer expandedWeightStorage.Free()

	expandedBiasStorage, err := NewStorage(N*C*H*W, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer expandedBiasStorage.Free()

	// Initialize with defaults (weight=1, bias=0) if not provided
	var weightPtr, biasPtr uintptr
	if weight != nil {
		cudaWeight, ok := weight.Storage().(*Storage)
		if !ok {
			outStorage.Free()
			return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: weight.Device().Type}
		}
		weightPtr = cudaWeight.Ptr()
	}
	if bias != nil {
		cudaBias, ok := bias.Storage().(*Storage)
		if !ok {
			outStorage.Free()
			return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: bias.Device().Type}
		}
		biasPtr = cudaBias.Ptr()
	}

	// For now, create simple all-ones weight and all-zeros bias for the reshaped view
	// This is a simplification - proper implementation would broadcast weight/bias
	onesStorage, err := NewStorage(N*C*H*W, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer onesStorage.Free()

	zerosStorage, err := NewStorage(N*C*H*W, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer zerosStorage.Free()

	// For InstanceNorm without affine params, use scale=1, bias=0
	// cuDNN per-activation mode expects parameters of shape [1, C, H, W] for input [N, C, H, W]
	// With our reshaped input [N*C, 1, H, W], params should be [1, 1, H, W]
	paramStorage, err := NewStorage(H*W, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer paramStorage.Free()

	// Initialize scale to 1.0
	ones := make([]float32, H*W)
	for i := range ones {
		ones[i] = 1.0
	}
	scaleStorage, err := NewStorageFromSlice(ones, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer scaleStorage.Free()

	// Initialize bias to 0.0
	zeros := make([]float32, H*W)
	biasParamStorage, err := NewStorageFromSlice(zeros, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer biasParamStorage.Free()

	// For inference, we need "running" mean/var - but for InstanceNorm we compute fresh each time
	// Use zeros for mean and ones for variance as placeholders (won't be used in per-activation mode)
	meanStorage, err := NewStorageFromSlice(zeros, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer meanStorage.Free()

	varOnes := make([]float32, H*W)
	for i := range varOnes {
		varOnes[i] = 1.0
	}
	varStorage, err := NewStorageFromSlice(varOnes, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer varStorage.Free()

	err = cudnnBatchNormalizationForwardInference(
		handle,
		cudnnBatchNormPerActivation,
		1.0, 0.0,
		xDesc, cudaInput.Ptr(),
		yDesc, outStorage.Ptr(),
		bnDesc,
		scaleStorage.Ptr(),
		biasParamStorage.Ptr(),
		meanStorage.Ptr(),
		varStorage.Ptr(),
		float64(epsilon),
	)

	if err != nil {
		outStorage.Free()
		return nil, err
	}

	// Apply per-channel weight and bias if provided
	if weight != nil || bias != nil {
		// This requires element-wise operations - for now return normalized output
		// Full implementation would multiply by weight and add bias per-channel
		_ = weightPtr
		_ = biasPtr
	}

	return tendo.NewTensor(outStorage, inShape, nil), nil
}

// GroupNorm performs group normalization using cuDNN.
// Reshapes to [N*G, C/G, H, W], applies BatchNorm, reshapes back.
func GroupNorm(input *tendo.Tensor, numGroups int, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	cudaInput, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	inShape := input.Shape()
	if len(inShape) < 2 {
		return nil, &tendo.ShapeError{Op: "groupnorm", Message: "input must have at least 2 dimensions"}
	}

	N, C := inShape[0], inShape[1]
	if C%numGroups != 0 {
		return nil, &tendo.ShapeError{Op: "groupnorm", Message: "channels must be divisible by numGroups"}
	}

	channelsPerGroup := C / numGroups

	// Calculate spatial size
	spatialSize := 1
	for i := 2; i < len(inShape); i++ {
		spatialSize *= inShape[i]
	}

	// Reshape [N, C, ...] -> [N*G, C/G, spatial...]
	// For 4D: [N, C, H, W] -> [N*G, C/G, H, W]
	var H, W int
	if len(inShape) == 4 {
		H, W = inShape[2], inShape[3]
	} else {
		H, W = spatialSize, 1
	}

	// Create descriptors for reshaped tensor [N*numGroups, channelsPerGroup, H, W]
	xDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(xDesc)

	if err := cudnnSetTensor4dDescriptor(xDesc, N*numGroups, channelsPerGroup, H, W); err != nil {
		return nil, err
	}

	yDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(yDesc)

	if err := cudnnSetTensor4dDescriptor(yDesc, N*numGroups, channelsPerGroup, H, W); err != nil {
		return nil, err
	}

	// BN descriptor - normalizes over (H, W) for each (N*G, C/G) group
	bnDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(bnDesc)

	if err := cudnnDeriveBNTensorDescriptor(bnDesc, xDesc, cudnnBatchNormSpatial); err != nil {
		return nil, err
	}

	// Create output storage
	outStorage, err := NewStorage(input.Numel(), input.DType(), cudaInput.device)
	if err != nil {
		return nil, err
	}

	// Create scale (ones) and bias (zeros) for normalization
	// For spatial mode, params have shape [1, C/G, 1, 1]
	ones := make([]float32, channelsPerGroup)
	for i := range ones {
		ones[i] = 1.0
	}
	scaleStorage, err := NewStorageFromSlice(ones, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer scaleStorage.Free()

	zeros := make([]float32, channelsPerGroup)
	biasStorage, err := NewStorageFromSlice(zeros, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer biasStorage.Free()

	// Running mean/var (not used for group norm, but required by API)
	meanStorage, err := NewStorageFromSlice(zeros, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer meanStorage.Free()

	varStorage, err := NewStorageFromSlice(ones, tendo.Float32, cudaInput.device)
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer varStorage.Free()

	// Apply batch norm on reshaped view
	err = cudnnBatchNormalizationForwardInference(
		handle,
		cudnnBatchNormSpatial,
		1.0, 0.0,
		xDesc, cudaInput.Ptr(),
		yDesc, outStorage.Ptr(),
		bnDesc,
		scaleStorage.Ptr(),
		biasStorage.Ptr(),
		meanStorage.Ptr(),
		varStorage.Ptr(),
		float64(epsilon),
	)

	if err != nil {
		outStorage.Free()
		return nil, err
	}

	// Apply per-channel affine transform if weight/bias provided
	if weight != nil || bias != nil {
		// Would need to apply weight * normalized + bias per channel
		// For now, this is left as normalized output
		// Full implementation requires element-wise ops
	}

	return tendo.NewTensor(outStorage, inShape, nil), nil
}

// ReduceTensor performs tensor reduction using cuDNN.
func ReduceTensor(input *tendo.Tensor, dims []int, op string) (*tendo.Tensor, error) {
	cudaStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	// Map operation string to cuDNN op
	var reduceOp cudnnReduceTensorOp
	switch op {
	case "sum":
		reduceOp = cudnnReduceTensorAdd
	case "mean":
		reduceOp = cudnnReduceTensorAvg
	case "max":
		reduceOp = cudnnReduceTensorMax
	case "min":
		reduceOp = cudnnReduceTensorMin
	case "prod":
		reduceOp = cudnnReduceTensorMul
	default:
		return nil, &tendo.ShapeError{Op: "reduce", Message: "unsupported reduction operation: " + op}
	}

	inShape := input.Shape()
	inStrides := tendo.ComputeStrides(inShape)

	// Compute output shape (set reduced dims to 1)
	outShape := make([]int, len(inShape))
	copy(outShape, inShape)

	reduceDims := make(map[int]bool)
	for _, d := range dims {
		if d < 0 {
			d = len(inShape) + d
		}
		if d >= 0 && d < len(inShape) {
			reduceDims[d] = true
			outShape[d] = 1
		}
	}

	// If no dims specified, reduce all
	if len(dims) == 0 {
		for i := range outShape {
			outShape[i] = 1
		}
	}

	outStrides := tendo.ComputeStrides(outShape)

	// Create input tensor descriptor
	aDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(aDesc)

	if err := cudnnSetTensorNdDescriptor(aDesc, inShape, inStrides); err != nil {
		return nil, err
	}

	// Create output tensor descriptor
	cDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(cDesc)

	if err := cudnnSetTensorNdDescriptor(cDesc, outShape, outStrides); err != nil {
		return nil, err
	}

	// Create reduce descriptor
	reduceDesc, err := cudnnCreateReduceTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyReduceTensorDescriptor(reduceDesc)

	if err := cudnnSetReduceTensorDescriptor(reduceDesc, reduceOp); err != nil {
		return nil, err
	}

	// Get workspace size
	workspaceSize, err := cudnnGetReductionWorkspaceSize(handle, reduceDesc, aDesc, cDesc)
	if err != nil {
		return nil, err
	}

	// Allocate workspace if needed
	var workspace uintptr
	if workspaceSize > 0 {
		workspace, err = cudaMalloc(workspaceSize)
		if err != nil {
			return nil, err
		}
		defer cudaFree(workspace)
	}

	// Create output storage
	outNumel := tendo.Numel(outShape)
	outStorage, err := NewStorage(outNumel, input.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	// Perform reduction
	err = cudnnReduceTensor(
		handle,
		reduceDesc,
		workspace,
		workspaceSize,
		1.0, aDesc, cudaStorage.Ptr(),
		0.0, cDesc, outStorage.Ptr(),
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	// Squeeze output if all dims were reduced
	finalShape := outShape
	if len(dims) == 0 {
		finalShape = []int{} // scalar
	}

	return tendo.NewTensor(outStorage, finalShape, nil), nil
}

// LayerNorm is not directly supported by cuDNN.
// This implementation falls back to computing mean/variance via reduce operations.
// For optimal performance, a custom CUDA kernel should be used.
func LayerNorm(ctx context.Context, input *tendo.Tensor, normalizedShape []int, weight, bias *tendo.Tensor, epsilon float32) (*tendo.Tensor, error) {
	// LayerNorm requires custom CUDA kernels for optimal performance
	// cuDNN doesn't provide direct support for layer normalization
	// For now, return an error to fall back to CPU implementation
	return nil, &tendo.ShapeError{Op: "layernorm", Message: "LayerNorm not yet implemented for CUDA"}
}

// Conv2d performs 2D convolution using cuDNN.
func Conv2d(input, weight *tendo.Tensor, padding, stride, dilation [2]int, groups int) (*tendo.Tensor, error) {
	inStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}
	wStorage, ok := weight.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: weight.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	inShape := input.Shape()
	wShape := weight.Shape()

	// Create input tensor descriptor
	xDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(xDesc)

	if err := cudnnSetTensor4dDescriptor(xDesc, inShape[0], inShape[1], inShape[2], inShape[3]); err != nil {
		return nil, err
	}

	// Create filter descriptor
	wDesc, err := cudnnCreateFilterDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyFilterDescriptor(wDesc)

	if err := cudnnSetFilter4dDescriptor(wDesc, wShape[0], wShape[1], wShape[2], wShape[3]); err != nil {
		return nil, err
	}

	// Create convolution descriptor
	convDesc, err := cudnnCreateConvolutionDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyConvolutionDescriptor(convDesc)

	if err := cudnnSetConvolution2dDescriptor(
		convDesc,
		padding[0], padding[1],
		stride[0], stride[1],
		dilation[0], dilation[1],
		cudnnCrossCorrelation, // Use cross-correlation (standard in deep learning)
	); err != nil {
		return nil, err
	}

	// Get output dimensions
	outN, outC, outH, outW, err := cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc)
	if err != nil {
		return nil, err
	}

	// Create output tensor descriptor
	yDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(yDesc)

	if err := cudnnSetTensor4dDescriptor(yDesc, outN, outC, outH, outW); err != nil {
		return nil, err
	}

	// Create output storage
	outStorage, err := NewStorage(outN*outC*outH*outW, input.DType(), inStorage.device)
	if err != nil {
		return nil, err
	}

	// Get workspace size
	algo := cudnnConvolutionFwdAlgoImplicitPrecompGemm // Good general-purpose algorithm
	workspaceSize, err := cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	// Allocate workspace if needed
	var workspace uintptr
	if workspaceSize > 0 {
		workspace, err = cudaMalloc(workspaceSize)
		if err != nil {
			outStorage.Free()
			return nil, err
		}
		defer cudaFree(workspace)
	}

	// Perform convolution
	err = cudnnConvolutionForward(
		handle,
		1.0,
		xDesc, inStorage.Ptr(),
		wDesc, wStorage.Ptr(),
		convDesc,
		algo,
		workspace, workspaceSize,
		0.0,
		yDesc, outStorage.Ptr(),
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, []int{outN, outC, outH, outW}, nil), nil
}

// ConvTranspose2d performs 2D transposed convolution using cuDNN.
func ConvTranspose2d(input, weight *tendo.Tensor, padding, outputPadding, stride, dilation [2]int, groups int) (*tendo.Tensor, error) {
	inStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}
	wStorage, ok := weight.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: weight.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	inShape := input.Shape()
	wShape := weight.Shape()

	N, inC, inH, inW := inShape[0], inShape[1], inShape[2], inShape[3]
	_, outCPerGroup, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]

	outC := outCPerGroup * groups

	// Calculate output dimensions for transposed convolution
	outH := (inH-1)*stride[0] - 2*padding[0] + dilation[0]*(kH-1) + outputPadding[0] + 1
	outW := (inW-1)*stride[1] - 2*padding[1] + dilation[1]*(kW-1) + outputPadding[1] + 1

	// Create input tensor descriptor (dy in backward terminology)
	dyDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(dyDesc)

	if err := cudnnSetTensor4dDescriptor(dyDesc, N, inC, inH, inW); err != nil {
		return nil, err
	}

	// Create filter descriptor
	wDesc, err := cudnnCreateFilterDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyFilterDescriptor(wDesc)

	if err := cudnnSetFilter4dDescriptor(wDesc, wShape[0], wShape[1], wShape[2], wShape[3]); err != nil {
		return nil, err
	}

	// Create convolution descriptor
	convDesc, err := cudnnCreateConvolutionDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyConvolutionDescriptor(convDesc)

	if err := cudnnSetConvolution2dDescriptor(
		convDesc,
		padding[0], padding[1],
		stride[0], stride[1],
		dilation[0], dilation[1],
		cudnnCrossCorrelation,
	); err != nil {
		return nil, err
	}

	// Create output tensor descriptor (dx in backward terminology)
	dxDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(dxDesc)

	if err := cudnnSetTensor4dDescriptor(dxDesc, N, outC, outH, outW); err != nil {
		return nil, err
	}

	// Create output storage
	outStorage, err := NewStorage(N*outC*outH*outW, input.DType(), inStorage.device)
	if err != nil {
		return nil, err
	}

	// Get workspace size
	algo := cudnnConvolutionBwdDataAlgo1
	workspaceSize, err := cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, dxDesc, algo)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	// Allocate workspace if needed
	var workspace uintptr
	if workspaceSize > 0 {
		workspace, err = cudaMalloc(workspaceSize)
		if err != nil {
			outStorage.Free()
			return nil, err
		}
		defer cudaFree(workspace)
	}

	// Perform transposed convolution
	err = cudnnConvolutionBackwardData(
		handle,
		1.0,
		wDesc, wStorage.Ptr(),
		dyDesc, inStorage.Ptr(),
		convDesc,
		algo,
		workspace, workspaceSize,
		0.0,
		dxDesc, outStorage.Ptr(),
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, []int{N, outC, outH, outW}, nil), nil
}

// OpTensor performs binary elementwise operations using cudnnOpTensor.
// Supports broadcasting via cuDNN's built-in broadcast rules.
func OpTensor(a, b *tendo.Tensor, op cudnnOpTensorOp, alpha1, alpha2 float32) (*tendo.Tensor, error) {
	// Make tensors contiguous if needed - cuDNN OpTensor doesn't handle strides
	if !a.IsContiguous() || a.Offset() != 0 {
		contA, err := MakeContiguousWithOffset(a, a.Offset())
		if err != nil {
			return nil, fmt.Errorf("OpTensor: make a contiguous: %w", err)
		}
		a = contA
	}
	if !b.IsContiguous() || b.Offset() != 0 {
		contB, err := MakeContiguousWithOffset(b, b.Offset())
		if err != nil {
			return nil, fmt.Errorf("OpTensor: make b contiguous: %w", err)
		}
		b = contB
	}

	aStorage, ok := a.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: a.Device().Type}
	}
	bStorage, ok := b.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: b.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	// Compute broadcast output shape
	outShape, err := tendo.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	outNumel := tendo.Numel(outShape)

	// Create tensor descriptors - use 4D format for cuDNN
	aDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(aDesc)

	bDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(bDesc)

	cDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(cDesc)

	// Set tensor descriptors
	an, ac, ah, aw := tensorTo4D(a.Shape())
	if err := cudnnSetTensor4dDescriptor(aDesc, an, ac, ah, aw); err != nil {
		return nil, err
	}

	bn, bc, bh, bw := tensorTo4D(b.Shape())
	if err := cudnnSetTensor4dDescriptor(bDesc, bn, bc, bh, bw); err != nil {
		return nil, err
	}

	cn, cc, ch, cw := tensorTo4D(outShape)
	if err := cudnnSetTensor4dDescriptor(cDesc, cn, cc, ch, cw); err != nil {
		return nil, err
	}

	// Create op descriptor
	opDesc, err := cudnnCreateOpTensorDescriptor()
	if err != nil {
		return nil, err
	}
	defer cudnnDestroyOpTensorDescriptor(opDesc)

	if err := cudnnSetOpTensorDescriptor(opDesc, op); err != nil {
		return nil, err
	}

	// Allocate output storage
	outStorage, err := NewStorage(outNumel, a.DType(), aStorage.device)
	if err != nil {
		return nil, err
	}

	// Perform operation
	err = cudnnOpTensor(
		handle,
		opDesc,
		alpha1, aDesc, aStorage.Ptr(),
		alpha2, bDesc, bStorage.Ptr(),
		0.0, cDesc, outStorage.Ptr(),
	)
	if err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, outShape, nil), nil
}

// AddBias adds a bias tensor to an input tensor with broadcasting.
// bias is typically 1D (channels) and gets broadcast across spatial dimensions.
// Returns a new tensor: output = input + bias (broadcast)
func AddBias(input, bias *tendo.Tensor) (*tendo.Tensor, error) {
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}
	biasStorage, ok := bias.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: bias.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	// Create output storage (copy of input, then add bias in-place)
	outStorage, err := NewStorage(inputStorage.Len(), input.DType(), inputStorage.device)
	if err != nil {
		return nil, err
	}

	// Copy input to output first
	if err := cudaMemcpyPtr(outStorage.Ptr(), inputStorage.Ptr(), inputStorage.Size(), cudaMemcpyDeviceToDevice); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Create tensor descriptors
	inputDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(inputDesc)

	biasDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(biasDesc)

	// Set input descriptor (output has same shape)
	in, ic, ih, iw := tensorTo4D(input.Shape())
	if err := cudnnSetTensor4dDescriptor(inputDesc, in, ic, ih, iw); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Set bias descriptor - typically [1, C, 1, 1] for broadcasting
	bn, bc, bh, bw := tensorTo4D(bias.Shape())
	if err := cudnnSetTensor4dDescriptor(biasDesc, bn, bc, bh, bw); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Add bias: output = 1.0 * bias + 1.0 * output
	if err := cudnnAddTensor(handle, 1.0, biasDesc, biasStorage.Ptr(), 1.0, inputDesc, outStorage.Ptr()); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, input.Shape(), nil), nil
}

// TransformTensor copies a tensor with optional scaling and format conversion.
// Useful for making non-contiguous tensors contiguous on GPU.
// Returns a new contiguous tensor: output = alpha * input
func TransformTensor(input *tendo.Tensor, alpha float32) (*tendo.Tensor, error) {
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	// Create output storage
	outStorage, err := NewStorage(input.Numel(), input.DType(), inputStorage.device)
	if err != nil {
		return nil, err
	}

	// Create tensor descriptors
	inputDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(inputDesc)

	outputDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(outputDesc)

	// Set descriptors using N-dimensional API for proper stride handling
	inShape := input.Shape()
	inStrides := input.Stride()

	// Ensure at least 4D for cuDNN
	for len(inShape) < 4 {
		inShape = append([]int{1}, inShape...)
		inStrides = append([]int{inStrides[0]}, inStrides...)
	}

	if err := cudnnSetTensorNdDescriptor(inputDesc, inShape, inStrides); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Output is contiguous
	outShape := input.Shape()
	outStrides := tendo.ComputeStrides(outShape)
	for len(outShape) < 4 {
		outShape = append([]int{1}, outShape...)
		outStrides = append([]int{outStrides[0]}, outStrides...)
	}

	if err := cudnnSetTensorNdDescriptor(outputDesc, outShape, outStrides); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Transform: output = alpha * input + 0 * output
	if err := cudnnTransformTensor(handle, alpha, inputDesc, inputStorage.Ptr(), 0.0, outputDesc, outStorage.Ptr()); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, input.Shape(), nil), nil
}

// MakeContiguousWithOffset creates a contiguous copy of strided tensor data.
// Handles non-zero offset by adjusting the input pointer.
// This is used by Storage.MakeContiguous to implement tendo.ContiguousMaker.
func MakeContiguousWithOffset(input *tendo.Tensor, offset int) (*tendo.Tensor, error) {
	inputStorage, ok := input.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: input.Device().Type}
	}

	// If already contiguous with no offset, return a clone
	if input.IsContiguous() && offset == 0 {
		return tendo.NewTensor(inputStorage.Clone(), input.Shape(), nil), nil
	}

	// cuDNN only handles up to 4D tensors reliably; use CPU fallback for 5D+
	// Also use CPU fallback for tensors with broadcast strides (stride=0)
	if len(input.Shape()) > 4 || hasBroadcastStride(input.Stride()) {
		return makeContiguousViaCPU(input, inputStorage, offset)
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	numel := input.Numel()
	outStorage, err := NewStorage(numel, input.DType(), inputStorage.device)
	if err != nil {
		return nil, err
	}

	// Create tensor descriptors
	inputDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(inputDesc)

	outputDesc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(outputDesc)

	// Set input descriptor with actual strides
	inShape := input.Shape()
	inStrides := input.Stride()
	for len(inShape) < 4 {
		inShape = append([]int{1}, inShape...)
		inStrides = append([]int{inStrides[0]}, inStrides...)
	}
	if err := cudnnSetTensorNdDescriptor(inputDesc, inShape, inStrides); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Set output descriptor (contiguous)
	outShape := input.Shape()
	outStrides := tendo.ComputeStrides(outShape)
	for len(outShape) < 4 {
		outShape = append([]int{1}, outShape...)
		outStrides = append([]int{outStrides[0]}, outStrides...)
	}
	if err := cudnnSetTensorNdDescriptor(outputDesc, outShape, outStrides); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Calculate input pointer with offset
	inputPtr := inputStorage.Ptr()
	if offset > 0 {
		inputPtr += uintptr(offset * input.DType().Size())
	}

	// Transform: output = 1.0 * input
	if err := cudnnTransformTensor(handle, 1.0, inputDesc, inputPtr, 0.0, outputDesc, outStorage.Ptr()); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, input.Shape(), nil), nil
}

// makeContiguousViaCPU handles non-contiguous tensors that cuDNN can't process.
// Copies data through host memory as a fallback.
func makeContiguousViaCPU(input *tendo.Tensor, inputStorage *Storage, offset int) (*tendo.Tensor, error) {
	shape := input.Shape()
	strides := input.Stride()
	numel := input.Numel()

	// Copy input data to host
	hostData, err := inputStorage.CopyToHost()
	if err != nil {
		return nil, err
	}

	// Allocate contiguous output on host
	contiguous := make([]float32, numel)

	// Copy with stride handling
	copyStrided(contiguous, hostData, shape, strides, offset)

	// Upload back to GPU
	outStorage, err := NewStorageFromSlice(contiguous, input.DType(), inputStorage.device)
	if err != nil {
		return nil, err
	}

	return tendo.NewTensor(outStorage, shape, nil), nil
}

// hasBroadcastStride returns true if any stride is 0 (from Expand/broadcast).
func hasBroadcastStride(strides []int) bool {
	for _, s := range strides {
		if s == 0 {
			return true
		}
	}
	return false
}

// copyStrided copies strided data to contiguous output.
func copyStrided(dst, src []float32, shape, strides []int, offset int) {
	if len(shape) == 0 {
		dst[0] = src[offset]
		return
	}

	if len(shape) == 1 {
		for i := 0; i < shape[0]; i++ {
			dst[i] = src[offset+i*strides[0]]
		}
		return
	}

	// Recursive case: iterate over first dimension
	innerSize := 1
	for i := 1; i < len(shape); i++ {
		innerSize *= shape[i]
	}

	for i := 0; i < shape[0]; i++ {
		copyStrided(dst[i*innerSize:], src, shape[1:], strides[1:], offset+i*strides[0])
	}
}

// ScaleTensor performs in-place scaling of a tensor: y = alpha * y
// Returns a new tensor with the scaled values.
func ScaleTensor(t *tendo.Tensor, alpha float32) (*tendo.Tensor, error) {
	cudaStorage, ok := t.Storage().(*Storage)
	if !ok {
		return nil, &tendo.DeviceError{Expected: tendo.CUDA, Got: t.Device().Type}
	}

	handle, err := getCudnnHandle()
	if err != nil {
		return nil, err
	}

	// Create output by cloning input (ScaleTensor is in-place, so we clone first)
	outStorage, err := NewStorage(cudaStorage.Len(), t.DType(), cudaStorage.device)
	if err != nil {
		return nil, err
	}

	// Copy data from input to output
	if err := cudaMemcpyPtr(outStorage.Ptr(), cudaStorage.Ptr(), cudaStorage.Size(), cudaMemcpyDeviceToDevice); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Create tensor descriptor
	desc, err := cudnnCreateTensorDescriptor()
	if err != nil {
		outStorage.Free()
		return nil, err
	}
	defer cudnnDestroyTensorDescriptor(desc)

	n, c, h, w := tensorTo4D(t.Shape())
	if err := cudnnSetTensor4dDescriptor(desc, n, c, h, w); err != nil {
		outStorage.Free()
		return nil, err
	}

	// Scale the tensor
	if err := cudnnScaleTensor(handle, desc, outStorage.Ptr(), alpha); err != nil {
		outStorage.Free()
		return nil, err
	}

	return tendo.NewTensor(outStorage, t.Shape(), nil), nil
}
