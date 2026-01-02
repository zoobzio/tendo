package tendo

import (
	"github.com/zoobzio/capitan"
)

// Custom variants for tendo types.
const (
	VariantTensor     capitan.Variant = "tendo.Tensor"
	VariantTensorList capitan.Variant = "tendo.TensorList"
	VariantShape      capitan.Variant = "tendo.Shape"
	VariantDevice     capitan.Variant = "tendo.Device"
	VariantIntPair    capitan.Variant = "tendo.IntPair"
	VariantUintptr    capitan.Variant = "tendo.Uintptr"
)

// Lifecycle signals.
var (
	// TensorCreated is emitted when a tensor is allocated.
	TensorCreated = capitan.NewSignal("tendo.tensor.created", "Tensor allocated")

	// TensorFreed is emitted when a tensor is deallocated.
	TensorFreed = capitan.NewSignal("tendo.tensor.freed", "Tensor deallocated")

	// TensorTransfer is emitted when a tensor is moved between devices.
	TensorTransfer = capitan.NewSignal("tendo.tensor.transfer", "Tensor moved between devices")
)

// Operation signals - emitted during forward pass for autograd capture.
var (
	// Element-wise operations.
	OpAdd    = capitan.NewSignal("tendo.op.add", "Element-wise addition")
	OpSub    = capitan.NewSignal("tendo.op.sub", "Element-wise subtraction")
	OpMul    = capitan.NewSignal("tendo.op.mul", "Element-wise multiplication")
	OpDiv    = capitan.NewSignal("tendo.op.div", "Element-wise division")
	OpNeg    = capitan.NewSignal("tendo.op.neg", "Negation")
	OpAbs    = capitan.NewSignal("tendo.op.abs", "Absolute value")
	OpExp    = capitan.NewSignal("tendo.op.exp", "Exponential")
	OpLog    = capitan.NewSignal("tendo.op.log", "Natural logarithm")
	OpSqrt   = capitan.NewSignal("tendo.op.sqrt", "Square root")
	OpSquare = capitan.NewSignal("tendo.op.square", "Square")
	OpPow    = capitan.NewSignal("tendo.op.pow", "Power")
	OpSin    = capitan.NewSignal("tendo.op.sin", "Sine")
	OpCos    = capitan.NewSignal("tendo.op.cos", "Cosine")

	// Matrix operations.
	OpMatMul    = capitan.NewSignal("tendo.op.matmul", "Matrix multiplication")
	OpTranspose = capitan.NewSignal("tendo.op.transpose", "Transpose")

	// Shape operations.
	OpReshape   = capitan.NewSignal("tendo.op.reshape", "Reshape")
	OpSqueeze   = capitan.NewSignal("tendo.op.squeeze", "Squeeze dimension")
	OpUnsqueeze = capitan.NewSignal("tendo.op.unsqueeze", "Unsqueeze dimension")
	OpSlice     = capitan.NewSignal("tendo.op.slice", "Slice")
	OpExpand    = capitan.NewSignal("tendo.op.expand", "Expand/broadcast")
	OpPermute   = capitan.NewSignal("tendo.op.permute", "Dimension permutation")
	OpCat       = capitan.NewSignal("tendo.op.cat", "Concatenate")
	OpStack     = capitan.NewSignal("tendo.op.stack", "Stack")

	// Reduction operations.
	OpSum    = capitan.NewSignal("tendo.op.sum", "Sum reduction")
	OpMean   = capitan.NewSignal("tendo.op.mean", "Mean reduction")
	OpMax    = capitan.NewSignal("tendo.op.max", "Max reduction")
	OpMin    = capitan.NewSignal("tendo.op.min", "Min reduction")
	OpArgMax = capitan.NewSignal("tendo.op.argmax", "Argmax")
	OpArgMin = capitan.NewSignal("tendo.op.argmin", "Argmin")
	OpVar    = capitan.NewSignal("tendo.op.var", "Variance reduction")
	OpStd    = capitan.NewSignal("tendo.op.std", "Standard deviation reduction")
	OpProd   = capitan.NewSignal("tendo.op.prod", "Product reduction")

	// Activation operations.
	OpReLU       = capitan.NewSignal("tendo.op.relu", "ReLU activation")
	OpLeakyReLU  = capitan.NewSignal("tendo.op.leaky_relu", "Leaky ReLU activation")
	OpSigmoid    = capitan.NewSignal("tendo.op.sigmoid", "Sigmoid activation")
	OpTanh       = capitan.NewSignal("tendo.op.tanh", "Tanh activation")
	OpGELU       = capitan.NewSignal("tendo.op.gelu", "GELU activation")
	OpSiLU       = capitan.NewSignal("tendo.op.silu", "SiLU/Swish activation")
	OpSoftmax    = capitan.NewSignal("tendo.op.softmax", "Softmax")
	OpLogSoftmax = capitan.NewSignal("tendo.op.logsoftmax", "Log softmax")
	OpDropout    = capitan.NewSignal("tendo.op.dropout", "Dropout")

	// Pooling operations.
	OpMaxPool2d         = capitan.NewSignal("tendo.op.maxpool2d", "2D max pooling")
	OpAvgPool2d         = capitan.NewSignal("tendo.op.avgpool2d", "2D average pooling")
	OpAdaptiveAvgPool2d = capitan.NewSignal("tendo.op.adaptiveavgpool2d", "Adaptive 2D average pooling")
	OpAdaptiveMaxPool2d = capitan.NewSignal("tendo.op.adaptivemaxpool2d", "Adaptive 2D max pooling")

	// Normalization operations.
	OpBatchNorm    = capitan.NewSignal("tendo.op.batchnorm", "Batch normalization")
	OpLayerNorm    = capitan.NewSignal("tendo.op.layernorm", "Layer normalization")
	OpRMSNorm      = capitan.NewSignal("tendo.op.rmsnorm", "RMS normalization")
	OpGroupNorm    = capitan.NewSignal("tendo.op.groupnorm", "Group normalization")
	OpInstanceNorm = capitan.NewSignal("tendo.op.instancenorm", "Instance normalization")

	// Additional element-wise operations.
	OpClamp = capitan.NewSignal("tendo.op.clamp", "Clamp values")
	OpWhere = capitan.NewSignal("tendo.op.where", "Conditional select")
	OpSign  = capitan.NewSignal("tendo.op.sign", "Sign function")
	OpTril  = capitan.NewSignal("tendo.op.tril", "Lower triangular")

	// Convolution operations.
	OpConvTranspose2d = capitan.NewSignal("tendo.op.convtranspose2d", "2D transposed convolution")

	// Embedding operations.
	OpEmbedding = capitan.NewSignal("tendo.op.embedding", "Embedding lookup")

	// Loss functions.
	OpMSELoss          = capitan.NewSignal("tendo.op.mse_loss", "Mean squared error loss")
	OpL1Loss           = capitan.NewSignal("tendo.op.l1_loss", "L1 loss")
	OpCrossEntropyLoss = capitan.NewSignal("tendo.op.cross_entropy", "Cross entropy loss")
	OpNLLLoss          = capitan.NewSignal("tendo.op.nll_loss", "Negative log likelihood loss")
)

// Memory signals.
var (
	// PoolAlloc is emitted when memory is allocated from the pool.
	PoolAlloc = capitan.NewSignal("tendo.pool.alloc", "Memory allocated from pool")

	// PoolFree is emitted when memory is returned to the pool.
	PoolFree = capitan.NewSignal("tendo.pool.free", "Memory returned to pool")
)

// Typed keys for event fields.
var (
	// KeyInput is the primary input tensor for unary operations.
	KeyInput = capitan.NewKey[*Tensor]("input", VariantTensor)

	// KeyInputA is the first input for binary operations.
	KeyInputA = capitan.NewKey[*Tensor]("input_a", VariantTensor)

	// KeyInputB is the second input for binary operations.
	KeyInputB = capitan.NewKey[*Tensor]("input_b", VariantTensor)

	// KeyInputs is a list of input tensors for n-ary operations.
	KeyInputs = capitan.NewKey[[]*Tensor]("inputs", VariantTensorList)

	// KeyOutput is the output tensor.
	KeyOutput = capitan.NewKey[*Tensor]("output", VariantTensor)

	// KeyShape is the shape of a tensor.
	KeyShape = capitan.NewKey[[]int]("shape", VariantShape)

	// KeyDevice is the device of a tensor.
	KeyDevice = capitan.NewKey[Device]("device", VariantDevice)

	// KeyFromDevice is the source device for transfers.
	KeyFromDevice = capitan.NewKey[Device]("from_device", VariantDevice)

	// KeyToDevice is the destination device for transfers.
	KeyToDevice = capitan.NewKey[Device]("to_device", VariantDevice)

	// KeyDim is a dimension index.
	KeyDim = capitan.NewIntKey("dim")

	// KeyDims is multiple dimension indices.
	KeyDims = capitan.NewKey[[]int]("dims", VariantShape)

	// KeyBytes is a byte count.
	KeyBytes = capitan.NewIntKey("bytes")

	// KeyScalar is a scalar value.
	KeyScalar = capitan.NewFloat32Key("scalar")

	// KeyOpName is the operation name.
	KeyOpName = capitan.NewStringKey("op_name")

	// KeyTraceID is the pipeline trace identifier for autograd graph correlation.
	KeyTraceID = capitan.NewStringKey("trace_id")

	// KeyMask is the dropout mask tensor.
	KeyMask = capitan.NewKey[*Tensor]("mask", VariantTensor)

	// KeyReserveSpace is a handle to cuDNN's internal reserve space (for GPU dropout).
	KeyReserveSpace = capitan.NewKey[uintptr]("reserve_space", VariantUintptr)

	// KeyStart is the start index for slice operations.
	KeyStart = capitan.NewIntKey("start")

	// KeyEnd is the end index for slice operations.
	KeyEnd = capitan.NewIntKey("end")

	// KeyDim0 is the first dimension for transpose operations.
	KeyDim0 = capitan.NewIntKey("dim0")

	// KeyDim1 is the second dimension for transpose operations.
	KeyDim1 = capitan.NewIntKey("dim1")

	// KeyPermutation is the dimension ordering for permute operations.
	KeyPermutation = capitan.NewKey[[]int]("permutation", VariantShape)

	// KeyPadding is the padding for convolution operations.
	KeyPadding = capitan.NewKey[[2]int]("padding", VariantIntPair)

	// KeyConvStride is the stride for convolution operations.
	KeyConvStride = capitan.NewKey[[2]int]("conv_stride", VariantIntPair)

	// KeyDilation is the dilation for convolution operations.
	KeyDilation = capitan.NewKey[[2]int]("dilation", VariantIntPair)

	// KeyGroups is the number of groups for grouped convolution.
	KeyGroups = capitan.NewIntKey("groups")

	// KeyKernelSize is the kernel size for pooling operations.
	KeyKernelSize = capitan.NewKey[[2]int]("kernel_size", VariantIntPair)

	// KeyPoolStride is the stride for pooling operations.
	KeyPoolStride = capitan.NewKey[[2]int]("pool_stride", VariantIntPair)

	// KeyIndices is the max indices tensor for max pooling (for backward pass).
	KeyIndices = capitan.NewKey[*Tensor]("indices", VariantTensor)

	// KeyEpsilon is the epsilon value for numerical stability in normalization.
	KeyEpsilon = capitan.NewFloat32Key("epsilon")

	// KeyMomentum is the momentum for running stats in batch normalization.
	KeyMomentum = capitan.NewFloat32Key("momentum")

	// KeyNormalizedShape is the shape for layer normalization.
	KeyNormalizedShape = capitan.NewKey[[]int]("normalized_shape", VariantShape)

	// KeyWeight is the weight parameter tensor.
	KeyWeight = capitan.NewKey[*Tensor]("weight", VariantTensor)

	// KeyBias is the bias parameter tensor.
	KeyBias = capitan.NewKey[*Tensor]("bias", VariantTensor)

	// KeyRunningMean is the running mean for batch normalization.
	KeyRunningMean = capitan.NewKey[*Tensor]("running_mean", VariantTensor)

	// KeyRunningVar is the running variance for batch normalization.
	KeyRunningVar = capitan.NewKey[*Tensor]("running_var", VariantTensor)

	// KeyMin is the minimum value for clamp operations.
	KeyMin = capitan.NewFloat32Key("min")

	// KeyMax is the maximum value for clamp operations.
	KeyMax = capitan.NewFloat32Key("max")

	// KeyCondition is the condition tensor for where operations.
	KeyCondition = capitan.NewKey[*Tensor]("condition", VariantTensor)

	// KeyOther is the alternative tensor for where operations.
	KeyOther = capitan.NewKey[*Tensor]("other", VariantTensor)

	// KeyTarget is the target tensor for loss functions.
	KeyTarget = capitan.NewKey[*Tensor]("target", VariantTensor)

	// KeyReduction is the reduction mode for loss functions ("none", "mean", "sum").
	KeyReduction = capitan.NewStringKey("reduction")

	// KeyK is the diagonal offset for tril operations.
	KeyK = capitan.NewIntKey("k")

	// KeyNumGroups is the number of groups for group normalization.
	KeyNumGroups = capitan.NewIntKey("num_groups")

	// KeyOutputSize is the target output size for adaptive pooling.
	KeyOutputSize = capitan.NewKey[[2]int]("output_size", VariantIntPair)

	// KeyOutputPadding is the output padding for transposed convolution.
	KeyOutputPadding = capitan.NewKey[[2]int]("output_padding", VariantIntPair)
)
