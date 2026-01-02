# tendo

A composable tensor library for Go with GPU acceleration.

Tendo provides building blocks for neural network operations through a pipeline-based architecture, integrating with [pipz](https://github.com/zoobzio/pipz) for operation composition and [capitan](https://github.com/zoobzio/capitan) for observability.

## Features

- **Multi-dimensional tensors** with shape and stride management
- **Device support** for CPU and NVIDIA CUDA GPUs
- **Multiple data types**: Float32, Float16, BFloat16
- **Rich operation set**:
  - Elementwise: Add, Sub, Mul, Div, Abs, Exp, Log, Sqrt, Pow
  - Matrix: MatMul, Transpose, batched operations
  - Shape: Reshape, Squeeze, Unsqueeze, Slice, Expand, Permute
  - Reductions: Sum, Mean, Max, Min, ArgMax, ArgMin
  - Activations: ReLU, Sigmoid, Tanh, GELU, SiLU, Softmax, Dropout
  - Convolution: Conv2d with grouped convolution support
- **Memory pooling** for efficient device memory reuse
- **Chainable operations** via pipz for composable computation graphs

## Requirements

- Go 1.21+
- For CUDA support: NVIDIA CUDA Toolkit with cuBLAS

## Installation

```bash
go get github.com/zoobzio/tendo
```

## Usage

### Creating Tensors

```go
package main

import "github.com/zoobzio/tendo"

func main() {
    // From slice
    t := tendo.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

    // Zeros, ones, random
    zeros := tendo.Zeros(3, 4)
    ones := tendo.Ones(2, 2)
    rand := tendo.RandN(10, 10)

    // Sequences
    seq := tendo.Arange(0, 10, 1)
    lin := tendo.Linspace(0, 1, 100)

    // Identity matrix
    eye := tendo.Eye(4)
}
```

### Operations

```go
// Elementwise
c := tendo.Add(a, b)
d := tendo.Mul(a, b)

// Matrix multiplication
result := tendo.MatMul(a, b)

// Reductions
sum := tendo.Sum(t, -1)      // sum along last dimension
mean := tendo.Mean(t, 0)     // mean along first dimension

// Activations
activated := tendo.ReLU(t)
probs := tendo.Softmax(logits, -1)

// Shape operations
reshaped := tendo.Reshape(t, 6, 1)
squeezed := tendo.Squeeze(t, 0)
```

### Device Management

```go
// Set default device
tendo.SetDefaultDevice(tendo.CUDADevice(0))

// Create on specific device
t := tendo.ZerosOn(tendo.CUDADevice(0), 1024, 1024)

// Transfer between devices
cpu := tendo.ToCPU(t)
gpu := tendo.ToCUDA(t, 0)
```

### Memory Pooling

```go
pool := tendo.NewPool()

// Allocate from pool
storage := pool.AllocCPU(1000, tendo.Float32)

// Return to pool for reuse
pool.FreeCPU(storage)

// Check statistics
stats := pool.Stats()
```

### Pipeline Composition

Operations return `pipz.Chainable` for composable computation graphs:

```go
import "github.com/zoobzio/pipz"

// Chain operations
result := pipz.Chain(
    tendo.MatMulOp(weights),
    tendo.AddOp(bias),
    tendo.ReLUOp(),
)
```

## Architecture

```
tendo/
├── storage.go          # Storage interface, Device, DType definitions
├── storage_cpu.go      # CPU memory backend
├── storage_cuda.go     # CUDA memory backend
├── tensor.go           # Tensor type with shape/stride
├── constructors.go     # Tensor creation functions
├── ops_*.go            # Operation implementations
├── pool.go             # Memory pooling
├── signals.go          # Observability events
└── cuda_*.go           # CUDA runtime bindings
```

## License

MIT
