# tendo

[![CI Status](https://github.com/zoobzio/tendo/workflows/CI/badge.svg)](https://github.com/zoobzio/tendo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/zoobzio/tendo/graph/badge.svg?branch=main)](https://codecov.io/gh/zoobzio/tendo)
[![Go Report Card](https://goreportcard.com/badge/github.com/zoobzio/tendo)](https://goreportcard.com/report/github.com/zoobzio/tendo)
[![CodeQL](https://github.com/zoobzio/tendo/workflows/CodeQL/badge.svg)](https://github.com/zoobzio/tendo/security/code-scanning)
[![Go Reference](https://pkg.go.dev/badge/github.com/zoobzio/tendo.svg)](https://pkg.go.dev/github.com/zoobzio/tendo)
[![License](https://img.shields.io/github/license/zoobzio/tendo)](LICENSE)
[![Go Version](https://img.shields.io/github/go-mod/go-version/zoobzio/tendo)](go.mod)
[![Release](https://img.shields.io/github/v/release/zoobzio/tendo)](https://github.com/zoobzio/tendo/releases)

A composable tensor library for Go with GPU acceleration.

Tendo provides building blocks for neural network operations through a pipeline-based architecture, integrating with [pipz](https://github.com/zoobzio/pipz) for operation composition and [capitan](https://github.com/zoobzio/capitan) for observability.

## Inference in Native Go

Load weights, run forward passes, get predictions — all in Go, with optional GPU acceleration:

```go
backend := cuda.NewBackend(0) // or cpu.NewBackend()

// Load pretrained weights
weights := tendo.MustLoad("model.safetensors")

// Build the forward pass
linear := tendo.NewMatMul(backend, weights.Get("fc.weight"))
bias := tendo.NewAdd(backend, weights.Get("fc.bias"))
activation := tendo.NewGELU(backend)

// Run inference
output, err := pipz.NewSequence(ForwardID, linear, bias, activation).Process(ctx, input)
```

No Python runtime. No CGO for CPU operations. One binary, native error handling, production-ready.

## Installation

```bash
go get github.com/zoobzio/tendo
```

Requires Go 1.24+. For CUDA support: NVIDIA CUDA Toolkit with cuBLAS

## Quick Start

```go
package main

import (
    "context"
    "fmt"

    "github.com/zoobzio/pipz"
    "github.com/zoobzio/tendo"
    "github.com/zoobzio/tendo/cpu"
)

var (
    LinearID = pipz.NewIdentity("linear", "Linear transformation")
)

func main() {
    ctx := context.Background()
    backend := cpu.NewBackend()

    // Create tensors
    input := tendo.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
    weights := tendo.MustRandN(3, 4)
    bias := tendo.MustZeros(4)

    // Build operations
    matmul := tendo.NewMatMul(backend, weights)
    add := tendo.NewAdd(backend, bias)
    relu := tendo.NewReLU(backend)

    // Compose into a pipeline
    forward := pipz.NewSequence(LinearID, matmul, add, relu)

    // Execute
    output, err := forward.Process(ctx, input)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Input:  %v\n", input.Shape())  // [2, 3]
    fmt.Printf("Output: %v\n", output.Shape()) // [2, 4]
}
```

## Why tendo?

- **Native Go**: No CGO required for CPU operations. Clean interfaces, explicit error handling.
- **Composable**: Operations are pipz.Chainable. Build computation graphs that compose and reuse.
- **Observable**: Every operation emits capitan signals. Hook listeners for logging, profiling, debugging.
- **Device portable**: Write operations once, run on CPU or CUDA. Transfer tensors with a single call.
- **Memory efficient**: Pool allocations to reduce GC pressure and CUDA malloc overhead.

## Capabilities

| Feature         | Description                                             | Docs                                           |
| --------------- | ------------------------------------------------------- | ---------------------------------------------- |
| Tensor Creation | Zeros, ones, random, ranges, identity matrices          | [Quickstart](docs/2.learn/1.quickstart.md)     |
| Elementwise Ops | Add, Sub, Mul, Div, Exp, Log, Sqrt, Pow, trig functions | [Operations](docs/5.reference/2.operations.md) |
| Matrix Ops      | MatMul, Transpose, batched multiplication               | [Operations](docs/5.reference/2.operations.md) |
| Activations     | ReLU, Sigmoid, Tanh, GELU, SiLU, Softmax, Dropout       | [Operations](docs/5.reference/2.operations.md) |
| Reductions      | Sum, Mean, Max, Min, Var, Std, ArgMax, ArgMin           | [Operations](docs/5.reference/2.operations.md) |
| Shape Ops       | Reshape, Squeeze, Unsqueeze, Slice, Permute, Cat, Stack | [Operations](docs/5.reference/2.operations.md) |
| Convolution     | Conv2d, ConvTranspose2d with grouped convolution        | [Operations](docs/5.reference/2.operations.md) |
| Pooling         | MaxPool2d, AvgPool2d, Adaptive variants                 | [Operations](docs/5.reference/2.operations.md) |
| Normalization   | BatchNorm, LayerNorm, RMSNorm, GroupNorm                | [Operations](docs/5.reference/2.operations.md) |
| Device Support  | CPU and NVIDIA CUDA with seamless transfer              | [Devices](docs/3.guides/1.devices.md)          |
| Memory Pooling  | Allocation reuse to reduce GC and malloc overhead       | [Memory](docs/3.guides/2.memory.md)            |
| Data Types      | Float32, Float16, BFloat16                              | [Data Types](docs/3.guides/3.dtypes.md)        |

## From Training to Production

Train in PyTorch, deploy in Go. Tendo bridges the gap:

```go
// Load exported weights
weights := tendo.MustLoad("transformer.safetensors")

// Build transformer blocks from tendo primitives
attention := NewMultiHeadAttention(backend, weights, "layer.0.attn")
ffn := NewFeedForward(backend, weights, "layer.0.ffn")
norm := tendo.NewLayerNorm(backend, weights.Get("layer.0.norm"))

// Compose into a model
layer := pipz.NewSequence(LayerID, attention, norm, ffn)

// Inference at scale — pure Go, GPU-accelerated
for batch := range batches {
    output, _ := layer.Process(ctx, batch)
    results <- output
}
```

Your inference service is a Go binary. Standard deployment, standard tooling, standard observability — with GPU performance where it matters.

## Documentation

- [Overview](docs/1.overview.md) - Package overview and architecture
- **Learn**
  - [Quickstart](docs/2.learn/1.quickstart.md) - Getting started
  - [Concepts](docs/2.learn/2.concepts.md) - Core concepts
  - [Architecture](docs/2.learn/3.architecture.md) - System design
- **Guides**
  - [Devices](docs/3.guides/1.devices.md) - CPU and CUDA device management
  - [Memory](docs/3.guides/2.memory.md) - Memory pooling and efficiency
  - [Data Types](docs/3.guides/3.dtypes.md) - Float32, Float16, BFloat16
  - [Testing](docs/3.guides/4.testing.md) - Testing tensor operations
- **Cookbook**
  - [Pipelines](docs/4.cookbook/1.pipelines.md) - Building computation pipelines
  - [Observability](docs/4.cookbook/2.observability.md) - Logging and profiling
- **Reference**
  - [API](docs/5.reference/1.api.md) - Function documentation
  - [Operations](docs/5.reference/2.operations.md) - Operation reference

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and guidelines.

```bash
# Quick start
make help          # Show available commands
make install-tools # Install development tools
make check         # Run tests and lint
```

## License

MIT
