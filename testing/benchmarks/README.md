# Benchmarks

Performance benchmarks for tendo tensor operations.

## Running Benchmarks

```bash
# Run all benchmarks
make bench

# Run specific benchmark
go test -bench=BenchmarkTensorCreation -benchmem ./testing/benchmarks/...

# Run with longer duration
go test -bench=. -benchtime=5s -benchmem ./testing/benchmarks/...
```

## Benchmark Categories

### Tensor Creation

Measures allocation and initialization overhead for tensor constructors:
- `Zeros_*`: Zero-filled tensor creation
- `RandN_*`: Random normal tensor creation

### Tensor Operations

Measures core tensor operation performance:
- `Clone`: Deep copy performance
- `Contiguous`: Memory layout normalization

### Memory Pool

Measures memory pooling efficiency:
- `Alloc_Free_CPU`: Full allocation/deallocation cycle
- `Alloc_Only`: Allocation without return to pool

## Interpreting Results

```
BenchmarkTensorCreation/Zeros_Small-8    1000000    1024 ns/op    4096 B/op    2 allocs/op
```

- `1000000`: Number of iterations
- `1024 ns/op`: Nanoseconds per operation
- `4096 B/op`: Bytes allocated per operation
- `2 allocs/op`: Heap allocations per operation

## Performance Guidelines

- Prefer pooled allocations for repeated operations
- Use contiguous tensors when possible
- Batch small operations to amortize overhead
