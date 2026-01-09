# Integration Tests

End-to-end integration tests for tendo.

## Purpose

Integration tests verify that components work together correctly:
- CPU and CUDA backends produce consistent results
- Memory pooling works across operations
- Pipelines execute correctly
- Device transfers maintain data integrity

## Running Integration Tests

```bash
# Run integration tests
make test-integration

# Run with verbose output
go test -v ./testing/integration/...

# Run specific test
go test -v -run TestPipelineIntegration ./testing/integration/...
```

## Test Categories

### Backend Consistency

Tests that CPU and CUDA backends produce equivalent results for the same operations.

### Memory Management

Tests that verify:
- Pool allocations are reused correctly
- No memory leaks across operation sequences
- Proper cleanup on tensor Free()

### Pipeline Execution

Tests that verify operation chaining via pipz works correctly.

### Device Transfer

Tests that verify data integrity when moving tensors between CPU and CUDA.

## Writing Integration Tests

```go
//go:build integration

package integration

import (
    "context"
    "testing"

    "github.com/zoobzio/tendo"
    ttesting "github.com/zoobzio/tendo/testing"
)

func TestOperationSequence(t *testing.T) {
    ctx := context.Background()

    // Create input
    input := tendo.RandN(32, 64)

    // Run operation sequence
    result, err := tendo.MatMul(input, weights)
    if err != nil {
        t.Fatalf("MatMul failed: %v", err)
    }

    result, err = tendo.ReLU(result)
    if err != nil {
        t.Fatalf("ReLU failed: %v", err)
    }

    // Verify output
    ttesting.AssertTensorShape(t, result, 32, 128)
}
```

## Build Tags

Integration tests use the `integration` build tag:

```go
//go:build integration
```

This allows them to be run separately from unit tests.
