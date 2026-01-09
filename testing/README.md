# Testing

Testing infrastructure for tendo.

## Directory Structure

```
testing/
├── README.md              # This file
├── helpers.go             # Test utilities and assertions
├── helpers_test.go        # Tests for helper functions
├── benchmarks/            # Performance benchmarks
│   ├── README.md
│   └── tensor_bench_test.go
└── integration/           # Integration tests
    ├── README.md
    └── helpers_test.go
```

## Running Tests

```bash
# Run all tests
make test

# Run unit tests only (short mode)
make test-unit

# Run integration tests
make test-integration

# Run benchmarks
make bench

# Generate coverage report
make coverage
```

## Test Helpers

The `testing` package provides utilities for tensor testing:

### Assertions

```go
import ttesting "github.com/zoobzio/tendo/testing"

func TestMyOperation(t *testing.T) {
    result := myOperation(input)
    expected := tendo.MustFromSlice([]float32{1, 2, 3}, 3)

    // Compare tensors with tolerance
    ttesting.AssertTensorEqual(t, expected, result, 1e-6)

    // Check shape only
    ttesting.AssertTensorShape(t, result, 3)

    // Check specific values
    ttesting.AssertTensorValues(t, result, []float32{1, 2, 3}, 1e-6)
}
```

### Test Data Generation

```go
// Sequential values
t := ttesting.Range(10)           // [0, 1, 2, ..., 9]
t := ttesting.RangeWithShape(2, 3) // [[0, 1, 2], [3, 4, 5]]

// Constant values
t := ttesting.Constant(3.14, 2, 2)
```

### Comparison Utilities

```go
// Check if values are close
ok := ttesting.ApproxEqual(1.0, 1.0001, 1e-3)

// Check relative tolerance
ok := ttesting.ApproxEqualRel(100.0, 100.5, 0.01)

// Maximum difference between tensors
diff := ttesting.MaxAbsDiff(a, b)

// All values within tolerance
ok := ttesting.AllClose(a, b, 1e-6)
```

## Writing Tests

### Conventions

1. Test files mirror source files: `foo.go` → `foo_test.go`
2. Use table-driven tests for multiple cases
3. Use `t.Helper()` in helper functions
4. Use descriptive subtest names

### Example

```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name     string
        a, b     []float32
        expected []float32
    }{
        {"positive", []float32{1, 2}, []float32{3, 4}, []float32{4, 6}},
        {"negative", []float32{-1, -2}, []float32{1, 2}, []float32{0, 0}},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            a := tendo.MustFromSlice(tt.a, len(tt.a))
            b := tendo.MustFromSlice(tt.b, len(tt.b))
            result, err := tendo.Add(a, b)
            if err != nil {
                t.Fatalf("Add failed: %v", err)
            }
            ttesting.AssertTensorValues(t, result, tt.expected, 1e-6)
        })
    }
}
```

## Coverage

Target coverage: 70% overall, 80% for new code.

```bash
# Generate HTML coverage report
make coverage

# View coverage in browser
open coverage.html
```
