package integration

import (
	"context"
	"testing"

	"github.com/zoobzio/tendo"
	tt "github.com/zoobzio/tendo/testing"
)

// -----------------------------------------------------------------------------
// Shape Transformation Pipelines
// -----------------------------------------------------------------------------

// TestReshapePermutePipeline tests shape transformations that affect memory layout.
func TestReshapePermutePipeline(t *testing.T) {
	// [0,1,2,3,4,5] shaped as [2,3]
	// [[0,1,2],
	//  [3,4,5]]
	input := tt.RangeWithShape(2, 3)

	pipeline := tendo.Sequence("transform",
		tendo.NewPermute(1, 0),    // [3,2] - transpose, now non-contiguous
		tendo.NewReshape(6),       // [6] - forces contiguous copy
		tendo.NewReshape(3, 2),    // [3,2] - new shape
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 3, 2)

	// After transpose: [[0,3],[1,4],[2,5]]
	// After flatten: [0,3,1,4,2,5]
	// After reshape to [3,2]: [[0,3],[1,4],[2,5]]
	tt.AssertTensorValues(t, result, []float32{0, 3, 1, 4, 2, 5}, 1e-6)
}

// TestTransposeContiguousPipeline verifies contiguous() is called when needed.
func TestTransposeContiguousPipeline(t *testing.T) {
	input := tt.RangeWithShape(2, 3)

	pipeline := tendo.Sequence("transpose-contiguous",
		tendo.NewPermute(1, 0), // Now non-contiguous
		tendo.MakeContiguous(), // Force contiguous copy
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	if !result.IsContiguous() {
		t.Error("expected contiguous tensor after MakeContiguous")
	}

	tt.AssertTensorShape(t, result, 3, 2)
	// Transposed data: [[0,3],[1,4],[2,5]]
	tt.AssertTensorValues(t, result, []float32{0, 3, 1, 4, 2, 5}, 1e-6)
}

// TestFlattenNonContiguous tests flatten on a non-contiguous tensor.
func TestFlattenNonContiguous(t *testing.T) {
	input := tt.RangeWithShape(2, 3, 4) // 24 elements

	pipeline := tendo.Sequence("permute-flatten",
		tendo.NewPermute(2, 0, 1),  // [4,2,3] - non-contiguous
		tendo.NewFlatten(0, -1),    // Flatten all dims - forces contiguous
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 24)
	if !result.IsContiguous() {
		t.Error("expected contiguous after flatten")
	}
}

// -----------------------------------------------------------------------------
// Slicing and View Semantics
// -----------------------------------------------------------------------------

// TestSlicePipeline tests slicing operations in a pipeline.
func TestSlicePipeline(t *testing.T) {
	// [0,1,2,3,4,5,6,7,8,9] shaped as [2,5]
	input := tt.RangeWithShape(2, 5)

	pipeline := tendo.Sequence("slice-pipeline",
		tendo.NewSlice(1, 1, 4), // Take columns 1,2,3 -> [2,3]
		tendo.NewSlice(0, 0, 1), // Take first row -> [1,3]
		tendo.NewSqueeze(),      // Remove size-1 dim -> [3]
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 3)
	// Row 0, columns 1,2,3 = [1,2,3]
	tt.AssertTensorValues(t, result, []float32{1, 2, 3}, 1e-6)
}

// TestSliceSharesStorage verifies slice creates a view, not a copy.
func TestSliceSharesStorage(t *testing.T) {
	input := tt.RangeWithShape(10)

	slice := tendo.NewSlice(0, 2, 5)
	result, err := slice.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("slice failed: %v", err)
	}

	// Verify it's a view (same underlying storage)
	if result.Storage() != input.Storage() {
		t.Error("expected slice to share storage with input")
	}

	tt.AssertTensorShape(t, result, 3)
	tt.AssertTensorValues(t, result, []float32{2, 3, 4}, 1e-6)
}

// TestNarrowPipeline tests narrow (slice by length) in pipeline.
func TestNarrowPipeline(t *testing.T) {
	input := tt.RangeWithShape(4, 4)

	pipeline := tendo.Sequence("narrow-pipeline",
		tendo.NewNarrow(0, 1, 2), // Rows 1-2
		tendo.NewNarrow(1, 1, 2), // Cols 1-2
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 2, 2)
	// From 4x4 grid, extract 2x2 starting at (1,1)
	// Original: [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
	// Result: [[5,6],[9,10]]
	tt.AssertTensorValues(t, result, []float32{5, 6, 9, 10}, 1e-6)
}

// -----------------------------------------------------------------------------
// Broadcasting and Expand
// -----------------------------------------------------------------------------

// TestExpandBroadcast tests expand operation for broadcasting.
func TestExpandBroadcast(t *testing.T) {
	// Scalar-like tensor [1] -> expand to [3,4]
	input := tendo.MustFromSlice([]float32{5}, 1)

	pipeline := tendo.Sequence("expand",
		tendo.NewExpand(3, 4),
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 3, 4)

	// All values should be 5 (broadcasted)
	data := tt.TensorData(result)
	for i, v := range data {
		if v != 5 {
			t.Errorf("expected 5 at index %d, got %v", i, v)
		}
	}
}

// TestExpandNonContiguous verifies expand creates non-contiguous tensor.
func TestExpandNonContiguous(t *testing.T) {
	input := tendo.MustFromSlice([]float32{1, 2, 3}, 3, 1)

	expand := tendo.NewExpand(3, 4)
	result, err := expand.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("expand failed: %v", err)
	}

	// Expanded tensor should not be contiguous (stride=0 for broadcast dims)
	if result.IsContiguous() {
		t.Error("expected non-contiguous tensor after expand")
	}

	tt.AssertTensorShape(t, result, 3, 4)
}

// TestExpandThenContiguous tests expand followed by contiguous.
func TestExpandThenContiguous(t *testing.T) {
	input := tendo.MustFromSlice([]float32{1, 2}, 2, 1)

	pipeline := tendo.Sequence("expand-contiguous",
		tendo.NewExpand(2, 3),
		tendo.MakeContiguous(),
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	if !result.IsContiguous() {
		t.Error("expected contiguous after MakeContiguous")
	}

	tt.AssertTensorShape(t, result, 2, 3)
	// [[1,1,1],[2,2,2]]
	tt.AssertTensorValues(t, result, []float32{1, 1, 1, 2, 2, 2}, 1e-6)
}

// -----------------------------------------------------------------------------
// Squeeze/Unsqueeze Pipelines
// -----------------------------------------------------------------------------

// TestSqueezeUnsqueezePipeline tests dimension manipulation.
func TestSqueezeUnsqueezePipeline(t *testing.T) {
	input := tt.RangeWithShape(3, 4)

	pipeline := tendo.Sequence("dim-manipulation",
		tendo.NewUnsqueeze(0),    // [1,3,4]
		tendo.NewUnsqueeze(-1),   // [1,3,4,1]
		tendo.NewSqueeze(),       // [3,4] - removes all size-1 dims
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 3, 4)
	// Data should be unchanged
	expected := make([]float32, 12)
	for i := range expected {
		expected[i] = float32(i)
	}
	tt.AssertTensorValues(t, result, expected, 1e-6)
}

// TestUnsqueezeForBroadcast tests adding dims for broadcasting.
func TestUnsqueezeForBroadcast(t *testing.T) {
	// Prepare vector for broadcasting with matrix
	input := tendo.MustFromSlice([]float32{1, 2, 3}, 3)

	pipeline := tendo.Sequence("broadcast-prep",
		tendo.NewUnsqueeze(0), // [1,3] - can broadcast with [N,3]
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 1, 3)
	tt.AssertTensorValues(t, result, []float32{1, 2, 3}, 1e-6)
}

// -----------------------------------------------------------------------------
// Concatenation and Stacking
// -----------------------------------------------------------------------------

// TestCatPipeline tests concatenation along a dimension.
func TestCatPipeline(t *testing.T) {
	a := tendo.MustFromSlice([]float32{1, 2}, 1, 2)
	b := tendo.MustFromSlice([]float32{3, 4}, 1, 2)
	c := tendo.MustFromSlice([]float32{5, 6}, 1, 2)

	// Cat 'a' with [b, c] along dim 0
	cat := tendo.NewCat(nil, []*tendo.Tensor{b, c}, 0)
	result, err := cat.Process(context.Background(), a)
	if err != nil {
		t.Fatalf("cat failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 3, 2)
	tt.AssertTensorValues(t, result, []float32{1, 2, 3, 4, 5, 6}, 1e-6)
}

// TestStackPipeline tests stacking tensors along new dimension.
func TestStackPipeline(t *testing.T) {
	a := tendo.MustFromSlice([]float32{1, 2, 3}, 3)
	b := tendo.MustFromSlice([]float32{4, 5, 6}, 3)

	// Stack 'a' with [b] along new dim 0
	stack := tendo.NewStack(nil, []*tendo.Tensor{b}, 0)
	result, err := stack.Process(context.Background(), a)
	if err != nil {
		t.Fatalf("stack failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 2, 3)
	tt.AssertTensorValues(t, result, []float32{1, 2, 3, 4, 5, 6}, 1e-6)
}

// TestCatNonContiguous tests cat with non-contiguous inputs.
func TestCatNonContiguous(t *testing.T) {
	// Create non-contiguous tensor via transpose
	a := tt.RangeWithShape(2, 3)
	transposed, err := tendo.NewPermute(1, 0).Process(context.Background(), a)
	if err != nil {
		t.Fatalf("permute failed: %v", err)
	}
	// transposed is [[0,3],[1,4],[2,5]] shape [3,2], non-contiguous

	b := tendo.MustFromSlice([]float32{10, 11, 12}, 3, 1)

	// Cat along dim 1: [3,2] + [3,1] = [3,3]
	cat := tendo.NewCat(nil, []*tendo.Tensor{b}, 1)
	result, err := cat.Process(context.Background(), transposed)
	if err != nil {
		t.Fatalf("cat failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 3, 3)
	// [[0,3,10],[1,4,11],[2,5,12]]
	tt.AssertTensorValues(t, result, []float32{0, 3, 10, 1, 4, 11, 2, 5, 12}, 1e-6)
}

// TestCatAfterSlice tests cat with sliced (view) tensors.
func TestCatAfterSlice(t *testing.T) {
	full := tt.RangeWithShape(4, 4)

	// Slice out two rows
	slice1 := tendo.NewSlice(0, 0, 2) // First 2 rows
	slice2 := tendo.NewSlice(0, 2, 4) // Last 2 rows

	top, err := slice1.Process(context.Background(), full)
	if err != nil {
		t.Fatalf("slice1 failed: %v", err)
	}
	bottom, err := slice2.Process(context.Background(), full)
	if err != nil {
		t.Fatalf("slice2 failed: %v", err)
	}

	// Cat them back together
	cat := tendo.NewCat(nil, []*tendo.Tensor{bottom}, 0)
	result, err := cat.Process(context.Background(), top)
	if err != nil {
		t.Fatalf("cat failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 4, 4)

	// Should reconstruct original
	expected := make([]float32, 16)
	for i := range expected {
		expected[i] = float32(i)
	}
	tt.AssertTensorValues(t, result, expected, 1e-6)
}

// -----------------------------------------------------------------------------
// Complex Multi-Step Pipelines
// -----------------------------------------------------------------------------

// TestImagePreprocessPipeline simulates image preprocessing.
func TestImagePreprocessPipeline(t *testing.T) {
	// Simulate HWC image [2,3,4] (height=2, width=3, channels=4)
	input := tt.RangeWithShape(2, 3, 4)

	pipeline := tendo.Sequence("image-preprocess",
		tendo.NewPermute(2, 0, 1),  // HWC -> CHW: [4,2,3]
		tendo.NewUnsqueeze(0),      // Add batch dim: [1,4,2,3]
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 1, 4, 2, 3)
}

// TestAttentionReshapePipeline simulates attention head splitting.
func TestAttentionReshapePipeline(t *testing.T) {
	// Simulate [batch=2, seq=4, hidden=6]
	input := tt.RangeWithShape(2, 4, 6)

	numHeads := 2
	headDim := 3

	pipeline := tendo.Sequence("attention-reshape",
		// Reshape to [batch, seq, heads, head_dim]
		tendo.NewReshape(2, 4, numHeads, headDim),
		// Permute to [batch, heads, seq, head_dim]
		tendo.NewPermute(0, 2, 1, 3),
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 2, 2, 4, 3)
}

// TestBatchExtractPipeline extracts a single batch element.
func TestBatchExtractPipeline(t *testing.T) {
	// [batch=3, features=4]
	input := tt.RangeWithShape(3, 4)

	pipeline := tendo.Sequence("extract-batch",
		tendo.NewSlice(0, 1, 2),   // Get batch 1: [1,4]
		tendo.NewSqueezeDim(0),    // Remove batch dim: [4]
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 4)
	// Batch 1 = [4,5,6,7]
	tt.AssertTensorValues(t, result, []float32{4, 5, 6, 7}, 1e-6)
}

// TestViewChainPipeline tests multiple view operations.
func TestViewChainPipeline(t *testing.T) {
	input := tt.RangeWithShape(24)

	pipeline := tendo.Sequence("view-chain",
		tendo.NewView(2, 3, 4),
		tendo.NewView(6, 4),
		tendo.NewView(2, 12),
		tendo.NewView(24),
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 24)

	// Data should be unchanged through all views
	expected := make([]float32, 24)
	for i := range expected {
		expected[i] = float32(i)
	}
	tt.AssertTensorValues(t, result, expected, 1e-6)
}

// TestCloneInPipeline tests that clone creates independent tensor.
func TestCloneInPipeline(t *testing.T) {
	input := tt.RangeWithShape(4)

	// Process through pipeline that clones
	clone := tendo.Transform("clone", func(ctx context.Context, t *tendo.Tensor) *tendo.Tensor {
		return t.Clone()
	})

	result, err := clone.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("clone failed: %v", err)
	}

	// Should have different storage
	if result.Storage() == input.Storage() {
		t.Error("expected clone to have different storage")
	}

	// But same values
	tt.AssertTensorValues(t, result, []float32{0, 1, 2, 3}, 1e-6)
}

// TestInferredReshape tests reshape with -1 dimension inference.
func TestInferredReshape(t *testing.T) {
	input := tt.RangeWithShape(24)

	pipeline := tendo.Sequence("infer-reshape",
		tendo.NewReshape(2, -1, 4), // -1 infers to 3
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 2, 3, 4)
}

// TestMultipleInferredReshapes tests chained inferred reshapes.
func TestMultipleInferredReshapes(t *testing.T) {
	input := tt.RangeWithShape(60)

	pipeline := tendo.Sequence("multi-infer",
		tendo.NewReshape(3, -1),    // [3, 20]
		tendo.NewReshape(-1, 5, 4), // [3, 5, 4]
		tendo.NewReshape(15, -1),   // [15, 4]
	)

	result, err := pipeline.Process(context.Background(), input)
	if err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}

	tt.AssertTensorShape(t, result, 15, 4)
}
