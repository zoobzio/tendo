package models

import (
	"testing"

	"github.com/zoobzio/tendo"
)

func TestWeights_Get(t *testing.T) {
	tensor, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 2, 2)
	weights := Weights{
		"layer.weight": tensor,
	}

	// Existing key
	got, err := weights.Get("layer.weight")
	if err != nil {
		t.Errorf("Get() unexpected error: %v", err)
	}
	if got != tensor {
		t.Error("Get() returned wrong tensor")
	}

	// Non-existent key
	_, err = weights.Get("nonexistent")
	if err == nil {
		t.Error("Get() with non-existent key should error")
	}
}

func TestWeights_GetOptional(t *testing.T) {
	tensor, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 2, 2)
	weights := Weights{
		"layer.weight": tensor,
	}

	// Existing key
	got := weights.GetOptional("layer.weight")
	if got != tensor {
		t.Error("GetOptional() returned wrong tensor")
	}

	// Non-existent key
	got = weights.GetOptional("nonexistent")
	if got != nil {
		t.Error("GetOptional() with non-existent key should return nil")
	}
}

func TestWeights_MustGet(t *testing.T) {
	tensor, _ := tendo.FromSlice([]float32{1, 2, 3, 4}, 2, 2)
	weights := Weights{
		"layer.weight": tensor,
	}

	// Existing key
	got := weights.MustGet("layer.weight")
	if got != tensor {
		t.Error("MustGet() returned wrong tensor")
	}

	// Non-existent key should panic
	defer func() {
		if r := recover(); r == nil {
			t.Error("MustGet() with non-existent key should panic")
		}
	}()
	weights.MustGet("nonexistent")
}

func TestWeights_Names(t *testing.T) {
	t1, _ := tendo.FromSlice([]float32{1, 2}, 2)
	t2, _ := tendo.FromSlice([]float32{3, 4}, 2)
	t3, _ := tendo.FromSlice([]float32{5, 6}, 2)

	weights := Weights{
		"a": t1,
		"b": t2,
		"c": t3,
	}

	names := weights.Names()
	if len(names) != 3 {
		t.Errorf("Names() length = %d, want 3", len(names))
	}

	// Check all names are present (order not guaranteed)
	nameSet := make(map[string]bool)
	for _, name := range names {
		nameSet[name] = true
	}
	for _, expected := range []string{"a", "b", "c"} {
		if !nameSet[expected] {
			t.Errorf("Names() missing %q", expected)
		}
	}
}

func TestWeights_Count(t *testing.T) {
	tests := []struct {
		name    string
		weights Weights
		want    int
	}{
		{
			name:    "empty",
			weights: Weights{},
			want:    0,
		},
		{
			name: "single",
			weights: func() Weights {
				t, _ := tendo.FromSlice([]float32{1}, 1)
				return Weights{"a": t}
			}(),
			want: 1,
		},
		{
			name: "multiple",
			weights: func() Weights {
				t, _ := tendo.FromSlice([]float32{1}, 1)
				return Weights{"a": t, "b": t, "c": t}
			}(),
			want: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.weights.Count(); got != tt.want {
				t.Errorf("Count() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestWeights_Empty(t *testing.T) {
	weights := Weights{}

	if len(weights.Names()) != 0 {
		t.Error("Empty weights should have no names")
	}
	if weights.Count() != 0 {
		t.Error("Empty weights should have count 0")
	}
	if weights.GetOptional("anything") != nil {
		t.Error("Empty weights GetOptional should return nil")
	}
}
