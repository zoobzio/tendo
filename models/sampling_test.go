package models

import (
	"math"
	"testing"
)

func TestDefaultSamplingConfig(t *testing.T) {
	cfg := DefaultSamplingConfig()

	if cfg.MaxTokens != 100 {
		t.Errorf("MaxTokens = %d, want 100", cfg.MaxTokens)
	}
	if cfg.Temperature != 1.0 {
		t.Errorf("Temperature = %f, want 1.0", cfg.Temperature)
	}
	if cfg.TopK != 0 {
		t.Errorf("TopK = %d, want 0", cfg.TopK)
	}
	if cfg.TopP != 0 {
		t.Errorf("TopP = %f, want 0", cfg.TopP)
	}
	if cfg.StopTokens != nil {
		t.Errorf("StopTokens = %v, want nil", cfg.StopTokens)
	}
}

func TestSamplingConfig(t *testing.T) {
	cfg := SamplingConfig{
		MaxTokens:   50,
		Temperature: 0.8,
		TopK:        40,
		TopP:        0.9,
		StopTokens:  []int{50256},
	}

	if cfg.MaxTokens != 50 {
		t.Errorf("MaxTokens = %d, want 50", cfg.MaxTokens)
	}
	if cfg.Temperature != 0.8 {
		t.Errorf("Temperature = %f, want 0.8", cfg.Temperature)
	}
	if cfg.TopK != 40 {
		t.Errorf("TopK = %d, want 40", cfg.TopK)
	}
	if cfg.TopP != 0.9 {
		t.Errorf("TopP = %f, want 0.9", cfg.TopP)
	}
	if len(cfg.StopTokens) != 1 || cfg.StopTokens[0] != 50256 {
		t.Errorf("StopTokens = %v, want [50256]", cfg.StopTokens)
	}
}

func TestNewSampler(t *testing.T) {
	cfg := DefaultSamplingConfig()
	sampler := NewSampler(cfg)

	if sampler == nil {
		t.Fatal("NewSampler() returned nil")
	}
	if sampler.rng == nil {
		t.Error("Sampler.rng is nil")
	}
}

func TestGreedy(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		want   int
	}{
		{
			name:   "first element largest",
			logits: []float32{5.0, 2.0, 1.0, 0.5},
			want:   0,
		},
		{
			name:   "last element largest",
			logits: []float32{1.0, 2.0, 3.0, 10.0},
			want:   3,
		},
		{
			name:   "middle element largest",
			logits: []float32{1.0, 5.0, 2.0},
			want:   1,
		},
		{
			name:   "negative values",
			logits: []float32{-1.0, -0.5, -2.0},
			want:   1,
		},
		{
			name:   "single element",
			logits: []float32{1.0},
			want:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Greedy(tt.logits)
			if got != tt.want {
				t.Errorf("Greedy() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestSampler_IsStopToken(t *testing.T) {
	cfg := SamplingConfig{
		StopTokens: []int{50256, 50257, 0},
	}
	sampler := NewSampler(cfg)

	tests := []struct {
		token int
		want  bool
	}{
		{50256, true},
		{50257, true},
		{0, true},
		{1, false},
		{12345, false},
	}

	for _, tt := range tests {
		if got := sampler.IsStopToken(tt.token); got != tt.want {
			t.Errorf("IsStopToken(%d) = %v, want %v", tt.token, got, tt.want)
		}
	}
}

func TestSampler_IsStopToken_Empty(t *testing.T) {
	cfg := SamplingConfig{StopTokens: nil}
	sampler := NewSampler(cfg)

	if sampler.IsStopToken(0) {
		t.Error("IsStopToken(0) with nil StopTokens should be false")
	}
}

func TestSoftmax(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	probs := softmax(logits)

	// Check sum is 1
	sum := float32(0)
	for _, p := range probs {
		sum += p
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}

	// Check probabilities are ordered (higher logit = higher prob)
	if probs[0] >= probs[1] || probs[1] >= probs[2] {
		t.Errorf("softmax order incorrect: %v", probs)
	}

	// All probabilities should be positive
	for i, p := range probs {
		if p <= 0 {
			t.Errorf("softmax[%d] = %f, want > 0", i, p)
		}
	}
}

func TestSoftmax_Stability(t *testing.T) {
	// Test with large values that could cause overflow without max subtraction
	logits := []float32{1000.0, 1001.0, 1002.0}
	probs := softmax(logits)

	sum := float32(0)
	for _, p := range probs {
		sum += p
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
			t.Error("softmax produced NaN or Inf")
		}
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}
}

func TestFindKthLargest(t *testing.T) {
	tests := []struct {
		name   string
		values []float32
		k      int
		want   float32
	}{
		{
			name:   "k=1 (largest)",
			values: []float32{3.0, 1.0, 4.0, 1.0, 5.0, 9.0},
			k:      1,
			want:   9.0,
		},
		{
			name:   "k=3",
			values: []float32{3.0, 1.0, 4.0, 1.0, 5.0, 9.0},
			k:      3,
			want:   4.0,
		},
		{
			name:   "k >= len (returns min)",
			values: []float32{3.0, 1.0, 4.0},
			k:      10,
			want:   1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy since the function may modify the slice
			values := make([]float32, len(tt.values))
			copy(values, tt.values)
			got := findKthLargest(values, tt.k)
			if got != tt.want {
				t.Errorf("findKthLargest() = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestSortIndicesByValue(t *testing.T) {
	values := []float32{1.0, 5.0, 2.0, 4.0, 3.0}
	indices := []int{0, 1, 2, 3, 4}

	sortIndicesByValue(indices, values)

	// Should be sorted descending by value: 1, 3, 4, 2, 0
	expected := []int{1, 3, 4, 2, 0}
	for i := range indices {
		if indices[i] != expected[i] {
			t.Errorf("indices[%d] = %d, want %d", i, indices[i], expected[i])
		}
	}
}

func TestSampler_Sample_Deterministic(t *testing.T) {
	// With temperature=0 (or very low), sampling should be nearly deterministic
	cfg := SamplingConfig{
		Temperature: 0.001, // Very low temperature
	}
	sampler := NewSampler(cfg)

	logits := []float32{1.0, 5.0, 2.0, 3.0}

	// Should almost always pick index 1 (highest logit)
	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		// Make copy since sampling may modify logits
		l := make([]float32, len(logits))
		copy(l, logits)
		token := sampler.Sample(l)
		counts[token]++
	}

	if counts[1] < 95 {
		t.Errorf("Low temperature sampling picked index 1 only %d/100 times", counts[1])
	}
}
