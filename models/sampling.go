package models

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"github.com/zoobzio/tendo"
)

// SamplingConfig configures text generation sampling.
type SamplingConfig struct {
	MaxTokens   int     // maximum tokens to generate
	Temperature float32 // sampling temperature (1.0 = normal, <1 = more deterministic)
	TopK        int     // top-k sampling (0 = disabled)
	TopP        float32 // nucleus sampling threshold (0 = disabled)
	StopTokens  []int   // tokens that stop generation
}

// DefaultSamplingConfig returns sensible defaults.
func DefaultSamplingConfig() SamplingConfig {
	return SamplingConfig{
		MaxTokens:   100,
		Temperature: 1.0,
		TopK:        0,
		TopP:        0,
		StopTokens:  nil,
	}
}

// Sampler handles token sampling from logits.
type Sampler struct {
	config SamplingConfig
	rng    *rand.Rand
}

// NewSampler creates a sampler with the given config.
func NewSampler(cfg SamplingConfig) *Sampler {
	return &Sampler{
		config: cfg,
		rng:    rand.New(rand.NewSource(rand.Int63())),
	}
}

// Sample selects the next token from logits [vocab_size].
func (s *Sampler) Sample(logits []float32) int {
	// Apply temperature
	if s.config.Temperature != 1.0 && s.config.Temperature > 0 {
		for i := range logits {
			logits[i] /= s.config.Temperature
		}
	}

	// Apply top-k filtering
	if s.config.TopK > 0 && s.config.TopK < len(logits) {
		logits = s.applyTopK(logits)
	}

	// Apply top-p (nucleus) filtering
	if s.config.TopP > 0 && s.config.TopP < 1.0 {
		logits = s.applyTopP(logits)
	}

	// Convert to probabilities
	probs := softmax(logits)

	// Sample from distribution
	return s.sampleFromProbs(probs)
}

// Greedy returns the token with highest logit (argmax).
func Greedy(logits []float32) int {
	maxIdx := 0
	maxVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
			maxIdx = i
		}
	}
	return maxIdx
}

func (s *Sampler) applyTopK(logits []float32) []float32 {
	// Find k-th largest value
	k := s.config.TopK
	threshold := findKthLargest(logits, k)

	// Mask values below threshold
	for i := range logits {
		if logits[i] < threshold {
			logits[i] = float32(math.Inf(-1))
		}
	}
	return logits
}

func (s *Sampler) applyTopP(logits []float32) []float32 {
	// Convert to probabilities
	probs := softmax(logits)

	// Sort indices by probability descending
	indices := make([]int, len(probs))
	for i := range indices {
		indices[i] = i
	}
	sortIndicesByValue(indices, probs)

	// Find cutoff
	cumsum := float32(0)
	cutoff := 0
	for i, idx := range indices {
		cumsum += probs[idx]
		if cumsum >= s.config.TopP {
			cutoff = i + 1
			break
		}
	}

	// Mask tokens beyond cutoff
	for i := cutoff; i < len(indices); i++ {
		logits[indices[i]] = float32(math.Inf(-1))
	}
	return logits
}

func (s *Sampler) sampleFromProbs(probs []float32) int {
	r := s.rng.Float32()
	cumsum := float32(0)
	for i, p := range probs {
		cumsum += p
		if r < cumsum {
			return i
		}
	}
	return len(probs) - 1
}

func softmax(logits []float32) []float32 {
	// Find max for numerical stability
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp and sum
	probs := make([]float32, len(logits))
	sum := float32(0)
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxVal)))
		sum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func findKthLargest(values []float32, k int) float32 {
	// Simple O(n*k) implementation - fine for vocab sizes
	if k >= len(values) {
		min := values[0]
		for _, v := range values[1:] {
			if v < min {
				min = v
			}
		}
		return min
	}

	// Make a copy and partially sort
	sorted := make([]float32, len(values))
	copy(sorted, values)

	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] > sorted[maxIdx] {
				maxIdx = j
			}
		}
		sorted[i], sorted[maxIdx] = sorted[maxIdx], sorted[i]
	}
	return sorted[k-1]
}

func sortIndicesByValue(indices []int, values []float32) {
	// Simple insertion sort - fine for small arrays
	for i := 1; i < len(indices); i++ {
		j := i
		for j > 0 && values[indices[j]] > values[indices[j-1]] {
			indices[j], indices[j-1] = indices[j-1], indices[j]
			j--
		}
	}
}

// IsStopToken checks if the token should stop generation.
func (s *Sampler) IsStopToken(token int) bool {
	for _, stop := range s.config.StopTokens {
		if token == stop {
			return true
		}
	}
	return false
}

// ExtractLastLogits gets the logits for the last position from [batch, seq, vocab] tensor.
func ExtractLastLogits(logits *tendo.Tensor) ([]float32, error) {
	// logits shape: [batch, seq, vocab]
	if logits.Dim() != 3 {
		return nil, fmt.Errorf("expected 3D logits, got %dD", logits.Dim())
	}

	seq := logits.Size(1)
	vocab := logits.Size(2)

	// Get the last position logits
	ctx := context.Background()
	lastLogits, err := tendo.NewNarrow(1, seq-1, 1).Process(ctx, logits)
	if err != nil {
		return nil, err
	}

	// Reshape to [vocab]
	lastLogits, err = tendo.NewReshape(vocab).Process(ctx, lastLogits)
	if err != nil {
		return nil, err
	}

	// Copy to host if needed
	data, err := lastLogits.Data()
	if err != nil {
		return nil, err
	}

	return data, nil
}
