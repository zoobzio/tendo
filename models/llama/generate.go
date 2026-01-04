package llama

import (
	"context"
	"fmt"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/models"
)

// GenerateConfig configures text generation.
type GenerateConfig struct {
	StopTokens  []int
	MaxTokens   int
	TopK        int
	Temperature float32
	TopP        float32
}

// DefaultGenerateConfig returns sensible defaults.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:   100,
		Temperature: 1.0,
		TopK:        0,
		TopP:        0,
		StopTokens:  nil,
	}
}

// GenerateResult contains generation output.
type GenerateResult struct {
	TokenIDs  []int
	NumTokens int
}

// Generate produces tokens from a prompt.
// Returns the generated token IDs (including the prompt).
//
//nolint:dupl // Intentional duplication with QuantizedModel.Generate - different types require separate implementations
func (m *Model) Generate(ctx context.Context, promptIDs []int, cfg GenerateConfig, backend Backend) (*GenerateResult, error) {
	samplingCfg := models.SamplingConfig{
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		TopK:        cfg.TopK,
		TopP:        cfg.TopP,
		StopTokens:  cfg.StopTokens,
	}
	sampler := models.NewSampler(samplingCfg)

	tokens := make([]int, len(promptIDs))
	copy(tokens, promptIDs)

	var caches []*KVCache
	generated := 0

	for generated < cfg.MaxTokens {
		var inputTokens []int
		if caches == nil {
			inputTokens = tokens
		} else {
			inputTokens = tokens[len(tokens)-1:]
		}

		input, err := tokensToTensor(inputTokens, backend)
		if err != nil {
			return nil, fmt.Errorf("llama: create input tensor: %w", err)
		}

		output, err := m.Forward(ctx, input, caches, backend)
		if err != nil {
			return nil, fmt.Errorf("llama: forward: %w", err)
		}

		caches = output.KVCaches

		logits, err := models.ExtractLastLogits(output.Logits)
		if err != nil {
			return nil, fmt.Errorf("llama: extract logits: %w", err)
		}

		nextToken := sampler.Sample(logits)

		if sampler.IsStopToken(nextToken) {
			break
		}

		tokens = append(tokens, nextToken)
		generated++
	}

	return &GenerateResult{
		TokenIDs:  tokens,
		NumTokens: generated,
	}, nil
}

// GenerateGreedy produces tokens using greedy decoding (deterministic).
func (m *Model) GenerateGreedy(ctx context.Context, promptIDs []int, maxTokens int, stopTokens []int, backend Backend) (*GenerateResult, error) {
	tokens := make([]int, len(promptIDs))
	copy(tokens, promptIDs)

	stopSet := make(map[int]bool)
	for _, t := range stopTokens {
		stopSet[t] = true
	}

	var caches []*KVCache
	generated := 0

	for generated < maxTokens {
		var inputTokens []int
		if caches == nil {
			inputTokens = tokens
		} else {
			inputTokens = tokens[len(tokens)-1:]
		}

		input, err := tokensToTensor(inputTokens, backend)
		if err != nil {
			return nil, fmt.Errorf("llama: create input tensor: %w", err)
		}

		output, err := m.Forward(ctx, input, caches, backend)
		if err != nil {
			return nil, fmt.Errorf("llama: forward: %w", err)
		}

		caches = output.KVCaches

		logits, err := models.ExtractLastLogits(output.Logits)
		if err != nil {
			return nil, fmt.Errorf("llama: extract logits: %w", err)
		}

		nextToken := models.Greedy(logits)

		if stopSet[nextToken] {
			break
		}

		tokens = append(tokens, nextToken)
		generated++
	}

	return &GenerateResult{
		TokenIDs:  tokens,
		NumTokens: generated,
	}, nil
}

func tokensToTensor(tokens []int, backend Backend) (*tendo.Tensor, error) {
	data := make([]int64, len(tokens))
	for i, t := range tokens {
		data[i] = int64(t)
	}
	return backend.FromInt64Slice(data, 1, len(tokens)) // [batch=1, seq]
}
