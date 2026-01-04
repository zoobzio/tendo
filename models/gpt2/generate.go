package gpt2

import (
	"context"
	"fmt"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/models"
	"github.com/zoobzio/tendo/nn"
)

// GenerateConfig configures text generation.
type GenerateConfig struct {
	MaxTokens   int
	TopK        int
	EOSToken    int
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
		EOSToken:    50256, // GPT-2 <|endoftext|>
	}
}

// GenerateResult contains generation output.
type GenerateResult struct {
	TokenIDs  []int
	NumTokens int
}

// Generate produces tokens from a prompt.
// Returns the generated token IDs (including the prompt).
func (m *Model) Generate(ctx context.Context, promptIDs []int, cfg GenerateConfig, backend Backend) (*GenerateResult, error) {
	// Convert sampling config
	samplingCfg := models.SamplingConfig{
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		TopK:        cfg.TopK,
		TopP:        cfg.TopP,
	}
	if cfg.EOSToken >= 0 {
		samplingCfg.StopTokens = []int{cfg.EOSToken}
	}
	sampler := models.NewSampler(samplingCfg)

	// Start with prompt tokens
	tokens := make([]int, len(promptIDs))
	copy(tokens, promptIDs)

	// KV cache for efficient generation
	var caches []*nn.KVCache

	generated := 0
	for generated < cfg.MaxTokens {
		// Create input tensor for current tokens
		// On first iteration: full prompt
		// On subsequent iterations: just the last token (with KV cache)
		var inputTokens []int
		if caches == nil {
			inputTokens = tokens
		} else {
			inputTokens = tokens[len(tokens)-1:]
		}

		input, err := tokensToTensor(inputTokens)
		if err != nil {
			return nil, fmt.Errorf("gpt2: create input tensor: %w", err)
		}

		// Forward pass
		output, err := m.Forward(ctx, input, caches, backend)
		if err != nil {
			return nil, fmt.Errorf("gpt2: forward: %w", err)
		}

		// Update cache for next iteration
		caches = output.KVCaches

		// Extract logits for last position
		logits, err := models.ExtractLastLogits(output.Logits)
		if err != nil {
			return nil, fmt.Errorf("gpt2: extract logits: %w", err)
		}

		// Sample next token
		nextToken := sampler.Sample(logits)

		// Check for stop token
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
func (m *Model) GenerateGreedy(ctx context.Context, promptIDs []int, maxTokens int, eosToken int, backend Backend) (*GenerateResult, error) {
	tokens := make([]int, len(promptIDs))
	copy(tokens, promptIDs)

	var caches []*nn.KVCache
	generated := 0

	for generated < maxTokens {
		var inputTokens []int
		if caches == nil {
			inputTokens = tokens
		} else {
			inputTokens = tokens[len(tokens)-1:]
		}

		input, err := tokensToTensor(inputTokens)
		if err != nil {
			return nil, fmt.Errorf("gpt2: create input tensor: %w", err)
		}

		output, err := m.Forward(ctx, input, caches, backend)
		if err != nil {
			return nil, fmt.Errorf("gpt2: forward: %w", err)
		}

		caches = output.KVCaches

		logits, err := models.ExtractLastLogits(output.Logits)
		if err != nil {
			return nil, fmt.Errorf("gpt2: extract logits: %w", err)
		}

		nextToken := models.Greedy(logits)

		if eosToken >= 0 && nextToken == eosToken {
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

func tokensToTensor(tokens []int) (*tendo.Tensor, error) {
	data := make([]float32, len(tokens))
	for i, t := range tokens {
		data[i] = float32(t)
	}
	return tendo.FromSlice(data, 1, len(tokens)) // [batch=1, seq]
}
