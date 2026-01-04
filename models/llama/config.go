// Package llama provides Llama model implementation for inference.
package llama

// Config holds Llama model configuration.
type Config struct {
	VocabSize     int     // vocabulary size
	Dim           int     // model dimension (hidden_size)
	NumLayers     int     // number of transformer layers
	NumHeads      int     // number of attention heads
	NumKVHeads    int     // number of key-value heads (for GQA; if == NumHeads, standard MHA)
	HiddenDim     int     // MLP intermediate dimension
	MaxSeqLen     int     // maximum sequence length
	RoPEBase      float32 // RoPE base frequency
	RMSEpsilon    float32 // RMSNorm epsilon
	TieEmbeddings bool    // whether input/output embeddings are tied
}

// HeadDim returns the dimension per attention head.
func (c Config) HeadDim() int {
	return c.Dim / c.NumHeads
}

// Predefined configurations for common models.
var (
	// SmolLM 135M - very small, good for testing.
	ConfigSmolLM135M = Config{
		VocabSize:     49152,
		Dim:           576,
		NumLayers:     30,
		NumHeads:      9,
		NumKVHeads:    3,
		HiddenDim:     1536,
		MaxSeqLen:     2048,
		RoPEBase:      10000,
		RMSEpsilon:    1e-5,
		TieEmbeddings: true,
	}

	// SmolLM 360M.
	ConfigSmolLM360M = Config{
		VocabSize:     49152,
		Dim:           960,
		NumLayers:     32,
		NumHeads:      15,
		NumKVHeads:    5,
		HiddenDim:     2560,
		MaxSeqLen:     2048,
		RoPEBase:      10000,
		RMSEpsilon:    1e-5,
		TieEmbeddings: true,
	}

	// Qwen2 0.5B.
	ConfigQwen2_0_5B = Config{
		VocabSize:     151936,
		Dim:           896,
		NumLayers:     24,
		NumHeads:      14,
		NumKVHeads:    2,
		HiddenDim:     4864,
		MaxSeqLen:     32768,
		RoPEBase:      1000000,
		RMSEpsilon:    1e-6,
		TieEmbeddings: true,
	}

	// Llama 3.2 1B.
	ConfigLlama3_2_1B = Config{
		VocabSize:     128256,
		Dim:           2048,
		NumLayers:     16,
		NumHeads:      32,
		NumKVHeads:    8,
		HiddenDim:     8192,
		MaxSeqLen:     131072,
		RoPEBase:      500000,
		RMSEpsilon:    1e-5,
		TieEmbeddings: true,
	}

	// Llama 3.2 3B.
	ConfigLlama3_2_3B = Config{
		VocabSize:     128256,
		Dim:           3072,
		NumLayers:     28,
		NumHeads:      24,
		NumKVHeads:    8,
		HiddenDim:     8192,
		MaxSeqLen:     131072,
		RoPEBase:      500000,
		RMSEpsilon:    1e-5,
		TieEmbeddings: true,
	}

	// TinyLlama 1.1B.
	ConfigTinyLlama = Config{
		VocabSize:     32000,
		Dim:           2048,
		NumLayers:     22,
		NumHeads:      32,
		NumKVHeads:    4,
		HiddenDim:     5632,
		MaxSeqLen:     2048,
		RoPEBase:      10000,
		RMSEpsilon:    1e-5,
		TieEmbeddings: false,
	}
)
