package gpt2

// Config holds GPT-2 model configuration.
type Config struct {
	VocabSize    int     // vocabulary size
	MaxSeqLen    int     // maximum sequence length
	Dim          int     // model dimension (n_embd)
	NumHeads     int     // number of attention heads
	NumLayers    int     // number of transformer layers
	HiddenDim    int     // MLP hidden dimension (4 * dim)
	Epsilon      float32 // layer norm epsilon
	DropoutProb  float32 // dropout probability (unused in inference)
}

// Predefined configurations for standard GPT-2 models.
var (
	// ConfigSmall is GPT-2 small (124M parameters).
	ConfigSmall = Config{
		VocabSize:   50257,
		MaxSeqLen:   1024,
		Dim:         768,
		NumHeads:    12,
		NumLayers:   12,
		HiddenDim:   3072, // 4 * 768
		Epsilon:     1e-5,
		DropoutProb: 0.1,
	}

	// ConfigMedium is GPT-2 medium (355M parameters).
	ConfigMedium = Config{
		VocabSize:   50257,
		MaxSeqLen:   1024,
		Dim:         1024,
		NumHeads:    16,
		NumLayers:   24,
		HiddenDim:   4096, // 4 * 1024
		Epsilon:     1e-5,
		DropoutProb: 0.1,
	}

	// ConfigLarge is GPT-2 large (774M parameters).
	ConfigLarge = Config{
		VocabSize:   50257,
		MaxSeqLen:   1024,
		Dim:         1280,
		NumHeads:    20,
		NumLayers:   36,
		HiddenDim:   5120, // 4 * 1280
		Epsilon:     1e-5,
		DropoutProb: 0.1,
	}

	// ConfigXL is GPT-2 XL (1.5B parameters).
	ConfigXL = Config{
		VocabSize:   50257,
		MaxSeqLen:   1024,
		Dim:         1600,
		NumHeads:    25,
		NumLayers:   48,
		HiddenDim:   6400, // 4 * 1600
		Epsilon:     1e-5,
		DropoutProb: 0.1,
	}
)
