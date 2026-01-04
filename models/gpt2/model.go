package gpt2

import (
	"context"
	"fmt"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/models"
	"github.com/zoobzio/tendo/nn"
)

// Model represents a GPT-2 model.
type Model struct {
	TokenEmbed      *tendo.Tensor
	PositionEmbed   *tendo.Tensor
	FinalNormWeight *tendo.Tensor
	FinalNormBias   *tendo.Tensor
	Layers          []*Layer
	Config          Config
}

// Layer represents a single GPT-2 transformer layer.
type Layer struct {
	Attention *nn.Attention
	MLP       *nn.MLP

	AttnNormWeight *tendo.Tensor
	AttnNormBias   *tendo.Tensor
	MLPNormWeight  *tendo.Tensor
	MLPNormBias    *tendo.Tensor
}

// Load creates a GPT-2 model from safetensors weights.
func Load(path string, cfg Config) (*Model, error) {
	weights, err := models.Load(path)
	if err != nil {
		return nil, fmt.Errorf("gpt2: load weights: %w", err)
	}
	return FromWeights(weights, cfg)
}

// FromWeights creates a GPT-2 model from pre-loaded weights.
func FromWeights(weights models.Weights, cfg Config) (*Model, error) {
	// Load embeddings
	tokenEmbed, err := weights.Get("wte.weight")
	if err != nil {
		return nil, fmt.Errorf("gpt2: token embedding: %w", err)
	}
	posEmbed, err := weights.Get("wpe.weight")
	if err != nil {
		return nil, fmt.Errorf("gpt2: position embedding: %w", err)
	}

	// Load final layer norm
	finalNormW, err := weights.Get("ln_f.weight")
	if err != nil {
		return nil, fmt.Errorf("gpt2: final norm weight: %w", err)
	}
	finalNormB, err := weights.Get("ln_f.bias")
	if err != nil {
		return nil, fmt.Errorf("gpt2: final norm bias: %w", err)
	}

	// Load transformer layers
	layers := make([]*Layer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		layer, err := loadLayer(weights, cfg, i)
		if err != nil {
			return nil, fmt.Errorf("gpt2: layer %d: %w", i, err)
		}
		layers[i] = layer
	}

	return &Model{
		Config:          cfg,
		TokenEmbed:      tokenEmbed,
		PositionEmbed:   posEmbed,
		Layers:          layers,
		FinalNormWeight: finalNormW,
		FinalNormBias:   finalNormB,
	}, nil
}

func loadLayer(weights models.Weights, cfg Config, idx int) (*Layer, error) {
	prefix := fmt.Sprintf("h.%d", idx)

	// Load attention weights
	// GPT-2 uses combined c_attn for Q, K, V: [dim, 3*dim]
	// We need to split it into separate Q, K, V weights
	cAttnWeight, err := weights.Get(prefix + ".attn.c_attn.weight")
	if err != nil {
		return nil, fmt.Errorf("c_attn weight: %w", err)
	}
	cAttnBias, err := weights.Get(prefix + ".attn.c_attn.bias")
	if err != nil {
		return nil, fmt.Errorf("c_attn bias: %w", err)
	}

	// Split combined QKV weight [dim, 3*dim] -> 3x [dim, dim]
	// GPT-2 Conv1D stores as [in, out], we need [out, in] for nn.Linear
	qWeight, kWeight, vWeight, err := splitQKV(cAttnWeight, cfg.Dim)
	if err != nil {
		return nil, fmt.Errorf("split qkv weight: %w", err)
	}
	qBias, kBias, vBias, err := splitQKVBias(cAttnBias, cfg.Dim)
	if err != nil {
		return nil, fmt.Errorf("split qkv bias: %w", err)
	}

	// Output projection
	cProjWeight, err := weights.Get(prefix + ".attn.c_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("c_proj weight: %w", err)
	}
	cProjBias, err := weights.Get(prefix + ".attn.c_proj.bias")
	if err != nil {
		return nil, fmt.Errorf("c_proj bias: %w", err)
	}

	// Transpose output projection weight from [in, out] to [out, in]
	oWeight, err := transposeWeight(cProjWeight)
	if err != nil {
		return nil, fmt.Errorf("transpose o weight: %w", err)
	}

	// Create attention layer
	attnCfg := nn.AttentionConfig{
		Dim:      cfg.Dim,
		NumHeads: cfg.NumHeads,
		Bias:     true,
	}
	attention, err := nn.NewAttention(attnCfg, qWeight, kWeight, vWeight, oWeight, qBias, kBias, vBias, cProjBias)
	if err != nil {
		return nil, fmt.Errorf("create attention: %w", err)
	}

	// Load MLP weights
	// GPT-2 MLP: c_fc [dim, 4*dim], c_proj [4*dim, dim]
	fcWeight, err := weights.Get(prefix + ".mlp.c_fc.weight")
	if err != nil {
		return nil, fmt.Errorf("c_fc weight: %w", err)
	}
	fcBias, err := weights.Get(prefix + ".mlp.c_fc.bias")
	if err != nil {
		return nil, fmt.Errorf("c_fc bias: %w", err)
	}
	projWeight, err := weights.Get(prefix + ".mlp.c_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("mlp c_proj weight: %w", err)
	}
	projBias, err := weights.Get(prefix + ".mlp.c_proj.bias")
	if err != nil {
		return nil, fmt.Errorf("mlp c_proj bias: %w", err)
	}

	// Transpose MLP weights from [in, out] to [out, in]
	upWeight, err := transposeWeight(fcWeight)
	if err != nil {
		return nil, fmt.Errorf("transpose up weight: %w", err)
	}
	downWeight, err := transposeWeight(projWeight)
	if err != nil {
		return nil, fmt.Errorf("transpose down weight: %w", err)
	}

	// Create MLP (GPT-2 uses GELU, non-gated)
	mlpCfg := nn.MLPConfig{
		Dim:        cfg.Dim,
		HiddenDim:  cfg.HiddenDim,
		Activation: nn.GELU,
		Gated:      false,
		Bias:       true,
	}
	mlp, err := nn.NewMLP(mlpCfg, upWeight, downWeight, nil, fcBias, projBias, nil)
	if err != nil {
		return nil, fmt.Errorf("create mlp: %w", err)
	}

	// Load layer norms
	ln1Weight, err := weights.Get(prefix + ".ln_1.weight")
	if err != nil {
		return nil, fmt.Errorf("ln_1 weight: %w", err)
	}
	ln1Bias, err := weights.Get(prefix + ".ln_1.bias")
	if err != nil {
		return nil, fmt.Errorf("ln_1 bias: %w", err)
	}
	ln2Weight, err := weights.Get(prefix + ".ln_2.weight")
	if err != nil {
		return nil, fmt.Errorf("ln_2 weight: %w", err)
	}
	ln2Bias, err := weights.Get(prefix + ".ln_2.bias")
	if err != nil {
		return nil, fmt.Errorf("ln_2 bias: %w", err)
	}

	return &Layer{
		Attention:      attention,
		MLP:            mlp,
		AttnNormWeight: ln1Weight,
		AttnNormBias:   ln1Bias,
		MLPNormWeight:  ln2Weight,
		MLPNormBias:    ln2Bias,
	}, nil
}

// splitQKV splits combined QKV weight [dim, 3*dim] into Q, K, V weights [dim, dim].
// Also transposes from GPT-2's [in, out] to our [out, in] format.
func splitQKV(combined *tendo.Tensor, dim int) (q, k, v *tendo.Tensor, err error) {
	ctx := context.Background()

	// combined is [dim, 3*dim], we want 3x [dim, dim] then transpose to [dim, dim]
	// First narrow along dim 1 to get Q, K, V portions
	qPart, err := tendo.NewNarrow(1, 0, dim).Process(ctx, combined)
	if err != nil {
		return nil, nil, nil, err
	}
	kPart, err := tendo.NewNarrow(1, dim, dim).Process(ctx, combined)
	if err != nil {
		return nil, nil, nil, err
	}
	vPart, err := tendo.NewNarrow(1, 2*dim, dim).Process(ctx, combined)
	if err != nil {
		return nil, nil, nil, err
	}

	// Transpose from [in, out] to [out, in]
	q, err = transposeWeight(qPart)
	if err != nil {
		return nil, nil, nil, err
	}
	k, err = transposeWeight(kPart)
	if err != nil {
		return nil, nil, nil, err
	}
	v, err = transposeWeight(vPart)
	if err != nil {
		return nil, nil, nil, err
	}

	return q, k, v, nil
}

// splitQKVBias splits combined QKV bias [3*dim] into Q, K, V biases [dim].
func splitQKVBias(combined *tendo.Tensor, dim int) (q, k, v *tendo.Tensor, err error) {
	ctx := context.Background()

	q, err = tendo.NewNarrow(0, 0, dim).Process(ctx, combined)
	if err != nil {
		return nil, nil, nil, err
	}
	k, err = tendo.NewNarrow(0, dim, dim).Process(ctx, combined)
	if err != nil {
		return nil, nil, nil, err
	}
	v, err = tendo.NewNarrow(0, 2*dim, dim).Process(ctx, combined)
	if err != nil {
		return nil, nil, nil, err
	}

	return q, k, v, nil
}

// transposeWeight transposes a 2D weight tensor.
func transposeWeight(t *tendo.Tensor) (*tendo.Tensor, error) {
	return tendo.NewT().Process(context.Background(), t)
}
