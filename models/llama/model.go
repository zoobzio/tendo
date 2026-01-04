package llama

import (
	"fmt"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/models"
	"github.com/zoobzio/tendo/nn"
)

// Model represents a Llama-style model.
type Model struct {
	Config Config
	RoPE   *nn.RoPE

	// Embeddings
	TokenEmbed *tendo.Tensor // [vocab_size, dim]

	// Transformer layers
	Layers []*Layer

	// Final normalization
	FinalNormWeight *tendo.Tensor

	// Output head (may be tied to TokenEmbed)
	OutputWeight *tendo.Tensor // [vocab_size, dim]
}

// Layer represents a single Llama transformer layer.
type Layer struct {
	// Attention
	QProj *nn.Linear
	KProj *nn.Linear
	VProj *nn.Linear
	OProj *nn.Linear

	// MLP (SwiGLU)
	GateProj *nn.Linear
	UpProj   *nn.Linear
	DownProj *nn.Linear

	// Norms
	AttnNormWeight *tendo.Tensor
	MLPNormWeight  *tendo.Tensor

	// Config for GQA
	NumHeads   int
	NumKVHeads int
	HeadDim    int
}

// Load creates a Llama model from safetensors weights (loads to CPU).
func Load(path string, cfg Config) (*Model, error) {
	weights, err := models.Load(path)
	if err != nil {
		return nil, fmt.Errorf("llama: load weights: %w", err)
	}
	return FromWeights(weights, cfg)
}

// DeviceBackend combines storage creation and tensor copying capabilities.
type DeviceBackend interface {
	tendo.StorageBackend
	nn.StorageBackend
}

// LoadOn creates a Llama model with weights on the specified backend (e.g., CUDA).
func LoadOn(path string, cfg Config, backend DeviceBackend) (*Model, error) {
	weights, err := models.LoadOn(path, backend)
	if err != nil {
		return nil, fmt.Errorf("llama: load weights: %w", err)
	}

	model, err := FromWeights(weights, cfg)
	if err != nil {
		return nil, err
	}

	// Move RoPE caches to target device
	rope, err := model.RoPE.ToDevice(backend)
	if err != nil {
		return nil, fmt.Errorf("llama: move rope to device: %w", err)
	}
	model.RoPE = rope

	return model, nil
}

// FromWeights creates a Llama model from pre-loaded weights.
func FromWeights(weights models.Weights, cfg Config) (*Model, error) {
	// Initialize RoPE
	rope, err := nn.NewRoPE(cfg.HeadDim(), cfg.MaxSeqLen, cfg.RoPEBase)
	if err != nil {
		return nil, fmt.Errorf("llama: create rope: %w", err)
	}

	// Load token embeddings
	tokenEmbed, err := weights.Get("model.embed_tokens.weight")
	if err != nil {
		return nil, fmt.Errorf("llama: token embedding: %w", err)
	}

	// Load final norm
	finalNorm, err := weights.Get("model.norm.weight")
	if err != nil {
		return nil, fmt.Errorf("llama: final norm: %w", err)
	}

	// Load output head (may be tied)
	var outputWeight *tendo.Tensor
	if cfg.TieEmbeddings {
		outputWeight = tokenEmbed
	} else {
		outputWeight, err = weights.Get("lm_head.weight")
		if err != nil {
			return nil, fmt.Errorf("llama: output weight: %w", err)
		}
	}

	// Load transformer layers
	layers := make([]*Layer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		layer, err := loadLayer(weights, cfg, i)
		if err != nil {
			return nil, fmt.Errorf("llama: layer %d: %w", i, err)
		}
		layers[i] = layer
	}

	return &Model{
		Config:          cfg,
		RoPE:            rope,
		TokenEmbed:      tokenEmbed,
		Layers:          layers,
		FinalNormWeight: finalNorm,
		OutputWeight:    outputWeight,
	}, nil
}

func loadLayer(weights models.Weights, cfg Config, idx int) (*Layer, error) {
	prefix := fmt.Sprintf("model.layers.%d", idx)

	// Attention projections
	// Llama stores as [out_features, in_features] which matches nn.Linear expectation
	qWeight, err := weights.Get(prefix + ".self_attn.q_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("q_proj: %w", err)
	}
	kWeight, err := weights.Get(prefix + ".self_attn.k_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("k_proj: %w", err)
	}
	vWeight, err := weights.Get(prefix + ".self_attn.v_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("v_proj: %w", err)
	}
	oWeight, err := weights.Get(prefix + ".self_attn.o_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("o_proj: %w", err)
	}

	// Load optional biases (Qwen uses them, Llama doesn't)
	qBias := weights.GetOptional(prefix + ".self_attn.q_proj.bias")
	kBias := weights.GetOptional(prefix + ".self_attn.k_proj.bias")
	vBias := weights.GetOptional(prefix + ".self_attn.v_proj.bias")

	qProj, err := nn.NewLinear(qWeight, qBias)
	if err != nil {
		return nil, fmt.Errorf("create q_proj: %w", err)
	}
	kProj, err := nn.NewLinear(kWeight, kBias)
	if err != nil {
		return nil, fmt.Errorf("create k_proj: %w", err)
	}
	vProj, err := nn.NewLinear(vWeight, vBias)
	if err != nil {
		return nil, fmt.Errorf("create v_proj: %w", err)
	}
	oProj, err := nn.NewLinear(oWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("create o_proj: %w", err)
	}

	// MLP projections (SwiGLU)
	gateWeight, err := weights.Get(prefix + ".mlp.gate_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("gate_proj: %w", err)
	}
	upWeight, err := weights.Get(prefix + ".mlp.up_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("up_proj: %w", err)
	}
	downWeight, err := weights.Get(prefix + ".mlp.down_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("down_proj: %w", err)
	}

	gateProj, err := nn.NewLinear(gateWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("create gate_proj: %w", err)
	}
	upProj, err := nn.NewLinear(upWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("create up_proj: %w", err)
	}
	downProj, err := nn.NewLinear(downWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("create down_proj: %w", err)
	}

	// Layer norms
	attnNorm, err := weights.Get(prefix + ".input_layernorm.weight")
	if err != nil {
		return nil, fmt.Errorf("input_layernorm: %w", err)
	}
	mlpNorm, err := weights.Get(prefix + ".post_attention_layernorm.weight")
	if err != nil {
		return nil, fmt.Errorf("post_attention_layernorm: %w", err)
	}

	return &Layer{
		QProj:          qProj,
		KProj:          kProj,
		VProj:          vProj,
		OProj:          oProj,
		GateProj:       gateProj,
		UpProj:         upProj,
		DownProj:       downProj,
		AttnNormWeight: attnNorm,
		MLPNormWeight:  mlpNorm,
		NumHeads:       cfg.NumHeads,
		NumKVHeads:     cfg.NumKVHeads,
		HeadDim:        cfg.HeadDim(),
	}, nil
}
