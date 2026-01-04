package llama

import (
	"context"
	"fmt"
	"math"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/models"
	"github.com/zoobzio/tendo/nn"
)

// QuantizedModel is a Llama model with quantized linear layers.
type QuantizedModel struct {
	Config Config
	RoPE   *nn.RoPE

	// Embeddings (kept in fp32 - small relative to total)
	TokenEmbed *tendo.Tensor

	// Quantized transformer layers
	Layers []*QuantizedLayer

	// Final normalization (kept in fp32)
	FinalNormWeight *tendo.Tensor

	// Output head (kept in fp32 - used once per forward)
	OutputWeight *tendo.Tensor
}

// QuantizedLayer is a transformer layer with quantized projections.
type QuantizedLayer struct {
	// Quantized attention projections
	QProj *nn.QuantizedLinear
	KProj *nn.QuantizedLinear
	VProj *nn.QuantizedLinear
	OProj *nn.QuantizedLinear

	// Quantized MLP projections
	GateProj *nn.QuantizedLinear
	UpProj   *nn.QuantizedLinear
	DownProj *nn.QuantizedLinear

	// Norms (kept in fp32)
	AttnNormWeight *tendo.Tensor
	MLPNormWeight  *tendo.Tensor

	// Config
	NumHeads   int
	NumKVHeads int
	HeadDim    int
}

// QuantizedBackend extends Backend with quantized operations.
type QuantizedBackend interface {
	Backend
	nn.QuantizedLinearBackend
}

// QuantizeModel converts a regular model to quantized form.
// groupSize: 0 for per-channel, or group size (e.g., 128) for per-group
func QuantizeModel(m *Model, groupSize int) (*QuantizedModel, error) {
	layers := make([]*QuantizedLayer, len(m.Layers))

	for i, layer := range m.Layers {
		qlayer, err := quantizeLayer(layer, groupSize)
		if err != nil {
			return nil, fmt.Errorf("llama: quantize layer %d: %w", i, err)
		}
		layers[i] = qlayer
	}

	return &QuantizedModel{
		Config:          m.Config,
		RoPE:            m.RoPE,
		TokenEmbed:      m.TokenEmbed,
		Layers:          layers,
		FinalNormWeight: m.FinalNormWeight,
		OutputWeight:    m.OutputWeight,
	}, nil
}

func quantizeLayer(layer *Layer, groupSize int) (*QuantizedLayer, error) {
	qProj, err := nn.NewQuantizedLinear(layer.QProj, groupSize)
	if err != nil {
		return nil, fmt.Errorf("q_proj: %w", err)
	}
	kProj, err := nn.NewQuantizedLinear(layer.KProj, groupSize)
	if err != nil {
		return nil, fmt.Errorf("k_proj: %w", err)
	}
	vProj, err := nn.NewQuantizedLinear(layer.VProj, groupSize)
	if err != nil {
		return nil, fmt.Errorf("v_proj: %w", err)
	}
	oProj, err := nn.NewQuantizedLinear(layer.OProj, groupSize)
	if err != nil {
		return nil, fmt.Errorf("o_proj: %w", err)
	}

	gateProj, err := nn.NewQuantizedLinear(layer.GateProj, groupSize)
	if err != nil {
		return nil, fmt.Errorf("gate_proj: %w", err)
	}
	upProj, err := nn.NewQuantizedLinear(layer.UpProj, groupSize)
	if err != nil {
		return nil, fmt.Errorf("up_proj: %w", err)
	}
	downProj, err := nn.NewQuantizedLinear(layer.DownProj, groupSize)
	if err != nil {
		return nil, fmt.Errorf("down_proj: %w", err)
	}

	return &QuantizedLayer{
		QProj:          qProj,
		KProj:          kProj,
		VProj:          vProj,
		OProj:          oProj,
		GateProj:       gateProj,
		UpProj:         upProj,
		DownProj:       downProj,
		AttnNormWeight: layer.AttnNormWeight,
		MLPNormWeight:  layer.MLPNormWeight,
		NumHeads:       layer.NumHeads,
		NumKVHeads:     layer.NumKVHeads,
		HeadDim:        layer.HeadDim,
	}, nil
}

// Forward runs the quantized model.
func (m *QuantizedModel) Forward(ctx context.Context, tokenIDs *tendo.Tensor, caches []*KVCache, backend QuantizedBackend) (*Output, error) {
	// Determine position offset from cache
	posOffset := 0
	if caches != nil && len(caches) > 0 && caches[0] != nil {
		posOffset = caches[0].K.Size(2)
	}

	// Token embeddings
	h, err := backend.Embedding(ctx, m.TokenEmbed, tokenIDs)
	if err != nil {
		return nil, fmt.Errorf("llama: token embedding: %w", err)
	}

	// Run through transformer layers
	newCaches := make([]*KVCache, len(m.Layers))
	for i, layer := range m.Layers {
		var cache *KVCache
		if caches != nil && i < len(caches) {
			cache = caches[i]
		}

		prevH := h
		h, newCaches[i], err = m.forwardQuantizedLayer(ctx, h, layer, cache, posOffset, backend)
		prevH.Free() // Free previous layer's output
		if err != nil {
			return nil, fmt.Errorf("llama: layer %d: %w", i, err)
		}
	}

	// Final RMSNorm
	normalizedShape := []int{m.Config.Dim}
	hNormed, err := backend.RMSNorm(ctx, h, normalizedShape, m.FinalNormWeight, m.Config.RMSEpsilon)
	h.Free()
	if err != nil {
		return nil, fmt.Errorf("llama: final norm: %w", err)
	}

	// Output logits (kept in fp32)
	// Note: NewT creates a view (no new storage), so don't free outputT
	outputT, err := tendo.NewT().Process(ctx, m.OutputWeight)
	if err != nil {
		return nil, fmt.Errorf("llama: transpose output weight: %w", err)
	}
	logits, err := backend.MatMul(ctx, hNormed, outputT)
	hNormed.Free()
	if err != nil {
		return nil, fmt.Errorf("llama: output projection: %w", err)
	}

	return &Output{
		Logits:   logits,
		KVCaches: newCaches,
	}, nil
}

func (m *QuantizedModel) forwardQuantizedLayer(ctx context.Context, x *tendo.Tensor, layer *QuantizedLayer, cache *KVCache, posOffset int, backend QuantizedBackend) (*tendo.Tensor, *KVCache, error) {
	batch := x.Size(0)
	seq := x.Size(1)

	// Pre-attention RMSNorm
	normalizedShape := []int{m.Config.Dim}
	normed, err := backend.RMSNorm(ctx, x, normalizedShape, layer.AttnNormWeight, m.Config.RMSEpsilon)
	if err != nil {
		return nil, nil, fmt.Errorf("attention norm: %w", err)
	}

	// Quantized attention
	attnOut, newCache, err := m.forwardQuantizedAttention(ctx, normed, layer, cache, batch, seq, posOffset, backend)
	normed.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("attention: %w", err)
	}

	// Residual
	h, err := backend.Add(ctx, x, attnOut)
	attnOut.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("attention residual: %w", err)
	}

	// Pre-MLP RMSNorm
	normedMLP, err := backend.RMSNorm(ctx, h, normalizedShape, layer.MLPNormWeight, m.Config.RMSEpsilon)
	if err != nil {
		return nil, nil, fmt.Errorf("mlp norm: %w", err)
	}

	// Quantized MLP
	mlpOut, err := m.forwardQuantizedMLP(ctx, normedMLP, layer, backend)
	normedMLP.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("mlp: %w", err)
	}

	// Residual
	out, err := backend.Add(ctx, h, mlpOut)
	h.Free()
	mlpOut.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("mlp residual: %w", err)
	}

	return out, newCache, nil
}

func (m *QuantizedModel) forwardQuantizedAttention(ctx context.Context, x *tendo.Tensor, layer *QuantizedLayer, cache *KVCache, batch, seq, posOffset int, backend QuantizedBackend) (*tendo.Tensor, *KVCache, error) {
	// Quantized Q, K, V projections - these own their storage
	qProj, err := layer.QProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("q projection: %w", err)
	}
	kProj, err := layer.KProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("k projection: %w", err)
	}
	vProj, err := layer.VProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, nil, fmt.Errorf("v projection: %w", err)
	}

	// Reshape for multi-head attention (creates views, don't free)
	q, err := tendo.NewReshape(batch, seq, layer.NumHeads, layer.HeadDim).Process(ctx, qProj)
	if err != nil {
		return nil, nil, fmt.Errorf("reshape q: %w", err)
	}
	q, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, q)
	if err != nil {
		return nil, nil, fmt.Errorf("permute q: %w", err)
	}

	k, err := tendo.NewReshape(batch, seq, layer.NumKVHeads, layer.HeadDim).Process(ctx, kProj)
	if err != nil {
		return nil, nil, fmt.Errorf("reshape k: %w", err)
	}
	k, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, k)
	if err != nil {
		return nil, nil, fmt.Errorf("permute k: %w", err)
	}

	v, err := tendo.NewReshape(batch, seq, layer.NumKVHeads, layer.HeadDim).Process(ctx, vProj)
	if err != nil {
		return nil, nil, fmt.Errorf("reshape v: %w", err)
	}
	v, err = tendo.NewPermute(0, 2, 1, 3).Process(ctx, v)
	if err != nil {
		return nil, nil, fmt.Errorf("permute v: %w", err)
	}

	// Apply RoPE - allocates new storage
	qRoped, kRoped, err := m.RoPE.Apply(ctx, q, k, posOffset, backend)
	qProj.Free() // Done with Q projection storage
	kProj.Free() // Done with K projection storage
	if err != nil {
		vProj.Free()
		return nil, nil, fmt.Errorf("rope: %w", err)
	}

	// Handle KV cache
	if cache != nil {
		kCached, err := backend.Cat(ctx, []*tendo.Tensor{cache.K, kRoped}, 2)
		kRoped.Free()
		if err != nil {
			vProj.Free()
			return nil, nil, fmt.Errorf("cat cached k: %w", err)
		}
		kRoped = kCached

		vCached, err := backend.Cat(ctx, []*tendo.Tensor{cache.V, v}, 2)
		vProj.Free()
		if err != nil {
			kRoped.Free()
			return nil, nil, fmt.Errorf("cat cached v: %w", err)
		}
		v = vCached
	} else {
		vProj.Free()
	}
	newCache := &KVCache{K: kRoped, V: v}

	// GQA: repeat K, V heads (creates views)
	kExpanded := kRoped
	vExpanded := v
	if layer.NumKVHeads < layer.NumHeads {
		kExpanded, err = repeatKVHeads(ctx, kRoped, layer.NumHeads/layer.NumKVHeads)
		if err != nil {
			return nil, nil, fmt.Errorf("repeat k heads: %w", err)
		}
		vExpanded, err = repeatKVHeads(ctx, v, layer.NumHeads/layer.NumKVHeads)
		if err != nil {
			return nil, nil, fmt.Errorf("repeat v heads: %w", err)
		}
	}

	// Scaled dot-product attention
	kT, err := tendo.NewTranspose(-2, -1).Process(ctx, kExpanded)
	if err != nil {
		return nil, nil, fmt.Errorf("transpose k: %w", err)
	}

	scores, err := backend.MatMul(ctx, qRoped, kT)
	qRoped.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("q @ k^T: %w", err)
	}

	scale := 1.0 / float32(math.Sqrt(float64(layer.HeadDim)))
	scaleTensor, err := backend.Full(scale, 1)
	if err != nil {
		scores.Free()
		return nil, nil, fmt.Errorf("create scale: %w", err)
	}
	scoresScaled, err := backend.Mul(ctx, scores, scaleTensor)
	scores.Free()
	scaleTensor.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("scale scores: %w", err)
	}

	// Causal mask
	scoresMasked, err := applyCausalMask(ctx, scoresScaled, backend)
	scoresScaled.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("causal mask: %w", err)
	}

	// Softmax
	attnWeights, err := backend.Softmax(ctx, scoresMasked, -1)
	scoresMasked.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("softmax: %w", err)
	}

	// Attention output
	attnMatmul, err := backend.MatMul(ctx, attnWeights, vExpanded)
	attnWeights.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("weights @ v: %w", err)
	}

	// Reshape back (creates views)
	attnOut, err := tendo.NewPermute(0, 2, 1, 3).Process(ctx, attnMatmul)
	if err != nil {
		return nil, nil, fmt.Errorf("permute output: %w", err)
	}
	attnOut, err = tendo.NewReshape(batch, -1, m.Config.Dim).Process(ctx, attnOut)
	if err != nil {
		return nil, nil, fmt.Errorf("reshape output: %w", err)
	}

	// Quantized output projection
	out, err := layer.OProj.Forward(ctx, attnOut, backend)
	attnMatmul.Free()
	if err != nil {
		return nil, nil, fmt.Errorf("o projection: %w", err)
	}

	return out, newCache, nil
}

func (m *QuantizedModel) forwardQuantizedMLP(ctx context.Context, x *tendo.Tensor, layer *QuantizedLayer, backend QuantizedBackend) (*tendo.Tensor, error) {
	// Quantized SwiGLU
	gateProj, err := layer.GateProj.Forward(ctx, x, backend)
	if err != nil {
		return nil, fmt.Errorf("gate projection: %w", err)
	}
	gate, err := backend.SiLU(ctx, gateProj)
	gateProj.Free()
	if err != nil {
		return nil, fmt.Errorf("silu: %w", err)
	}

	up, err := layer.UpProj.Forward(ctx, x, backend)
	if err != nil {
		gate.Free()
		return nil, fmt.Errorf("up projection: %w", err)
	}

	h, err := backend.Mul(ctx, gate, up)
	gate.Free()
	up.Free()
	if err != nil {
		return nil, fmt.Errorf("gate * up: %w", err)
	}

	out, err := layer.DownProj.Forward(ctx, h, backend)
	h.Free()
	if err != nil {
		return nil, fmt.Errorf("down projection: %w", err)
	}

	return out, nil
}

// Generate produces tokens using the quantized model.
func (m *QuantizedModel) Generate(ctx context.Context, promptIDs []int, cfg GenerateConfig, backend QuantizedBackend) (*GenerateResult, error) {
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
