package gpt2

import (
	"testing"

	"github.com/zoobzio/tendo"
)

func TestModel_Struct(t *testing.T) {
	// Test that Model struct has expected fields
	tokenEmbed, _ := tendo.FromSlice(make([]float32, 50257*768), 50257, 768)
	posEmbed, _ := tendo.FromSlice(make([]float32, 1024*768), 1024, 768)
	normW, _ := tendo.FromSlice(make([]float32, 768), 768)
	normB, _ := tendo.FromSlice(make([]float32, 768), 768)

	model := &Model{
		TokenEmbed:      tokenEmbed,
		PositionEmbed:   posEmbed,
		FinalNormWeight: normW,
		FinalNormBias:   normB,
		Layers:          make([]*Layer, 12),
		Config:          ConfigSmall,
	}

	if model.TokenEmbed == nil {
		t.Error("TokenEmbed is nil")
	}
	if model.PositionEmbed == nil {
		t.Error("PositionEmbed is nil")
	}
	if model.FinalNormWeight == nil {
		t.Error("FinalNormWeight is nil")
	}
	if model.FinalNormBias == nil {
		t.Error("FinalNormBias is nil")
	}
	if len(model.Layers) != 12 {
		t.Errorf("Layers length = %d, want 12", len(model.Layers))
	}
}

func TestLayer_Struct(t *testing.T) {
	// Test that Layer struct has expected fields
	normW, _ := tendo.FromSlice(make([]float32, 768), 768)
	normB, _ := tendo.FromSlice(make([]float32, 768), 768)

	layer := &Layer{
		Attention:      nil, // Would need actual nn.Attention
		MLP:            nil, // Would need actual nn.MLP
		AttnNormWeight: normW,
		AttnNormBias:   normB,
		MLPNormWeight:  normW,
		MLPNormBias:    normB,
	}

	if layer.AttnNormWeight == nil {
		t.Error("AttnNormWeight is nil")
	}
	if layer.AttnNormBias == nil {
		t.Error("AttnNormBias is nil")
	}
	if layer.MLPNormWeight == nil {
		t.Error("MLPNormWeight is nil")
	}
	if layer.MLPNormBias == nil {
		t.Error("MLPNormBias is nil")
	}
}

func TestLoad_NonExistent(t *testing.T) {
	_, err := Load("/nonexistent/path/model.safetensors", ConfigSmall)
	if err == nil {
		t.Error("Load() with non-existent file should error")
	}
}
