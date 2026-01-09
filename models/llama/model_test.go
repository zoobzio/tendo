package llama

import (
	"testing"

	"github.com/zoobzio/tendo"
	"github.com/zoobzio/tendo/nn"
)

func TestModel_Struct(t *testing.T) {
	// Test that Model struct has expected fields
	cfg := ConfigSmolLM135M
	tokenEmbed, _ := tendo.FromSlice(make([]float32, cfg.VocabSize*cfg.Dim), cfg.VocabSize, cfg.Dim)
	normW, _ := tendo.FromSlice(make([]float32, cfg.Dim), cfg.Dim)

	rope, _ := nn.NewRoPE(cfg.HeadDim(), cfg.MaxSeqLen, cfg.RoPEBase)

	model := &Model{
		RoPE:            rope,
		TokenEmbed:      tokenEmbed,
		FinalNormWeight: normW,
		OutputWeight:    tokenEmbed, // tied
		Layers:          make([]*Layer, cfg.NumLayers),
		Config:          cfg,
	}

	if model.RoPE == nil {
		t.Error("RoPE is nil")
	}
	if model.TokenEmbed == nil {
		t.Error("TokenEmbed is nil")
	}
	if model.FinalNormWeight == nil {
		t.Error("FinalNormWeight is nil")
	}
	if model.OutputWeight == nil {
		t.Error("OutputWeight is nil")
	}
	if len(model.Layers) != cfg.NumLayers {
		t.Errorf("Layers length = %d, want %d", len(model.Layers), cfg.NumLayers)
	}
}

func TestLayer_Struct(t *testing.T) {
	cfg := ConfigSmolLM135M
	dim := cfg.Dim
	hiddenDim := cfg.HiddenDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim()

	// Create minimal tensors for layer fields
	qWeight, _ := tendo.FromSlice(make([]float32, dim*dim), dim, dim)
	kWeight, _ := tendo.FromSlice(make([]float32, kvDim*dim), kvDim, dim)
	vWeight, _ := tendo.FromSlice(make([]float32, kvDim*dim), kvDim, dim)
	oWeight, _ := tendo.FromSlice(make([]float32, dim*dim), dim, dim)
	gateWeight, _ := tendo.FromSlice(make([]float32, hiddenDim*dim), hiddenDim, dim)
	upWeight, _ := tendo.FromSlice(make([]float32, hiddenDim*dim), hiddenDim, dim)
	downWeight, _ := tendo.FromSlice(make([]float32, dim*hiddenDim), dim, hiddenDim)
	normW, _ := tendo.FromSlice(make([]float32, dim), dim)

	qProj, _ := nn.NewLinear(qWeight, nil)
	kProj, _ := nn.NewLinear(kWeight, nil)
	vProj, _ := nn.NewLinear(vWeight, nil)
	oProj, _ := nn.NewLinear(oWeight, nil)
	gateProj, _ := nn.NewLinear(gateWeight, nil)
	upProj, _ := nn.NewLinear(upWeight, nil)
	downProj, _ := nn.NewLinear(downWeight, nil)

	layer := &Layer{
		QProj:          qProj,
		KProj:          kProj,
		VProj:          vProj,
		OProj:          oProj,
		GateProj:       gateProj,
		UpProj:         upProj,
		DownProj:       downProj,
		AttnNormWeight: normW,
		MLPNormWeight:  normW,
		NumHeads:       cfg.NumHeads,
		NumKVHeads:     cfg.NumKVHeads,
		HeadDim:        cfg.HeadDim(),
	}

	if layer.QProj == nil {
		t.Error("QProj is nil")
	}
	if layer.KProj == nil {
		t.Error("KProj is nil")
	}
	if layer.VProj == nil {
		t.Error("VProj is nil")
	}
	if layer.OProj == nil {
		t.Error("OProj is nil")
	}
	if layer.GateProj == nil {
		t.Error("GateProj is nil")
	}
	if layer.UpProj == nil {
		t.Error("UpProj is nil")
	}
	if layer.DownProj == nil {
		t.Error("DownProj is nil")
	}
	if layer.NumHeads != cfg.NumHeads {
		t.Errorf("NumHeads = %d, want %d", layer.NumHeads, cfg.NumHeads)
	}
	if layer.NumKVHeads != cfg.NumKVHeads {
		t.Errorf("NumKVHeads = %d, want %d", layer.NumKVHeads, cfg.NumKVHeads)
	}
}

func TestLoad_NonExistent(t *testing.T) {
	_, err := Load("/nonexistent/path/model.safetensors", ConfigSmolLM135M)
	if err == nil {
		t.Error("Load() with non-existent file should error")
	}
}

func TestDeviceBackend_Interface(t *testing.T) {
	// Compile-time check that DeviceBackend interface is correctly defined
	var _ DeviceBackend = (*mockDeviceBackend)(nil)
}

// mockDeviceBackend is a minimal implementation for interface testing.
type mockDeviceBackend struct{}

func (m *mockDeviceBackend) DeviceType() tendo.DeviceType {
	return tendo.CPU
}

func (m *mockDeviceBackend) NewStorage(numel int, dtype tendo.DType, deviceIndex int) (tendo.Storage, error) {
	return nil, nil
}

func (m *mockDeviceBackend) NewStorageFromSlice(data []float32, dtype tendo.DType, deviceIndex int) (tendo.Storage, error) {
	return nil, nil
}

func (m *mockDeviceBackend) CopyFrom(t *tendo.Tensor) (*tendo.Tensor, error) {
	return nil, nil
}
