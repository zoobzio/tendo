package gpt2

import "testing"

func TestConfig(t *testing.T) {
	cfg := Config{
		VocabSize:   50257,
		MaxSeqLen:   1024,
		Dim:         768,
		NumHeads:    12,
		NumLayers:   12,
		HiddenDim:   3072,
		Epsilon:     1e-5,
		DropoutProb: 0.1,
	}

	if cfg.VocabSize != 50257 {
		t.Errorf("VocabSize = %d, want 50257", cfg.VocabSize)
	}
	if cfg.MaxSeqLen != 1024 {
		t.Errorf("MaxSeqLen = %d, want 1024", cfg.MaxSeqLen)
	}
	if cfg.Dim != 768 {
		t.Errorf("Dim = %d, want 768", cfg.Dim)
	}
	if cfg.NumHeads != 12 {
		t.Errorf("NumHeads = %d, want 12", cfg.NumHeads)
	}
	if cfg.NumLayers != 12 {
		t.Errorf("NumLayers = %d, want 12", cfg.NumLayers)
	}
	if cfg.HiddenDim != 3072 {
		t.Errorf("HiddenDim = %d, want 3072", cfg.HiddenDim)
	}
}

func TestConfigSmall(t *testing.T) {
	cfg := ConfigSmall

	if cfg.VocabSize != 50257 {
		t.Errorf("VocabSize = %d, want 50257", cfg.VocabSize)
	}
	if cfg.Dim != 768 {
		t.Errorf("Dim = %d, want 768", cfg.Dim)
	}
	if cfg.NumHeads != 12 {
		t.Errorf("NumHeads = %d, want 12", cfg.NumHeads)
	}
	if cfg.NumLayers != 12 {
		t.Errorf("NumLayers = %d, want 12", cfg.NumLayers)
	}
	if cfg.HiddenDim != 3072 {
		t.Errorf("HiddenDim = %d, want 3072 (4*768)", cfg.HiddenDim)
	}
}

func TestConfigMedium(t *testing.T) {
	cfg := ConfigMedium

	if cfg.Dim != 1024 {
		t.Errorf("Dim = %d, want 1024", cfg.Dim)
	}
	if cfg.NumHeads != 16 {
		t.Errorf("NumHeads = %d, want 16", cfg.NumHeads)
	}
	if cfg.NumLayers != 24 {
		t.Errorf("NumLayers = %d, want 24", cfg.NumLayers)
	}
	if cfg.HiddenDim != 4096 {
		t.Errorf("HiddenDim = %d, want 4096 (4*1024)", cfg.HiddenDim)
	}
}

func TestConfigLarge(t *testing.T) {
	cfg := ConfigLarge

	if cfg.Dim != 1280 {
		t.Errorf("Dim = %d, want 1280", cfg.Dim)
	}
	if cfg.NumHeads != 20 {
		t.Errorf("NumHeads = %d, want 20", cfg.NumHeads)
	}
	if cfg.NumLayers != 36 {
		t.Errorf("NumLayers = %d, want 36", cfg.NumLayers)
	}
	if cfg.HiddenDim != 5120 {
		t.Errorf("HiddenDim = %d, want 5120 (4*1280)", cfg.HiddenDim)
	}
}

func TestConfigXL(t *testing.T) {
	cfg := ConfigXL

	if cfg.Dim != 1600 {
		t.Errorf("Dim = %d, want 1600", cfg.Dim)
	}
	if cfg.NumHeads != 25 {
		t.Errorf("NumHeads = %d, want 25", cfg.NumHeads)
	}
	if cfg.NumLayers != 48 {
		t.Errorf("NumLayers = %d, want 48", cfg.NumLayers)
	}
	if cfg.HiddenDim != 6400 {
		t.Errorf("HiddenDim = %d, want 6400 (4*1600)", cfg.HiddenDim)
	}
}

func TestConfig_HeadDimDivisibility(t *testing.T) {
	configs := []struct {
		name string
		cfg  Config
	}{
		{"Small", ConfigSmall},
		{"Medium", ConfigMedium},
		{"Large", ConfigLarge},
		{"XL", ConfigXL},
	}

	for _, tc := range configs {
		t.Run(tc.name, func(t *testing.T) {
			if tc.cfg.Dim%tc.cfg.NumHeads != 0 {
				t.Errorf("Dim %d not divisible by NumHeads %d", tc.cfg.Dim, tc.cfg.NumHeads)
			}
		})
	}
}
