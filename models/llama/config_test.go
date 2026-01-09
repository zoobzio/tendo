package llama

import "testing"

func TestConfig(t *testing.T) {
	cfg := Config{
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

	if cfg.VocabSize != 49152 {
		t.Errorf("VocabSize = %d, want 49152", cfg.VocabSize)
	}
	if cfg.Dim != 576 {
		t.Errorf("Dim = %d, want 576", cfg.Dim)
	}
	if cfg.NumKVHeads != 3 {
		t.Errorf("NumKVHeads = %d, want 3", cfg.NumKVHeads)
	}
}

func TestConfig_HeadDim(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want int
	}{
		{
			name: "SmolLM 135M",
			cfg:  ConfigSmolLM135M,
			want: 64, // 576 / 9
		},
		{
			name: "SmolLM 360M",
			cfg:  ConfigSmolLM360M,
			want: 64, // 960 / 15
		},
		{
			name: "Qwen2 0.5B",
			cfg:  ConfigQwen2_0_5B,
			want: 64, // 896 / 14
		},
		{
			name: "Llama 3.2 1B",
			cfg:  ConfigLlama3_2_1B,
			want: 64, // 2048 / 32
		},
		{
			name: "Llama 3.2 3B",
			cfg:  ConfigLlama3_2_3B,
			want: 128, // 3072 / 24
		},
		{
			name: "TinyLlama",
			cfg:  ConfigTinyLlama,
			want: 64, // 2048 / 32
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.cfg.HeadDim()
			if got != tt.want {
				t.Errorf("HeadDim() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestConfigSmolLM135M(t *testing.T) {
	cfg := ConfigSmolLM135M

	if cfg.VocabSize != 49152 {
		t.Errorf("VocabSize = %d, want 49152", cfg.VocabSize)
	}
	if cfg.Dim != 576 {
		t.Errorf("Dim = %d, want 576", cfg.Dim)
	}
	if cfg.NumLayers != 30 {
		t.Errorf("NumLayers = %d, want 30", cfg.NumLayers)
	}
	if cfg.NumHeads != 9 {
		t.Errorf("NumHeads = %d, want 9", cfg.NumHeads)
	}
	if cfg.NumKVHeads != 3 {
		t.Errorf("NumKVHeads = %d, want 3", cfg.NumKVHeads)
	}
	if !cfg.TieEmbeddings {
		t.Error("TieEmbeddings = false, want true")
	}
}

func TestConfigSmolLM360M(t *testing.T) {
	cfg := ConfigSmolLM360M

	if cfg.Dim != 960 {
		t.Errorf("Dim = %d, want 960", cfg.Dim)
	}
	if cfg.NumHeads != 15 {
		t.Errorf("NumHeads = %d, want 15", cfg.NumHeads)
	}
	if cfg.NumKVHeads != 5 {
		t.Errorf("NumKVHeads = %d, want 5", cfg.NumKVHeads)
	}
}

func TestConfigQwen2(t *testing.T) {
	cfg := ConfigQwen2_0_5B

	if cfg.VocabSize != 151936 {
		t.Errorf("VocabSize = %d, want 151936", cfg.VocabSize)
	}
	if cfg.MaxSeqLen != 32768 {
		t.Errorf("MaxSeqLen = %d, want 32768", cfg.MaxSeqLen)
	}
	if cfg.RoPEBase != 1000000 {
		t.Errorf("RoPEBase = %f, want 1000000", cfg.RoPEBase)
	}
	if cfg.RMSEpsilon != 1e-6 {
		t.Errorf("RMSEpsilon = %e, want 1e-6", cfg.RMSEpsilon)
	}
}

func TestConfigLlama3_2(t *testing.T) {
	configs := []struct {
		name string
		cfg  Config
		dim  int
	}{
		{"1B", ConfigLlama3_2_1B, 2048},
		{"3B", ConfigLlama3_2_3B, 3072},
	}

	for _, tc := range configs {
		t.Run(tc.name, func(t *testing.T) {
			if tc.cfg.VocabSize != 128256 {
				t.Errorf("VocabSize = %d, want 128256", tc.cfg.VocabSize)
			}
			if tc.cfg.Dim != tc.dim {
				t.Errorf("Dim = %d, want %d", tc.cfg.Dim, tc.dim)
			}
			if tc.cfg.MaxSeqLen != 131072 {
				t.Errorf("MaxSeqLen = %d, want 131072", tc.cfg.MaxSeqLen)
			}
			if tc.cfg.RoPEBase != 500000 {
				t.Errorf("RoPEBase = %f, want 500000", tc.cfg.RoPEBase)
			}
		})
	}
}

func TestConfigTinyLlama(t *testing.T) {
	cfg := ConfigTinyLlama

	if cfg.VocabSize != 32000 {
		t.Errorf("VocabSize = %d, want 32000", cfg.VocabSize)
	}
	if cfg.Dim != 2048 {
		t.Errorf("Dim = %d, want 2048", cfg.Dim)
	}
	if cfg.NumLayers != 22 {
		t.Errorf("NumLayers = %d, want 22", cfg.NumLayers)
	}
	if cfg.NumKVHeads != 4 {
		t.Errorf("NumKVHeads = %d, want 4", cfg.NumKVHeads)
	}
	if cfg.TieEmbeddings {
		t.Error("TieEmbeddings = true, want false")
	}
}

func TestConfig_GQARatio(t *testing.T) {
	// Test that GQA ratios are valid (NumHeads divisible by NumKVHeads)
	configs := []struct {
		name string
		cfg  Config
	}{
		{"SmolLM135M", ConfigSmolLM135M},
		{"SmolLM360M", ConfigSmolLM360M},
		{"Qwen2_0_5B", ConfigQwen2_0_5B},
		{"Llama3_2_1B", ConfigLlama3_2_1B},
		{"Llama3_2_3B", ConfigLlama3_2_3B},
		{"TinyLlama", ConfigTinyLlama},
	}

	for _, tc := range configs {
		t.Run(tc.name, func(t *testing.T) {
			if tc.cfg.NumHeads%tc.cfg.NumKVHeads != 0 {
				t.Errorf("NumHeads %d not divisible by NumKVHeads %d",
					tc.cfg.NumHeads, tc.cfg.NumKVHeads)
			}
		})
	}
}
