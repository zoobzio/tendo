//go:build cuda

// Package main provides an example inference CLI for Llama models.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/zoobzio/tendo/cpu"
	"github.com/zoobzio/tendo/cuda"
	"github.com/zoobzio/tendo/models"
	"github.com/zoobzio/tendo/models/llama"
)

const (
	deviceCUDA = "cuda"
	deviceCPU  = "cpu"
)

func main() {
	modelPath := flag.String("model", "", "path to model.safetensors")
	tokenizerPath := flag.String("tokenizer", "", "path to tokenizer.json")
	prompt := flag.String("prompt", "The capital of France is", "input prompt")
	maxTokens := flag.Int("max-tokens", 50, "maximum tokens to generate")
	temperature := flag.Float64("temperature", 0.7, "sampling temperature")
	device := flag.String("device", "cuda", "device: cuda or cpu")
	configName := flag.String("config", "smollm-135m", "model config: smollm-135m, smollm-360m, qwen2-0.5b, llama3.2-1b, llama3.2-3b, tinyllama")
	flag.Parse()

	if *modelPath == "" || *tokenizerPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s --model <path> --tokenizer <path> [options]\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Select model configuration
	cfg, err := getConfig(*configName)
	if err != nil {
		log.Fatalf("Invalid config: %v", err)
	}

	// Validate device early (before allocating resources)
	deviceLower := strings.ToLower(*device)
	switch deviceLower {
	case deviceCUDA:
		if !cuda.IsCUDAAvailable() {
			log.Fatal("CUDA not available")
		}
	case deviceCPU:
		// CPU always available
	default:
		log.Fatalf("Unknown device: %s (use 'cuda' or 'cpu')", *device)
	}

	// Run inference (deferred cleanup happens correctly within this function)
	if err := runInference(*modelPath, *tokenizerPath, *configName, deviceLower, cfg, *prompt, *maxTokens, *temperature); err != nil {
		log.Fatalf("Inference failed: %v", err)
	}
}

func runInference(modelPath, tokenizerPath, configName, deviceLower string, cfg llama.Config, prompt string, maxTokens int, temperature float64) error {
	// Load tokenizer
	fmt.Println("Loading tokenizer...")
	tokenizer, err := models.LoadTokenizer(tokenizerPath)
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}
	defer func() { _ = tokenizer.Close() }() //nolint:errcheck // best-effort cleanup

	// Load model
	fmt.Printf("Loading model (%s on %s)...\n", configName, deviceLower)
	var model *llama.Model

	switch deviceLower {
	case deviceCUDA:
		backend := cuda.NewBackend()
		model, err = llama.LoadOn(modelPath, cfg, backend)
	case deviceCPU:
		// For CPU, load to CPU then use CPU backend
		model, err = llama.Load(modelPath, cfg)
	}

	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	fmt.Println("Model loaded.")

	// Encode prompt
	promptIDs := tokenizer.Encode(prompt, false)
	fmt.Printf("Prompt: %q (%d tokens)\n", prompt, len(promptIDs))

	// Set up generation config
	genCfg := llama.GenerateConfig{
		MaxTokens:   maxTokens,
		Temperature: float32(temperature),
	}

	// Create backend for inference
	var backend llama.Backend
	switch deviceLower {
	case deviceCUDA:
		backend = cuda.NewBackend()
	case deviceCPU:
		backend = cpu.NewBackend()
	}

	// Generate
	fmt.Println("Generating...")
	ctx := context.Background()
	result, err := model.Generate(ctx, promptIDs, genCfg, backend)
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}

	// Decode output
	output := tokenizer.Decode(result.TokenIDs, true)
	fmt.Printf("\n--- Output (%d tokens generated) ---\n%s\n", result.NumTokens, output)

	return nil
}

func getConfig(name string) (llama.Config, error) {
	switch strings.ToLower(name) {
	case "smollm-135m":
		return llama.ConfigSmolLM135M, nil
	case "smollm-360m":
		return llama.ConfigSmolLM360M, nil
	case "qwen2-0.5b":
		return llama.ConfigQwen2_0_5B, nil
	case "llama3.2-1b":
		return llama.ConfigLlama3_2_1B, nil
	case "llama3.2-3b":
		return llama.ConfigLlama3_2_3B, nil
	case "tinyllama":
		return llama.ConfigTinyLlama, nil
	default:
		return llama.Config{}, fmt.Errorf("unknown config: %s", name)
	}
}
