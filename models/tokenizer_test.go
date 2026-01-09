package models

import (
	"os"
	"testing"
)

// TestTokenizer_LoadNonExistent tests loading from a non-existent file.
func TestTokenizer_LoadNonExistent(t *testing.T) {
	_, err := LoadTokenizer("/nonexistent/path/tokenizer.json")
	if err == nil {
		t.Error("LoadTokenizer() with non-existent file should error")
	}
}

// TestTokenizer_LoadInvalid tests loading from an invalid file.
func TestTokenizer_LoadInvalid(t *testing.T) {
	// Create a temporary invalid file
	f, err := os.CreateTemp("", "invalid_tokenizer_*.json")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(f.Name())

	// Write invalid JSON
	if _, err := f.WriteString("not valid json"); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}
	f.Close()

	_, err = LoadTokenizer(f.Name())
	if err == nil {
		t.Error("LoadTokenizer() with invalid file should error")
	}
}

// TestTokenizer_Interface verifies the Tokenizer struct has expected methods.
// This is a compile-time check that the interface is correctly defined.
func TestTokenizer_Interface(t *testing.T) {
	// This test verifies the Tokenizer type has the expected methods
	// by checking that method references compile.
	var tok *Tokenizer

	// Encode method signature
	_ = func(text string, addSpecial bool) []int {
		return tok.Encode(text, addSpecial)
	}

	// Decode method signature
	_ = func(ids []int, skipSpecial bool) string {
		return tok.Decode(ids, skipSpecial)
	}

	// VocabSize method signature
	_ = func() int {
		return tok.VocabSize()
	}

	// Close method signature
	_ = func() error {
		return tok.Close()
	}
}
