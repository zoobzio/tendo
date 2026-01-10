//go:build tokenizers

package models

import (
	"fmt"

	"github.com/daulet/tokenizers"
)

// Tokenizer wraps a HuggingFace tokenizer for encoding/decoding text.
type Tokenizer struct {
	tk *tokenizers.Tokenizer
}

// LoadTokenizer loads a tokenizer from a tokenizer.json file.
func LoadTokenizer(path string) (*Tokenizer, error) {
	tk, err := tokenizers.FromFile(path)
	if err != nil {
		return nil, fmt.Errorf("models: load tokenizer: %w", err)
	}
	return &Tokenizer{tk: tk}, nil
}

// Encode converts text to token IDs.
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) []int {
	ids, _ := t.tk.Encode(text, addSpecialTokens)
	result := make([]int, len(ids))
	for i, id := range ids {
		result[i] = int(id)
	}
	return result
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int, skipSpecialTokens bool) string {
	uids := make([]uint32, len(ids))
	for i, id := range ids {
		uids[i] = uint32(id) //nolint:gosec // token IDs are always non-negative
	}
	return t.tk.Decode(uids, skipSpecialTokens)
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int {
	return int(t.tk.VocabSize())
}

// Close releases the tokenizer resources.
func (t *Tokenizer) Close() error {
	return t.tk.Close()
}
