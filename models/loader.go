package models

import (
	"fmt"

	"github.com/zoobzio/tendo"
)

// Weights holds a collection of named tensors loaded from a model file.
type Weights map[string]*tendo.Tensor

// Load opens a safetensors file and loads all tensors to CPU.
// Tensors are converted to Float32 for computation.
func Load(path string) (Weights, error) {
	f, err := tendo.OpenSafeTensors(path)
	if err != nil {
		return nil, fmt.Errorf("models: open %s: %w", path, err)
	}
	defer f.Close()

	tensors := f.Tensors()
	weights := make(Weights, len(tensors))
	targetDType := tendo.Float32

	for name := range tensors {
		t, err := f.LoadCPU(name, &targetDType)
		if err != nil {
			return nil, fmt.Errorf("models: load tensor %q: %w", name, err)
		}
		weights[name] = t
	}

	return weights, nil
}

// LoadOn opens a safetensors file and loads all tensors using the provided backend.
// Tensors are converted to Float32 for computation.
func LoadOn(path string, backend tendo.StorageBackend) (Weights, error) {
	f, err := tendo.OpenSafeTensors(path)
	if err != nil {
		return nil, fmt.Errorf("models: open %s: %w", path, err)
	}
	defer f.Close()

	tensors := f.Tensors()
	weights := make(Weights, len(tensors))
	targetDType := tendo.Float32

	for name := range tensors {
		t, err := f.Load(name, backend, &targetDType)
		if err != nil {
			return nil, fmt.Errorf("models: load tensor %q: %w", name, err)
		}
		weights[name] = t
	}

	return weights, nil
}

// Get retrieves a tensor by name, returning an error if not found.
func (w Weights) Get(name string) (*tendo.Tensor, error) {
	t, ok := w[name]
	if !ok {
		return nil, fmt.Errorf("models: weight %q not found", name)
	}
	return t, nil
}

// GetOptional retrieves a tensor by name, returning nil if not found.
func (w Weights) GetOptional(name string) *tendo.Tensor {
	return w[name]
}

// MustGet retrieves a tensor by name, panicking if not found.
func (w Weights) MustGet(name string) *tendo.Tensor {
	t, err := w.Get(name)
	if err != nil {
		panic(err)
	}
	return t
}

// Names returns all weight names.
func (w Weights) Names() []string {
	names := make([]string, 0, len(w))
	for name := range w {
		names = append(names, name)
	}
	return names
}

// Count returns the number of weights.
func (w Weights) Count() int {
	return len(w)
}
