//go:build cuda

package cuda

import (
	"testing"

	"github.com/zoobzio/tendo"
)

// Note: These tests require CUDA hardware and will be skipped
// if CUDA is not available.

func TestStorage_ImplementsInterface(t *testing.T) {
	// Compile-time check that Storage implements tendo.Storage
	var _ tendo.Storage = (*Storage)(nil)
}

func TestStorage_NewStorage(t *testing.T) {
	if !IsAvailable() {
		t.Skip("CUDA not available")
	}

	s, err := NewStorage(100, tendo.Float32, 0)
	if err != nil {
		t.Fatalf("NewStorage failed: %v", err)
	}
	defer s.Free()

	if s.Len() != 100 {
		t.Errorf("Len() = %d, want 100", s.Len())
	}

	if s.DType() != tendo.Float32 {
		t.Errorf("DType() = %v, want Float32", s.DType())
	}

	if !s.Device().IsCUDA() {
		t.Error("Device() should be CUDA")
	}
}

func TestStorage_Clone(t *testing.T) {
	if !IsAvailable() {
		t.Skip("CUDA not available")
	}

	original, err := NewStorage(50, tendo.Float32, 0)
	if err != nil {
		t.Fatalf("NewStorage failed: %v", err)
	}
	defer original.Free()

	cloned := original.Clone()
	defer cloned.Free()

	if cloned.Len() != original.Len() {
		t.Errorf("Cloned Len() = %d, want %d", cloned.Len(), original.Len())
	}
}

func TestStorage_CopyToHost(t *testing.T) {
	if !IsAvailable() {
		t.Skip("CUDA not available")
	}

	// Create storage and copy to device
	hostData := []float32{1, 2, 3, 4, 5}
	s, err := NewStorageFromSlice(hostData, tendo.Float32, 0)
	if err != nil {
		t.Fatalf("NewStorageFromSlice failed: %v", err)
	}
	defer s.Free()

	// Copy back to host
	retrieved, err := s.CopyToHost()
	if err != nil {
		t.Fatalf("CopyToHost failed: %v", err)
	}

	if len(retrieved) != len(hostData) {
		t.Fatalf("Retrieved length = %d, want %d", len(retrieved), len(hostData))
	}

	for i, v := range hostData {
		if retrieved[i] != v {
			t.Errorf("Retrieved[%d] = %v, want %v", i, retrieved[i], v)
		}
	}
}
