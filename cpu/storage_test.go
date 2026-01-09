package cpu

import (
	"testing"

	"github.com/zoobzio/tendo"
)

func TestStorage_NewStorage(t *testing.T) {
	s := NewStorage(100, tendo.Float32)

	if s.Len() != 100 {
		t.Errorf("Len() = %d, want 100", s.Len())
	}

	if s.Size() != 400 {
		t.Errorf("Size() = %d, want 400", s.Size())
	}

	if s.DType() != tendo.Float32 {
		t.Errorf("DType() = %v, want Float32", s.DType())
	}

	if !s.Device().IsCPU() {
		t.Error("Device() should be CPU")
	}
}

func TestStorage_NewStorageInt64(t *testing.T) {
	s := NewStorage(50, tendo.Int64)

	if s.Len() != 50 {
		t.Errorf("Len() = %d, want 50", s.Len())
	}

	if s.Size() != 400 {
		t.Errorf("Size() = %d, want 400 (50 * 8)", s.Size())
	}

	if s.DType() != tendo.Int64 {
		t.Errorf("DType() = %v, want Int64", s.DType())
	}
}

func TestStorage_FromSlice(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5}
	s := NewStorageFromSlice(data, tendo.Float32)

	if s.Len() != 5 {
		t.Errorf("Len() = %d, want 5", s.Len())
	}

	// Verify data is copied
	data[0] = 999
	if s.Data()[0] == 999 {
		t.Error("Data should be copied, not referenced")
	}
}

func TestStorage_Int64FromSlice(t *testing.T) {
	data := []int64{10, 20, 30}
	s := NewInt64StorageFromSlice(data)

	if s.Len() != 3 {
		t.Errorf("Len() = %d, want 3", s.Len())
	}

	if s.DType() != tendo.Int64 {
		t.Errorf("DType() = %v, want Int64", s.DType())
	}

	// Verify data is copied
	data[0] = 999
	if s.Int64Data()[0] == 999 {
		t.Error("Data should be copied, not referenced")
	}
}

func TestStorage_Clone(t *testing.T) {
	original := NewStorageFromSlice([]float32{1, 2, 3}, tendo.Float32)
	cloned := original.Clone()

	if cloned.Len() != original.Len() {
		t.Errorf("Cloned Len() = %d, want %d", cloned.Len(), original.Len())
	}

	// Verify independence
	original.SetData(0, 999)
	clonedStorage, ok := cloned.(*Storage)
	if !ok {
		t.Fatal("Clone should return *Storage")
	}
	if clonedStorage.GetData(0) == 999 {
		t.Error("Clone should be independent of original")
	}
}

func TestStorage_Fill(t *testing.T) {
	s := NewStorage(5, tendo.Float32)
	s.Fill(2.71)

	for i := 0; i < s.Len(); i++ {
		if s.GetData(i) != 2.71 {
			t.Errorf("Data[%d] = %v, want 2.71", i, s.GetData(i))
		}
	}
}

func TestStorage_SetGetData(t *testing.T) {
	s := NewStorage(10, tendo.Float32)
	s.SetData(3, 123.0)

	if s.GetData(3) != 123.0 {
		t.Errorf("GetData(3) = %v, want 123.0", s.GetData(3))
	}
}

func TestStorage_Free(t *testing.T) {
	s := NewStorage(100, tendo.Float32)
	s.Free()

	if s.Data() != nil {
		t.Error("Data should be nil after Free")
	}
}

func TestStorage_Ptr(t *testing.T) {
	s := NewStorage(10, tendo.Float32)

	if s.Ptr() == 0 {
		t.Error("Ptr() should not be 0 for non-empty storage")
	}

	empty := NewStorage(0, tendo.Float32)
	if empty.Ptr() != 0 {
		t.Error("Ptr() should be 0 for empty storage")
	}
}

func TestStorage_PtrInt64(t *testing.T) {
	s := NewStorage(10, tendo.Int64)

	if s.Ptr() == 0 {
		t.Error("Ptr() should not be 0 for non-empty int64 storage")
	}

	empty := NewStorage(0, tendo.Int64)
	if empty.Ptr() != 0 {
		t.Error("Ptr() should be 0 for empty int64 storage")
	}
}

func TestStorage_ImplementsInterface(t *testing.T) {
	var _ tendo.Storage = (*Storage)(nil)
}
