package cpu

import (
	"unsafe"

	"github.com/zoobzio/tendo"
)

// Storage stores tensor data in CPU memory.
type Storage struct {
	data  []float32
	dtype tendo.DType
}

// NewStorage creates a new CPU storage with the given number of elements.
func NewStorage(numel int, dtype tendo.DType) *Storage {
	return &Storage{
		data:  make([]float32, numel),
		dtype: dtype,
	}
}

// NewStorageFromSlice creates a CPU storage from an existing slice.
// The slice is copied, not referenced.
func NewStorageFromSlice(data []float32, dtype tendo.DType) *Storage {
	copied := make([]float32, len(data))
	copy(copied, data)
	return &Storage{
		data:  copied,
		dtype: dtype,
	}
}

// Ptr returns the raw pointer to the data.
func (s *Storage) Ptr() uintptr {
	if len(s.data) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&s.data[0]))
}

// Size returns the size in bytes of the allocated memory.
func (s *Storage) Size() int {
	return len(s.data) * 4 // float32 = 4 bytes
}

// Len returns the number of elements in the storage.
func (s *Storage) Len() int {
	return len(s.data)
}

// Device returns the CPU device.
func (s *Storage) Device() tendo.Device {
	return tendo.CPUDevice()
}

// DType returns the data type.
func (s *Storage) DType() tendo.DType {
	return s.dtype
}

// Clone creates a deep copy of the storage.
func (s *Storage) Clone() tendo.Storage {
	return NewStorageFromSlice(s.data, s.dtype)
}

// Free releases the storage. For CPU storage, this is a no-op
// since Go's GC handles memory management.
func (s *Storage) Free() {
	s.data = nil
}

// Data returns the underlying float32 slice for direct access.
func (s *Storage) Data() []float32 {
	return s.data
}

// SetData sets the value at the given index.
func (s *Storage) SetData(index int, value float32) {
	s.data[index] = value
}

// GetData gets the value at the given index.
func (s *Storage) GetData(index int) float32 {
	return s.data[index]
}

// Fill sets all elements to the given value.
func (s *Storage) Fill(value float32) {
	for i := range s.data {
		s.data[i] = value
	}
}

// Compile-time check: Storage implements tendo.Storage
var _ tendo.Storage = (*Storage)(nil)
