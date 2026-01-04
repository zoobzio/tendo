package cpu

import (
	"unsafe"

	"github.com/zoobzio/tendo"
)

// Storage stores tensor data in CPU memory.
// Supports both float32 data (for compute) and int64 data (for indices).
type Storage struct {
	float32Data []float32
	int64Data   []int64
	dtype       tendo.DType
}

// NewStorage creates a new CPU storage with the given number of elements.
func NewStorage(numel int, dtype tendo.DType) *Storage {
	s := &Storage{dtype: dtype}
	if dtype == tendo.Int64 {
		s.int64Data = make([]int64, numel)
	} else {
		s.float32Data = make([]float32, numel)
	}
	return s
}

// NewStorageFromSlice creates a CPU storage from an existing float32 slice.
// The slice is copied, not referenced.
func NewStorageFromSlice(data []float32, dtype tendo.DType) *Storage {
	copied := make([]float32, len(data))
	copy(copied, data)
	return &Storage{
		float32Data: copied,
		dtype:       dtype,
	}
}

// NewInt64StorageFromSlice creates a CPU storage from an existing int64 slice.
// The slice is copied, not referenced.
func NewInt64StorageFromSlice(data []int64) *Storage {
	copied := make([]int64, len(data))
	copy(copied, data)
	return &Storage{
		int64Data: copied,
		dtype:     tendo.Int64,
	}
}

// Ptr returns the raw pointer to the data.
func (s *Storage) Ptr() uintptr {
	if s.dtype == tendo.Int64 {
		if len(s.int64Data) == 0 {
			return 0
		}
		return uintptr(unsafe.Pointer(&s.int64Data[0]))
	}
	if len(s.float32Data) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&s.float32Data[0]))
}

// Size returns the size in bytes of the allocated memory.
func (s *Storage) Size() int {
	return s.Len() * s.dtype.Size()
}

// Len returns the number of elements in the storage.
func (s *Storage) Len() int {
	if s.dtype == tendo.Int64 {
		return len(s.int64Data)
	}
	return len(s.float32Data)
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
	if s.dtype == tendo.Int64 {
		return NewInt64StorageFromSlice(s.int64Data)
	}
	return NewStorageFromSlice(s.float32Data, s.dtype)
}

// Free releases the storage. For CPU storage, this is a no-op
// since Go's GC handles memory management.
func (s *Storage) Free() {
	s.float32Data = nil
	s.int64Data = nil
}

// Data returns the underlying float32 slice for direct access.
// Panics if the storage contains int64 data.
func (s *Storage) Data() []float32 {
	if s.dtype == tendo.Int64 {
		panic("cpu.Storage.Data() called on int64 storage; use Int64Data()")
	}
	return s.float32Data
}

// Int64Data returns the underlying int64 slice for direct access.
// Panics if the storage contains float32 data.
func (s *Storage) Int64Data() []int64 {
	if s.dtype != tendo.Int64 {
		panic("cpu.Storage.Int64Data() called on non-int64 storage; use Data()")
	}
	return s.int64Data
}

// SetData sets the value at the given index.
func (s *Storage) SetData(index int, value float32) {
	s.float32Data[index] = value
}

// GetData gets the value at the given index.
func (s *Storage) GetData(index int) float32 {
	return s.float32Data[index]
}

// Fill sets all elements to the given value.
func (s *Storage) Fill(value float32) {
	for i := range s.float32Data {
		s.float32Data[i] = value
	}
}

// Compile-time check: Storage implements tendo.Storage
var _ tendo.Storage = (*Storage)(nil)
