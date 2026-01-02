package tendo

// MockStorage is a minimal storage implementation for testing.
// It tracks metadata but does not allocate real memory.
// Operations are no-ops; use this for testing shape logic and pipz integration.
type MockStorage struct {
	len    int
	device Device
	dtype  DType
}

// NewMockStorage creates a mock storage with the given parameters.
func NewMockStorage(numel int, dtype DType, device Device) *MockStorage {
	return &MockStorage{
		len:    numel,
		device: device,
		dtype:  dtype,
	}
}

// Ptr returns 0 (no real memory).
func (s *MockStorage) Ptr() uintptr {
	return 0
}

// Size returns the theoretical size in bytes.
func (s *MockStorage) Size() int {
	return s.len * s.dtype.Size()
}

// Len returns the number of elements.
func (s *MockStorage) Len() int {
	return s.len
}

// Device returns the mock device.
func (s *MockStorage) Device() Device {
	return s.device
}

// DType returns the data type.
func (s *MockStorage) DType() DType {
	return s.dtype
}

// Clone creates a copy of the mock storage.
func (s *MockStorage) Clone() Storage {
	return &MockStorage{
		len:    s.len,
		device: s.device,
		dtype:  s.dtype,
	}
}

// Free is a no-op for mock storage.
func (s *MockStorage) Free() {}

// Compile-time check: MockStorage implements Storage.
var _ Storage = (*MockStorage)(nil)
