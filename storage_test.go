package tendo

import (
	"testing"
)

func TestDeviceType_String(t *testing.T) {
	tests := []struct {
		dt       DeviceType
		expected string
	}{
		{CPU, "cpu"},
		{CUDA, "cuda"},
		{DeviceType(99), "unknown"},
	}

	for _, tt := range tests {
		if got := tt.dt.String(); got != tt.expected {
			t.Errorf("DeviceType(%d).String() = %q, want %q", tt.dt, got, tt.expected)
		}
	}
}

func TestDevice_String(t *testing.T) {
	tests := []struct {
		device   Device
		expected string
	}{
		{CPUDevice(), "cpu"},
		{CUDADevice(0), "cuda:0"},
		{CUDADevice(1), "cuda:1"},
	}

	for _, tt := range tests {
		if got := tt.device.String(); got != tt.expected {
			t.Errorf("Device.String() = %q, want %q", got, tt.expected)
		}
	}
}

func TestDevice_IsCPU(t *testing.T) {
	if !CPUDevice().IsCPU() {
		t.Error("CPUDevice().IsCPU() should be true")
	}
	if CUDADevice(0).IsCPU() {
		t.Error("CUDADevice(0).IsCPU() should be false")
	}
}

func TestDevice_IsCUDA(t *testing.T) {
	if CPUDevice().IsCUDA() {
		t.Error("CPUDevice().IsCUDA() should be false")
	}
	if !CUDADevice(0).IsCUDA() {
		t.Error("CUDADevice(0).IsCUDA() should be true")
	}
}

func TestDType_String(t *testing.T) {
	tests := []struct {
		dt       DType
		expected string
	}{
		{Float32, "float32"},
		{Float16, "float16"},
		{BFloat16, "bfloat16"},
		{Int64, "int64"},
		{DType(99), "unknown"},
	}

	for _, tt := range tests {
		if got := tt.dt.String(); got != tt.expected {
			t.Errorf("DType(%d).String() = %q, want %q", tt.dt, got, tt.expected)
		}
	}
}

func TestDType_Size(t *testing.T) {
	tests := []struct {
		dt       DType
		expected int
	}{
		{Float32, 4},
		{Float16, 2},
		{BFloat16, 2},
		{Int64, 8},
		{DType(99), 0},
	}

	for _, tt := range tests {
		if got := tt.dt.Size(); got != tt.expected {
			t.Errorf("DType(%d).Size() = %d, want %d", tt.dt, got, tt.expected)
		}
	}
}

func TestCPUStorage_NewAndBasics(t *testing.T) {
	s := NewCPUStorage(100, Float32)

	if s.Len() != 100 {
		t.Errorf("Len() = %d, want 100", s.Len())
	}

	if s.Size() != 400 {
		t.Errorf("Size() = %d, want 400", s.Size())
	}

	if s.DType() != Float32 {
		t.Errorf("DType() = %v, want Float32", s.DType())
	}

	if !s.Device().IsCPU() {
		t.Error("Device() should be CPU")
	}

	if s.Ptr() == 0 {
		t.Error("Ptr() should not be 0")
	}
}

func TestCPUStorage_Int64(t *testing.T) {
	s := NewCPUStorage(50, Int64)

	if s.Len() != 50 {
		t.Errorf("Len() = %d, want 50", s.Len())
	}

	if s.Size() != 400 {
		t.Errorf("Size() = %d, want 400 (50 * 8)", s.Size())
	}

	if s.DType() != Int64 {
		t.Errorf("DType() = %v, want Int64", s.DType())
	}
}

func TestCPUStorage_FromSlice(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5}
	s := NewCPUStorageFromSlice(data, Float32)

	if s.Len() != 5 {
		t.Errorf("Len() = %d, want 5", s.Len())
	}

	// Verify data is copied
	data[0] = 999
	if s.Data()[0] == 999 {
		t.Error("Data should be copied, not referenced")
	}
}

func TestCPUStorage_Int64FromSlice(t *testing.T) {
	data := []int64{1, 2, 3, 4, 5}
	s := NewCPUInt64StorageFromSlice(data)

	if s.Len() != 5 {
		t.Errorf("Len() = %d, want 5", s.Len())
	}

	if s.DType() != Int64 {
		t.Errorf("DType() = %v, want Int64", s.DType())
	}

	// Verify data is copied
	data[0] = 999
	if s.Int64Data()[0] == 999 {
		t.Error("Data should be copied, not referenced")
	}
}

func TestCPUStorage_Clone(t *testing.T) {
	original := NewCPUStorageFromSlice([]float32{1, 2, 3}, Float32)
	cloned := original.Clone()

	cpuCloned, ok := cloned.(*CPUStorage)
	if !ok {
		t.Fatal("Clone should return *CPUStorage")
	}

	if cpuCloned.Len() != original.Len() {
		t.Errorf("Cloned Len() = %d, want %d", cpuCloned.Len(), original.Len())
	}

	// Verify independence
	original.SetData(0, 999)
	if cpuCloned.GetData(0) == 999 {
		t.Error("Clone should be independent of original")
	}
}

func TestCPUStorage_Fill(t *testing.T) {
	s := NewCPUStorage(5, Float32)
	s.Fill(3.14)

	for i := 0; i < s.Len(); i++ {
		if s.GetData(i) != 3.14 {
			t.Errorf("Data[%d] = %v, want 3.14", i, s.GetData(i))
		}
	}
}

func TestCPUStorage_SetGetData(t *testing.T) {
	s := NewCPUStorage(10, Float32)
	s.SetData(5, 42.0)

	if s.GetData(5) != 42.0 {
		t.Errorf("GetData(5) = %v, want 42.0", s.GetData(5))
	}
}

func TestCPUStorage_Free(t *testing.T) {
	s := NewCPUStorage(100, Float32)
	s.Free()

	if s.Data() != nil {
		t.Error("Data should be nil after Free")
	}
}

func TestCPUStorage_PoolKey(t *testing.T) {
	s := NewCPUStorage(100, Float32)
	numel, dtype, deviceIndex := s.PoolKey()

	if numel != 100 {
		t.Errorf("PoolKey numel = %d, want 100", numel)
	}
	if dtype != Float32 {
		t.Errorf("PoolKey dtype = %v, want Float32", dtype)
	}
	if deviceIndex != 0 {
		t.Errorf("PoolKey deviceIndex = %d, want 0", deviceIndex)
	}
}

func TestCPUStorage_EmptySlice(t *testing.T) {
	s := NewCPUStorage(0, Float32)

	if s.Len() != 0 {
		t.Errorf("Len() = %d, want 0", s.Len())
	}

	if s.Ptr() != 0 {
		t.Error("Ptr() should be 0 for empty storage")
	}
}

func TestBackendError(t *testing.T) {
	err := &BackendError{Message: "test error"}

	if err.Error() != "test error" {
		t.Errorf("Error() = %q, want %q", err.Error(), "test error")
	}

	// Test Is method
	other := &BackendError{Message: "test error"}
	if !err.Is(other) {
		t.Error("Is() should return true for matching message")
	}

	different := &BackendError{Message: "different"}
	if err.Is(different) {
		t.Error("Is() should return false for different message")
	}
}

func TestCUDAError(t *testing.T) {
	err := &CUDAError{Message: "allocation failed", Code: 2}

	expected := "CUDA error 2: allocation failed"
	if err.Error() != expected {
		t.Errorf("Error() = %q, want %q", err.Error(), expected)
	}

	errNoCode := &CUDAError{Message: "generic error"}
	expectedNoCode := "CUDA error: generic error"
	if errNoCode.Error() != expectedNoCode {
		t.Errorf("Error() = %q, want %q", errNoCode.Error(), expectedNoCode)
	}
}

func TestRegisterBackend(t *testing.T) {
	// This tests the registration mechanism exists
	// Actual backend registration is done by cpu/cuda packages
	_, ok := GetBackend(CPU)
	// May or may not be registered depending on test order
	_ = ok
}

func TestRegisterMemoryAllocator(t *testing.T) {
	// Test that the mechanism exists
	_, ok := GetMemoryAllocator(CPU)
	_ = ok
}
