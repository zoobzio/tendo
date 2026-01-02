// Package tendo provides a composable tensor library for Go with GPU acceleration.
//
// Tendo builds on pipz for operation composition and capitan for observability,
// enabling construction of neural network forward passes as pipz pipelines
// with automatic event emission for downstream autograd integration.
package tendo

import (
	"strconv"
	"unsafe"
)

// DeviceType represents the type of compute device.
type DeviceType uint8

const (
	// CPU represents a CPU device.
	CPU DeviceType = iota
	// CUDA represents an NVIDIA GPU device.
	CUDA
)

// String returns the string representation of the device type.
func (d DeviceType) String() string {
	switch d {
	case CPU:
		return "cpu"
	case CUDA:
		return "cuda"
	default:
		return "unknown"
	}
}

// Device represents a compute device (CPU or GPU).
type Device struct {
	Type  DeviceType
	Index int
}

// CPUDevice returns the default CPU device.
func CPUDevice() Device {
	return Device{Type: CPU, Index: 0}
}

// CUDADevice returns a CUDA device with the given index.
func CUDADevice(index int) Device {
	return Device{Type: CUDA, Index: index}
}

// String returns the string representation of the device.
func (d Device) String() string {
	switch d.Type {
	case CPU:
		return "cpu"
	case CUDA:
		return "cuda:" + strconv.Itoa(d.Index)
	default:
		return "unknown:" + strconv.Itoa(d.Index)
	}
}

// IsCPU returns true if this is a CPU device.
func (d Device) IsCPU() bool {
	return d.Type == CPU
}

// IsCUDA returns true if this is a CUDA device.
func (d Device) IsCUDA() bool {
	return d.Type == CUDA
}

// DType represents the data type of tensor elements.
type DType uint8

const (
	// Float32 is 32-bit floating point.
	Float32 DType = iota
	// Float16 is 16-bit floating point.
	Float16
	// BFloat16 is brain floating point (16-bit with 8-bit exponent).
	BFloat16
	// Int64 is 64-bit signed integer, used for indices.
	// Note: On CPU, Int64 is stored as float32 internally for simplicity.
	// Index precision is exact up to 16,777,216 (2^24).
	Int64
)

// String returns the string representation of the dtype.
func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	case BFloat16:
		return "bfloat16"
	case Int64:
		return "int64"
	default:
		return "unknown"
	}
}

// Size returns the size in bytes of one element of this dtype.
func (d DType) Size() int {
	switch d {
	case Float32:
		return 4
	case Float16, BFloat16:
		return 2
	case Int64:
		return 8
	default:
		return 0
	}
}

// Storage is the interface for tensor data storage backends.
// Implementations handle memory allocation and device-specific operations.
type Storage interface {
	// Ptr returns the raw pointer to the data (CPU or GPU address).
	Ptr() uintptr

	// Size returns the size in bytes of the allocated memory.
	Size() int

	// Len returns the number of elements in the storage.
	Len() int

	// Device returns the device where this storage resides.
	Device() Device

	// DType returns the data type of elements in this storage.
	DType() DType

	// Clone creates a deep copy of the storage.
	Clone() Storage

	// Free releases the underlying memory.
	// After calling Free, the storage should not be used.
	Free()
}

// CPUDataAccessor is an interface for CPU storage types that provide direct data access.
// This allows helper functions to work with different CPU storage implementations.
type CPUDataAccessor interface {
	// Data returns the underlying float32 slice for direct access.
	Data() []float32

	// SetData sets the value at the given index.
	SetData(index int, value float32)

	// GetData gets the value at the given index.
	GetData(index int) float32

	// Fill sets all elements to the given value.
	Fill(value float32)
}

// Filler is an interface for storage types that support filling with a value.
type Filler interface {
	Fill(value float32) error
}

// Zeroer is an interface for storage types that support zeroing.
type Zeroer interface {
	Zero() error
}

// HostCopier is an interface for storage types that can copy data to host memory.
type HostCopier interface {
	CopyToHost() ([]float32, error)
}

// PoolableStorage extends Storage with metadata needed for memory pool operations.
// Storage implementations that support pooling should implement this interface.
type PoolableStorage interface {
	Storage
	// PoolKey returns the metadata needed for pool free operations.
	PoolKey() (numel int, dtype DType, deviceIndex int)
}

// Syncer is an interface for storage types that support device synchronization.
type Syncer interface {
	Sync() error
}

// ContiguousMaker creates contiguous copies of strided tensor data.
// Storage types that can perform strided-to-contiguous copies on their
// native device should implement this interface.
type ContiguousMaker interface {
	// MakeContiguous copies strided data to new contiguous storage.
	// Returns storage containing only the logical elements.
	MakeContiguous(shape, stride []int, offset int) (Storage, error)
}

// CUDAError represents a CUDA-related error.
type CUDAError struct {
	Message string
	Code    int
}

func (e *CUDAError) Error() string {
	if e.Code != 0 {
		return "CUDA error " + strconv.Itoa(e.Code) + ": " + e.Message
	}
	return "CUDA error: " + e.Message
}

// CPUStorage stores tensor data in CPU memory.
// This is the core CPU storage implementation used by internal operations.
type CPUStorage struct {
	data  []float32
	dtype DType
}

// NewCPUStorage creates a new CPU storage with the given number of elements.
func NewCPUStorage(numel int, dtype DType) *CPUStorage {
	return &CPUStorage{
		data:  make([]float32, numel),
		dtype: dtype,
	}
}

// NewCPUStorageFromSlice creates a CPU storage from an existing slice.
// The slice is copied, not referenced.
func NewCPUStorageFromSlice(data []float32, dtype DType) *CPUStorage {
	copied := make([]float32, len(data))
	copy(copied, data)
	return &CPUStorage{
		data:  copied,
		dtype: dtype,
	}
}

// Ptr returns the raw pointer to the data.
func (s *CPUStorage) Ptr() uintptr {
	if len(s.data) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&s.data[0]))
}

// Size returns the size in bytes of the allocated memory.
func (s *CPUStorage) Size() int {
	return len(s.data) * s.dtype.Size()
}

// Len returns the number of elements in the storage.
func (s *CPUStorage) Len() int {
	return len(s.data)
}

// Device returns the CPU device.
func (s *CPUStorage) Device() Device {
	return CPUDevice()
}

// DType returns the data type.
func (s *CPUStorage) DType() DType {
	return s.dtype
}

// Clone creates a deep copy of the storage.
func (s *CPUStorage) Clone() Storage {
	return NewCPUStorageFromSlice(s.data, s.dtype)
}

// Free releases the storage. For CPU storage, this is a no-op
// since Go's GC handles memory management.
func (s *CPUStorage) Free() {
	s.data = nil
}

// Data returns the underlying float32 slice for direct access.
func (s *CPUStorage) Data() []float32 {
	return s.data
}

// SetData sets the value at the given index.
func (s *CPUStorage) SetData(index int, value float32) {
	s.data[index] = value
}

// GetData gets the value at the given index.
func (s *CPUStorage) GetData(index int) float32 {
	return s.data[index]
}

// Fill sets all elements to the given value.
func (s *CPUStorage) Fill(value float32) {
	for i := range s.data {
		s.data[i] = value
	}
}

// PoolKey returns the metadata needed for pool operations.
func (s *CPUStorage) PoolKey() (numel int, dtype DType, deviceIndex int) {
	return len(s.data), s.dtype, 0
}

// Compile-time checks.
var (
	_ Storage         = (*CPUStorage)(nil)
	_ CPUDataAccessor = (*CPUStorage)(nil)
	_ PoolableStorage = (*CPUStorage)(nil)
)

// Common errors.
var (
	// ErrNoBackend is returned when a required backend is not available.
	ErrNoBackend = &BackendError{Message: "backend not available"}
	// ErrCUDANotAvailable is returned when CUDA operations are attempted without CUDA support.
	ErrCUDANotAvailable = &BackendError{Message: "CUDA not available: no CUDA devices found"}
)

// BackendError represents a backend-related error.
type BackendError struct {
	Message string
}

func (e *BackendError) Error() string {
	return e.Message
}

// Is reports whether target matches this BackendError by message.
func (e *BackendError) Is(target error) bool {
	t, ok := target.(*BackendError)
	if !ok {
		return false
	}
	return e.Message == t.Message
}

// MemoryAllocator is the interface for backend-specific memory allocation.
// This is used by the memory pool for device memory management.
type MemoryAllocator interface {
	// Malloc allocates memory of the given size in bytes.
	Malloc(bytes int) (uintptr, error)
	// Free releases previously allocated memory.
	Free(ptr uintptr) error
}

// memoryAllocators holds registered memory allocators per device type.
var memoryAllocators = make(map[DeviceType]MemoryAllocator)

// RegisterMemoryAllocator registers a memory allocator for a device type.
func RegisterMemoryAllocator(dt DeviceType, alloc MemoryAllocator) {
	memoryAllocators[dt] = alloc
}

// GetMemoryAllocator returns the memory allocator for a device type.
func GetMemoryAllocator(dt DeviceType) (MemoryAllocator, bool) {
	alloc, ok := memoryAllocators[dt]
	return alloc, ok
}

// BackendRegistry holds registered backends for storage operations.
// Note: This is kept for backwards compatibility during CUDA/ROCm backend updates.
// New code should inject backends directly rather than using this registry.
type backendRegistry struct {
	backends map[DeviceType]StorageBackend
}

var globalBackends = &backendRegistry{
	backends: make(map[DeviceType]StorageBackend),
}

// StorageBackend is a minimal interface for storage creation.
// This is used during the transition period for backwards compatibility.
type StorageBackend interface {
	DeviceType() DeviceType
	NewStorage(numel int, dtype DType, deviceIndex int) (Storage, error)
	NewStorageFromSlice(data []float32, dtype DType, deviceIndex int) (Storage, error)
}

// RegisterBackend registers a storage backend.
func RegisterBackend(b StorageBackend) {
	globalBackends.backends[b.DeviceType()] = b
}

// GetBackend returns a storage backend by device type.
func GetBackend(dt DeviceType) (StorageBackend, bool) {
	b, ok := globalBackends.backends[dt]
	return b, ok
}
