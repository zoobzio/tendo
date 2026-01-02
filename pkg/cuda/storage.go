package cuda

import (
	"unsafe"

	"github.com/zoobzio/tendo"
)

// Storage implements tendo.Storage for CUDA device memory.
type Storage struct {
	ptr    uintptr
	size   int // size in bytes
	len    int // number of elements
	device int
	dtype  tendo.DType
}

// NewStorage allocates new CUDA storage.
// Returns an error if CUDA is not available.
func NewStorage(numel int, dtype tendo.DType, device int) (*Storage, error) {
	if !IsCUDAAvailable() {
		return nil, ErrCUDANotAvailable
	}

	if err := cudaSetDevice(device); err != nil {
		return nil, err
	}

	size := numel * dtypeSize(dtype)

	ptr, err := cudaMalloc(size)
	if err != nil {
		return nil, err
	}

	return &Storage{
		ptr:    ptr,
		size:   size,
		len:    numel,
		device: device,
		dtype:  dtype,
	}, nil
}

// NewStorageFromSlice creates CUDA storage initialized with host data.
func NewStorageFromSlice(data []float32, dtype tendo.DType, device int) (*Storage, error) {
	storage, err := NewStorage(len(data), dtype, device)
	if err != nil {
		return nil, err
	}

	if err := storage.CopyFromHost(data); err != nil {
		storage.Free()
		return nil, err
	}

	return storage, nil
}

// Ptr returns the device pointer.
func (s *Storage) Ptr() uintptr {
	return s.ptr
}

// Size returns the size in bytes.
func (s *Storage) Size() int {
	return s.size
}

// Len returns the number of elements.
func (s *Storage) Len() int {
	return s.len
}

// Device returns the CUDA device.
func (s *Storage) Device() tendo.Device {
	return tendo.Device{Type: tendo.CUDA, Index: s.device}
}

// DType returns the data type.
func (s *Storage) DType() tendo.DType {
	return s.dtype
}

// Clone creates a copy of the storage on the same device.
func (s *Storage) Clone() tendo.Storage {
	newStorage, err := NewStorage(s.len, s.dtype, s.device)
	if err != nil {
		return nil
	}

	// Copy data device-to-device (both pointers are device memory, use uintptr version)
	err = cudaMemcpyPtr(newStorage.ptr, s.ptr, s.size, cudaMemcpyDeviceToDevice)
	if err != nil {
		newStorage.Free()
		return nil
	}

	return newStorage
}

// Free releases the storage memory.
func (s *Storage) Free() {
	if s.ptr == 0 {
		return
	}

	_ = cudaFree(s.ptr)
	s.ptr = 0
}

// CopyFromHost copies data from a host slice to the device.
// For Float16/BFloat16 storage, data is converted from float32.
func (s *Storage) CopyFromHost(data []float32) error {
	if len(data) > s.len {
		return &tendo.ShapeError{Op: "CopyFromHost", Message: "data too large for storage"}
	}
	if len(data) == 0 {
		return nil
	}

	switch s.dtype {
	case tendo.Float32:
		src := unsafe.Pointer(&data[0])
		size := len(data) * 4
		return cudaMemcpy(unsafe.Pointer(s.ptr), src, size, cudaMemcpyHostToDevice)
	case tendo.Float16:
		converted := tendo.Float32SliceToFloat16(data)
		src := unsafe.Pointer(&converted[0])
		size := len(data) * 2
		return cudaMemcpy(unsafe.Pointer(s.ptr), src, size, cudaMemcpyHostToDevice)
	case tendo.BFloat16:
		converted := tendo.Float32SliceToBFloat16(data)
		src := unsafe.Pointer(&converted[0])
		size := len(data) * 2
		return cudaMemcpy(unsafe.Pointer(s.ptr), src, size, cudaMemcpyHostToDevice)
	default:
		return &tendo.ShapeError{Op: "CopyFromHost", Message: "unsupported dtype"}
	}
}

// CopyToHost copies data from the device to a host slice.
// For Float16/BFloat16 storage, data is converted to float32.
func (s *Storage) CopyToHost() ([]float32, error) {
	if s.len == 0 {
		return make([]float32, 0), nil
	}

	switch s.dtype {
	case tendo.Float32:
		result := make([]float32, s.len)
		dst := unsafe.Pointer(&result[0])
		err := cudaMemcpy(dst, unsafe.Pointer(s.ptr), s.size, cudaMemcpyDeviceToHost)
		if err != nil {
			return nil, err
		}
		return result, nil

	case tendo.Float16:
		raw := make([]uint16, s.len)
		dst := unsafe.Pointer(&raw[0])
		err := cudaMemcpy(dst, unsafe.Pointer(s.ptr), s.size, cudaMemcpyDeviceToHost)
		if err != nil {
			return nil, err
		}
		return tendo.Float16SliceToFloat32(raw), nil

	case tendo.BFloat16:
		raw := make([]uint16, s.len)
		dst := unsafe.Pointer(&raw[0])
		err := cudaMemcpy(dst, unsafe.Pointer(s.ptr), s.size, cudaMemcpyDeviceToHost)
		if err != nil {
			return nil, err
		}
		return tendo.BFloat16SliceToFloat32(raw), nil

	default:
		return nil, &tendo.ShapeError{Op: "CopyToHost", Message: "unsupported dtype"}
	}
}

// CopyToHostSlice copies data from the device to an existing host slice.
// For Float16/BFloat16 storage, data is converted to float32.
func (s *Storage) CopyToHostSlice(dst []float32) error {
	if len(dst) < s.len {
		return &tendo.ShapeError{Op: "CopyToHostSlice", Message: "destination slice too small"}
	}
	if s.len == 0 {
		return nil
	}

	switch s.dtype {
	case tendo.Float32:
		dstPtr := unsafe.Pointer(&dst[0])
		return cudaMemcpy(dstPtr, unsafe.Pointer(s.ptr), s.size, cudaMemcpyDeviceToHost)

	case tendo.Float16:
		raw := make([]uint16, s.len)
		rawPtr := unsafe.Pointer(&raw[0])
		if err := cudaMemcpy(rawPtr, unsafe.Pointer(s.ptr), s.size, cudaMemcpyDeviceToHost); err != nil {
			return err
		}
		for i, v := range raw {
			dst[i] = tendo.Float16ToFloat32(v)
		}
		return nil

	case tendo.BFloat16:
		raw := make([]uint16, s.len)
		rawPtr := unsafe.Pointer(&raw[0])
		if err := cudaMemcpy(rawPtr, unsafe.Pointer(s.ptr), s.size, cudaMemcpyDeviceToHost); err != nil {
			return err
		}
		for i, v := range raw {
			dst[i] = tendo.BFloat16ToFloat32(v)
		}
		return nil

	default:
		return &tendo.ShapeError{Op: "CopyToHostSlice", Message: "unsupported dtype"}
	}
}

// Zero fills the storage with zeros.
func (s *Storage) Zero() error {
	return cudaMemset(s.ptr, 0, s.size)
}

// Fill fills the storage with a scalar value.
// Note: This requires copying from host since cudaMemset only works with bytes.
func (s *Storage) Fill(value float32) error {
	data := make([]float32, s.len)
	for i := range data {
		data[i] = value
	}
	return s.CopyFromHost(data)
}

// Data returns the device pointer as a uintptr.
// Note: This is NOT directly usable from Go - use CopyToHost() to get data.
func (s *Storage) Data() uintptr {
	return s.ptr
}

// Sync ensures all pending operations on this storage are complete.
func (s *Storage) Sync() error {
	if err := cudaSetDevice(s.device); err != nil {
		return err
	}
	return cudaDeviceSynchronize()
}

// DeviceIndex returns the device index this storage is on.
func (s *Storage) DeviceIndex() int {
	return s.device
}

// PoolKey returns the metadata needed for pool operations.
func (s *Storage) PoolKey() (numel int, dtype tendo.DType, deviceIndex int) {
	return s.len, s.dtype, s.device
}

// MakeContiguous creates a contiguous copy of strided tensor data.
// Implements tendo.ContiguousMaker.
func (s *Storage) MakeContiguous(shape, stride []int, offset int) (tendo.Storage, error) {
	// Create temporary tensor wrapper for the operation
	tempTensor := tendo.NewTensor(s, shape, stride)
	// Note: tempTensor.offset is 0, we pass offset separately

	result, err := MakeContiguousWithOffset(tempTensor, offset)
	if err != nil {
		return nil, err
	}

	return result.Storage(), nil
}

// dtypeSize returns the size in bytes for a dtype.
func dtypeSize(dtype tendo.DType) int {
	switch dtype {
	case tendo.Float32:
		return 4
	case tendo.Float16, tendo.BFloat16:
		return 2
	default:
		return 4
	}
}

// Compile-time checks
var (
	_ tendo.Storage         = (*Storage)(nil)
	_ tendo.PoolableStorage = (*Storage)(nil)
	_ tendo.ContiguousMaker = (*Storage)(nil)
)
