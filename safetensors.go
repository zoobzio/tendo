package tendo

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/zoobzio/tendo/internal/safetensors"
)

// SafeTensorsFile represents an opened SafeTensors file.
type SafeTensorsFile struct {
	internal *safetensors.File
}

// SafeTensorInfo contains metadata about a tensor in a SafeTensors file.
type SafeTensorInfo struct { //nolint:govet // field alignment is less important than readability
	Name  string
	DType DType
	Shape []int
}

// OpenSafeTensors opens a SafeTensors file for reading.
func OpenSafeTensors(path string) (*SafeTensorsFile, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	return &SafeTensorsFile{internal: f}, nil
}

// Tensors returns metadata for all tensors in the file.
// Tensors with unsupported dtypes are omitted from the result.
func (f *SafeTensorsFile) Tensors() map[string]SafeTensorInfo {
	internal := f.internal.Tensors()
	result := make(map[string]SafeTensorInfo, len(internal))
	for name, info := range internal {
		dtype, err := safeTensorsDTypeToTendo(info.DType)
		if err != nil {
			continue // skip unsupported dtypes
		}
		result[name] = SafeTensorInfo{
			Name:  name,
			DType: dtype,
			Shape: info.Shape,
		}
	}
	return result
}

// TensorInfo returns metadata for a specific tensor, or nil if not found
// or if the dtype is unsupported.
func (f *SafeTensorsFile) TensorInfo(name string) *SafeTensorInfo {
	info := f.internal.TensorInfo(name)
	if info == nil {
		return nil
	}
	dtype, err := safeTensorsDTypeToTendo(info.DType)
	if err != nil {
		return nil // unsupported dtype
	}
	return &SafeTensorInfo{
		Name:  name,
		DType: dtype,
		Shape: info.Shape,
	}
}

// Load reads a tensor from the file and creates a tendo Tensor using the provided backend.
// If targetDType is nil, the native file dtype is preserved.
// If targetDType is specified, the storage will use that dtype marker.
func (f *SafeTensorsFile) Load(name string, backend StorageBackend, targetDType *DType) (*Tensor, error) {
	// Read raw bytes
	data, info, err := f.internal.ReadTensorBytes(name)
	if err != nil {
		return nil, err
	}

	// Convert file dtype to tendo dtype
	fileDType, err := safeTensorsDTypeToTendo(info.DType)
	if err != nil {
		return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
	}

	// Determine output dtype
	outDType := fileDType
	if targetDType != nil {
		outDType = *targetDType
	}

	// Convert bytes to float32 slice (our common interchange format)
	float32Data, err := bytesToFloat32(data, info.DType)
	if err != nil {
		return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
	}

	// Create storage via backend
	storage, err := backend.NewStorageFromSlice(float32Data, outDType, 0)
	if err != nil {
		return nil, fmt.Errorf("safetensors: create storage for %q: %w", name, err)
	}

	return NewTensor(storage, info.Shape, nil), nil
}

// LoadCPU is a convenience method that loads a tensor to CPU storage.
// If targetDType is nil, the native file dtype is preserved.
func (f *SafeTensorsFile) LoadCPU(name string, targetDType *DType) (*Tensor, error) {
	// Read raw bytes
	data, info, err := f.internal.ReadTensorBytes(name)
	if err != nil {
		return nil, err
	}

	// Convert file dtype to tendo dtype
	fileDType, err := safeTensorsDTypeToTendo(info.DType)
	if err != nil {
		return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
	}

	// Determine output dtype
	outDType := fileDType
	if targetDType != nil {
		outDType = *targetDType
	}

	// Convert bytes to float32 slice
	float32Data, err := bytesToFloat32(data, info.DType)
	if err != nil {
		return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
	}

	storage := NewCPUStorageFromSlice(float32Data, outDType)
	return NewTensor(storage, info.Shape, nil), nil
}

// Close closes the file.
func (f *SafeTensorsFile) Close() error {
	return f.internal.Close()
}

// Path returns the file path.
func (f *SafeTensorsFile) Path() string {
	return f.internal.Path()
}

// NumTensors returns the number of tensors in the file.
func (f *SafeTensorsFile) NumTensors() int {
	return f.internal.NumTensors()
}

// SafeTensors dtype string constants.
const (
	stDTypeF32  = "F32"
	stDTypeF16  = "F16"
	stDTypeBF16 = "BF16"
	stDTypeI64  = "I64"
)

// safeTensorsDTypeToTendo converts a SafeTensors dtype string to tendo DType.
func safeTensorsDTypeToTendo(dtype string) (DType, error) {
	switch dtype {
	case stDTypeF32:
		return Float32, nil
	case stDTypeF16:
		return Float16, nil
	case stDTypeBF16:
		return BFloat16, nil
	case stDTypeI64:
		return Int64, nil
	default:
		return 0, fmt.Errorf("unsupported dtype: %s", dtype)
	}
}

// bytesToFloat32 converts raw bytes from a SafeTensors file to float32 slice.
func bytesToFloat32(data []byte, dtype string) ([]float32, error) {
	switch dtype {
	case stDTypeF32:
		return bytesToFloat32Direct(data), nil
	case stDTypeF16:
		return bytesF16ToFloat32(data), nil
	case stDTypeBF16:
		return bytesBF16ToFloat32(data), nil
	case stDTypeI64:
		return bytesI64ToFloat32(data), nil
	default:
		return nil, fmt.Errorf("unsupported dtype for conversion: %s", dtype)
	}
}

// bytesToFloat32Direct reinterprets bytes as float32 slice.
func bytesToFloat32Direct(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	n := len(data) / 4
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		result[i] = math.Float32frombits(bits)
	}
	return result
}

// bytesF16ToFloat32 converts float16 bytes to float32 slice.
func bytesF16ToFloat32(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	n := len(data) / 2
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2:])
		result[i] = Float16ToFloat32(bits)
	}
	return result
}

// bytesBF16ToFloat32 converts bfloat16 bytes to float32 slice.
func bytesBF16ToFloat32(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	n := len(data) / 2
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2:])
		result[i] = BFloat16ToFloat32(bits)
	}
	return result
}

// bytesI64ToFloat32 converts int64 bytes to float32 slice.
// Note: This may lose precision for large integers.
func bytesI64ToFloat32(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	n := len(data) / 8
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		val := int64(binary.LittleEndian.Uint64(data[i*8:])) //nolint:gosec // converting raw bytes from file
		result[i] = float32(val)
	}
	return result
}

