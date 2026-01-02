// Package safetensors implements parsing of the SafeTensors file format.
// SafeTensors is a simple, safe format for storing tensors created by Hugging Face.
//
// Format:
//   - 8 bytes: header length (little-endian uint64)
//   - N bytes: JSON header with tensor metadata
//   - Remaining: raw tensor data
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// TensorInfo contains metadata about a tensor in a SafeTensors file.
type TensorInfo struct {
	Name        string
	DType       string // "F16", "BF16", "F32", "F64", "I8", "I16", "I32", "I64", etc.
	Shape       []int
	DataOffsets [2]uint64 // [start, end] byte offsets into data section
}

// File represents an opened SafeTensors file.
type File struct { //nolint:govet // field alignment is less important than readability
	path       string
	file       *os.File
	tensors    map[string]*TensorInfo
	dataOffset int64  // offset where tensor data begins (after header)
	size       int64  // total file size
	mmap       []byte // memory-mapped file contents (nil if mmap unavailable)
}

// tensorEntry is the JSON structure for each tensor in the header.
type tensorEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Open opens a SafeTensors file and parses its header.
func Open(path string) (*File, error) {
	f, err := os.Open(path) //nolint:gosec // path from trusted caller
	if err != nil {
		return nil, fmt.Errorf("safetensors: open: %w", err)
	}

	// Ensure file is closed on error
	success := false
	defer func() {
		if !success {
			f.Close() //nolint:errcheck // error on cleanup is informational only
		}
	}()

	// Get file size
	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("safetensors: stat: %w", err)
	}

	// Read header length (8 bytes, little-endian uint64)
	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("safetensors: read header length: %w", err)
	}

	// Sanity check header length
	if headerLen > uint64(stat.Size())-8 { //nolint:gosec // size is always positive
		return nil, fmt.Errorf("safetensors: header length %d exceeds file size", headerLen)
	}

	// Read header JSON
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}

	// Parse header
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	// Extract tensor info
	tensors := make(map[string]*TensorInfo)
	for name, raw := range header {
		// Skip metadata entry
		if name == "__metadata__" {
			continue
		}

		var entry tensorEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}

		tensors[name] = &TensorInfo{
			Name:        name,
			DType:       entry.DType,
			Shape:       entry.Shape,
			DataOffsets: [2]uint64{uint64(entry.DataOffsets[0]), uint64(entry.DataOffsets[1])}, //nolint:gosec // offsets validated by file format
		}
	}

	success = true

	// Attempt to mmap the file for efficient tensor access
	mmapData, _ := mmapFile(f, stat.Size()) //nolint:errcheck // fallback to reads on error

	return &File{
		path:       path,
		file:       f,
		tensors:    tensors,
		dataOffset: int64(8 + headerLen), //nolint:gosec // headerLen validated above
		size:       stat.Size(),
		mmap:       mmapData,
	}, nil
}

// Tensors returns metadata for all tensors in the file.
func (f *File) Tensors() map[string]*TensorInfo {
	result := make(map[string]*TensorInfo, len(f.tensors))
	for k, v := range f.tensors {
		info := *v // copy
		result[k] = &info
	}
	return result
}

// TensorInfo returns metadata for a specific tensor, or nil if not found.
func (f *File) TensorInfo(name string) *TensorInfo {
	info, ok := f.tensors[name]
	if !ok {
		return nil
	}
	infoCopy := *info
	return &infoCopy
}

// ReadTensorBytes reads the raw bytes for a tensor.
// The caller is responsible for interpreting the bytes according to dtype.
func (f *File) ReadTensorBytes(name string) ([]byte, *TensorInfo, error) {
	info, ok := f.tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}

	start := f.dataOffset + int64(info.DataOffsets[0])         //nolint:gosec // offset from parsed header
	length := int64(info.DataOffsets[1] - info.DataOffsets[0]) //nolint:gosec // offset from parsed header

	// Use mmap if available
	if f.mmap != nil {
		end := start + length
		if end > int64(len(f.mmap)) {
			return nil, nil, fmt.Errorf("safetensors: tensor %q extends beyond file", name)
		}
		// Copy from mmap to avoid keeping reference to mapped memory
		data := make([]byte, length)
		copy(data, f.mmap[start:end])
		return data, info, nil
	}

	// Fallback to file reads
	if _, err := f.file.Seek(start, io.SeekStart); err != nil {
		return nil, nil, fmt.Errorf("safetensors: seek to tensor %q: %w", name, err)
	}

	data := make([]byte, length)
	if _, err := io.ReadFull(f.file, data); err != nil {
		return nil, nil, fmt.Errorf("safetensors: read tensor %q: %w", name, err)
	}

	return data, info, nil
}

// Close closes the file.
func (f *File) Close() error {
	// Unmap first if mmap was used
	if f.mmap != nil {
		munmapFile(f.mmap) //nolint:errcheck // best effort cleanup
		f.mmap = nil
	}
	if f.file != nil {
		err := f.file.Close()
		f.file = nil
		return err
	}
	return nil
}

// Path returns the file path.
func (f *File) Path() string {
	return f.path
}

// NumTensors returns the number of tensors in the file.
func (f *File) NumTensors() int {
	return len(f.tensors)
}

// Numel calculates the number of elements from a shape.
func Numel(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}

// DTypeSize returns the size in bytes for a SafeTensors dtype string.
func DTypeSize(dtype string) int {
	switch dtype {
	case "F16", "BF16", "I16", "U16":
		return 2
	case "F32", "I32", "U32":
		return 4
	case "F64", "I64", "U64":
		return 8
	case "I8", "U8", "BOOL":
		return 1
	default:
		return 0
	}
}
