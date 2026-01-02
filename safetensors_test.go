package tendo

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// closeSafeTensors is a helper to close SafeTensors files in tests.
func closeSafeTensors(t *testing.T, f *SafeTensorsFile) {
	t.Helper()
	if err := f.Close(); err != nil {
		t.Errorf("close safetensors: %v", err)
	}
}

// createTestSafeTensors creates a minimal SafeTensors file for testing.
func createTestSafeTensors(t *testing.T, tensors map[string]testTensor) string {
	t.Helper()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	// Build header and data
	header := "{"
	var dataBytes []byte
	first := true

	for name, tensor := range tensors {
		if !first {
			header += ","
		}
		first = false

		startOffset := len(dataBytes)
		tensorData := encodeTensorData(tensor.dtype, tensor.data)
		dataBytes = append(dataBytes, tensorData...)
		endOffset := len(dataBytes)

		// Build shape string
		shapeStr := "["
		for i, dim := range tensor.shape {
			if i > 0 {
				shapeStr += ","
			}
			shapeStr += string(rune('0') + rune(dim%10))
			if dim >= 10 {
				shapeStr = shapeStr[:len(shapeStr)-1]
				shapeStr += intToStr(dim)
			}
		}
		shapeStr += "]"

		header += `"` + name + `":{"dtype":"` + tensor.dtype + `","shape":` + shapeStr + `,"data_offsets":[` + intToStr(startOffset) + `,` + intToStr(endOffset) + `]}`
	}
	header += "}"

	// Write file
	osFile, err := os.Create(path) //nolint:gosec // test creates file in temp dir
	if err != nil {
		t.Fatalf("create test file: %v", err)
	}
	defer osFile.Close() //nolint:errcheck // test file cleanup

	// Write header length
	headerLen := uint64(len(header))
	if err := binary.Write(osFile, binary.LittleEndian, headerLen); err != nil {
		t.Fatalf("write header length: %v", err)
	}

	// Write header
	if _, err := osFile.WriteString(header); err != nil {
		t.Fatalf("write header: %v", err)
	}

	// Write data
	if _, err := osFile.Write(dataBytes); err != nil {
		t.Fatalf("write data: %v", err)
	}

	return path
}

type testTensor struct {
	dtype string
	shape []int
	data  []float32
}

func intToStr(n int) string {
	if n == 0 {
		return "0"
	}
	result := ""
	for n > 0 {
		result = string(rune('0'+n%10)) + result
		n /= 10
	}
	return result
}

func encodeTensorData(dtype string, data []float32) []byte {
	switch dtype {
	case "F32":
		result := make([]byte, len(data)*4)
		for i, v := range data {
			binary.LittleEndian.PutUint32(result[i*4:], math.Float32bits(v))
		}
		return result
	case "F16":
		result := make([]byte, len(data)*2)
		for i, v := range data {
			binary.LittleEndian.PutUint16(result[i*2:], Float32ToFloat16(v))
		}
		return result
	case "BF16":
		result := make([]byte, len(data)*2)
		for i, v := range data {
			binary.LittleEndian.PutUint16(result[i*2:], Float32ToBFloat16(v))
		}
		return result
	default:
		panic("unsupported dtype: " + dtype)
	}
}

func TestOpenSafeTensors(t *testing.T) {
	path := createTestSafeTensors(t, map[string]testTensor{
		"weight": {dtype: "F32", shape: []int{2, 3}, data: []float32{1, 2, 3, 4, 5, 6}},
		"bias":   {dtype: "F32", shape: []int{3}, data: []float32{0.1, 0.2, 0.3}},
	})

	f, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer closeSafeTensors(t, f)

	if f.NumTensors() != 2 {
		t.Errorf("expected 2 tensors, got %d", f.NumTensors())
	}

	tensors := f.Tensors()
	if len(tensors) != 2 {
		t.Errorf("expected 2 tensors in map, got %d", len(tensors))
	}

	weight := tensors["weight"]
	if weight.DType != Float32 {
		t.Errorf("weight dtype: expected Float32, got %v", weight.DType)
	}
	if len(weight.Shape) != 2 || weight.Shape[0] != 2 || weight.Shape[1] != 3 {
		t.Errorf("weight shape: expected [2,3], got %v", weight.Shape)
	}
}

func TestLoadCPU_Float32(t *testing.T) {
	path := createTestSafeTensors(t, map[string]testTensor{
		"data": {dtype: "F32", shape: []int{2, 2}, data: []float32{1, 2, 3, 4}},
	})

	f, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer closeSafeTensors(t, f)

	tensor, err := f.LoadCPU("data", nil)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if tensor.DType() != Float32 {
		t.Errorf("dtype: expected Float32, got %v", tensor.DType())
	}

	shape := tensor.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Errorf("shape: expected [2,2], got %v", shape)
	}

	data, err := tensor.Data()
	if err != nil {
		t.Fatalf("data: %v", err)
	}

	expected := []float32{1, 2, 3, 4}
	for i, v := range data {
		if v != expected[i] {
			t.Errorf("data[%d]: expected %v, got %v", i, expected[i], v)
		}
	}
}

func TestLoadCPU_Float16(t *testing.T) {
	path := createTestSafeTensors(t, map[string]testTensor{
		"data": {dtype: "F16", shape: []int{4}, data: []float32{1, 2, 3, 4}},
	})

	f, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer closeSafeTensors(t, f)

	tensor, err := f.LoadCPU("data", nil)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// Native dtype should be Float16
	if tensor.DType() != Float16 {
		t.Errorf("dtype: expected Float16, got %v", tensor.DType())
	}

	data, err := tensor.Data()
	if err != nil {
		t.Fatalf("data: %v", err)
	}

	// F16 conversion may have small precision loss
	expected := []float32{1, 2, 3, 4}
	for i, v := range data {
		if math.Abs(float64(v-expected[i])) > 0.01 {
			t.Errorf("data[%d]: expected ~%v, got %v", i, expected[i], v)
		}
	}
}

func TestLoadCPU_BFloat16(t *testing.T) {
	path := createTestSafeTensors(t, map[string]testTensor{
		"data": {dtype: "BF16", shape: []int{4}, data: []float32{1, 2, 3, 4}},
	})

	f, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer closeSafeTensors(t, f)

	tensor, err := f.LoadCPU("data", nil)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if tensor.DType() != BFloat16 {
		t.Errorf("dtype: expected BFloat16, got %v", tensor.DType())
	}

	data, err := tensor.Data()
	if err != nil {
		t.Fatalf("data: %v", err)
	}

	expected := []float32{1, 2, 3, 4}
	for i, v := range data {
		if math.Abs(float64(v-expected[i])) > 0.01 {
			t.Errorf("data[%d]: expected ~%v, got %v", i, expected[i], v)
		}
	}
}

func TestLoadCPU_TargetDType(t *testing.T) {
	path := createTestSafeTensors(t, map[string]testTensor{
		"data": {dtype: "F16", shape: []int{4}, data: []float32{1, 2, 3, 4}},
	})

	f, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer closeSafeTensors(t, f)

	// Load F16 data but request F32 output
	targetDType := Float32
	tensor, err := f.LoadCPU("data", &targetDType)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if tensor.DType() != Float32 {
		t.Errorf("dtype: expected Float32 (converted), got %v", tensor.DType())
	}
}

func TestLoadCPU_NotFound(t *testing.T) {
	path := createTestSafeTensors(t, map[string]testTensor{
		"data": {dtype: "F32", shape: []int{4}, data: []float32{1, 2, 3, 4}},
	})

	f, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer closeSafeTensors(t, f)

	_, err = f.LoadCPU("nonexistent", nil)
	if err == nil {
		t.Error("expected error for nonexistent tensor")
	}
}

func TestTensorInfo(t *testing.T) {
	path := createTestSafeTensors(t, map[string]testTensor{
		"weight": {dtype: "F32", shape: []int{10, 20}, data: make([]float32, 200)},
	})

	f, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer closeSafeTensors(t, f)

	info := f.TensorInfo("weight")
	if info == nil {
		t.Fatal("expected tensor info for 'weight'")
	}

	if info.Name != "weight" {
		t.Errorf("name: expected 'weight', got %q", info.Name)
	}
	if info.DType != Float32 {
		t.Errorf("dtype: expected Float32, got %v", info.DType)
	}
	if len(info.Shape) != 2 || info.Shape[0] != 10 || info.Shape[1] != 20 {
		t.Errorf("shape: expected [10,20], got %v", info.Shape)
	}

	// Nonexistent tensor
	if f.TensorInfo("nonexistent") != nil {
		t.Error("expected nil for nonexistent tensor")
	}
}
