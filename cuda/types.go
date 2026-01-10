//go:build cuda

package cuda

import "errors"

// CUDAError represents a CUDA runtime error.
type CUDAError struct {
	Code    int
	Message string
}

func (e *CUDAError) Error() string {
	return e.Message
}

// ErrCUDANotAvailable indicates CUDA is not available on this system.
var ErrCUDANotAvailable = errors.New("CUDA not available: no CUDA devices found")

// CUDADeviceProperties contains properties of a CUDA device.
type CUDADeviceProperties struct {
	Name                string
	TotalGlobalMem      int64
	SharedMemPerBlock   int64
	MaxThreadsPerBlock  int
	MultiProcessorCount int
	ComputeCapability   [2]int // major, minor
}

// IsCUDAAvailable returns true if CUDA is available.
func IsCUDAAvailable() bool {
	count, err := cudaGetDeviceCount()
	return err == nil && count > 0
}

// CUDADeviceCount returns the number of available CUDA devices.
func CUDADeviceCount() int {
	count, err := cudaGetDeviceCount()
	if err != nil {
		return 0
	}
	return count
}
