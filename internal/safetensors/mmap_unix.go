//go:build unix

package safetensors

import (
	"os"
	"syscall"
)

// mmapFile memory-maps the given file for reading.
func mmapFile(f *os.File, size int64) ([]byte, error) {
	if size == 0 {
		return nil, nil
	}
	data, err := syscall.Mmap(
		int(f.Fd()),
		0,
		int(size), //nolint:gosec // size validated by caller
		syscall.PROT_READ,
		syscall.MAP_SHARED,
	)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// munmapFile unmaps a memory-mapped region.
func munmapFile(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return syscall.Munmap(data)
}
