//go:build !unix

package safetensors

import (
	"os"
)

// mmapFile returns nil on non-Unix platforms (fallback to file reads).
func mmapFile(f *os.File, size int64) ([]byte, error) {
	return nil, nil
}

// munmapFile is a no-op on non-Unix platforms.
func munmapFile(data []byte) error {
	return nil
}
