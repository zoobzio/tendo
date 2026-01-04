package cuda

import (
	"sync"

	"github.com/zoobzio/tendo"
)

// PoolAllocator implements tendo.PoolAllocator for CUDA memory.
type PoolAllocator struct {
	mu          sync.Mutex
	device      int
	blocks      map[int][]poolBlock // sizeClass -> available blocks
	totalCached int64
}

type poolBlock struct {
	ptr   uintptr
	numel int
	dtype tendo.DType
}

// NewPoolAllocator creates a new CUDA pool allocator for a specific device.
func NewPoolAllocator(device int) *PoolAllocator {
	return &PoolAllocator{
		device: device,
		blocks: make(map[int][]poolBlock),
	}
}

// Alloc allocates or retrieves CUDA storage from the pool.
func (p *PoolAllocator) Alloc(numel int, dtype tendo.DType) (tendo.Storage, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	sizeClass := roundUpPow2(numel)
	size := sizeClass * dtypeSize(dtype)

	// Check for cached block with matching dtype
	if blocks, ok := p.blocks[sizeClass]; ok && len(blocks) > 0 {
		for i := len(blocks) - 1; i >= 0; i-- {
			if blocks[i].numel >= numel && blocks[i].dtype == dtype {
				// Remove from pool
				block := blocks[i]
				p.blocks[sizeClass] = append(blocks[:i], blocks[i+1:]...)
				p.totalCached -= int64(size)

				return &Storage{
					ptr:    block.ptr,
					size:   numel * dtypeSize(dtype),
					len:    numel,
					device: p.device,
					dtype:  dtype,
				}, nil
			}
		}
	}

	// Allocate new
	return NewStorage(numel, dtype, p.device)
}

// Free returns storage to the pool for reuse.
func (p *PoolAllocator) Free(s tendo.Storage) {
	storage, ok := s.(*Storage)
	if !ok {
		return
	}

	// Only cache if from same device
	if storage.device != p.device {
		storage.Free()
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	sizeClass := roundUpPow2(storage.len)
	size := sizeClass * dtypeSize(storage.dtype)

	p.blocks[sizeClass] = append(p.blocks[sizeClass], poolBlock{
		ptr:   storage.ptr,
		numel: storage.len,
		dtype: storage.dtype,
	})
	p.totalCached += int64(size)

	// Don't free the underlying memory - it's now in the pool
	storage.ptr = 0
}

// Clear releases all cached memory.
func (p *PoolAllocator) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, blocks := range p.blocks {
		for _, block := range blocks {
			_ = cudaFree(block.ptr)
		}
	}

	p.blocks = make(map[int][]poolBlock)
	p.totalCached = 0
}

// CachedBytes returns the total bytes currently cached in the pool.
func (p *PoolAllocator) CachedBytes() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.totalCached
}

// roundUpPow2 rounds n up to the next power of 2.
func roundUpPow2(n int) int {
	if n <= 0 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n++
	return n
}

// Compile-time check
var _ tendo.PoolAllocator = (*PoolAllocator)(nil)
