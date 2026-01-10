package cpu

import (
	"sync"

	"github.com/zoobzio/tendo"
)

// PoolAllocator implements tendo.PoolAllocator for CPU memory.
type PoolAllocator struct {
	blocks map[int][]*Storage // sizeClass -> available blocks
	mu     sync.Mutex
}

// NewPoolAllocator creates a new CPU pool allocator.
func NewPoolAllocator() *PoolAllocator {
	return &PoolAllocator{
		blocks: make(map[int][]*Storage),
	}
}

// Alloc allocates or retrieves CPU storage from the pool.
func (p *PoolAllocator) Alloc(numel int, dtype tendo.DType) (tendo.Storage, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	sizeClass := roundUpPow2(numel)

	// Check for cached block with matching dtype
	if blocks, ok := p.blocks[sizeClass]; ok && len(blocks) > 0 {
		for i := len(blocks) - 1; i >= 0; i-- {
			if blocks[i].Len() >= numel && blocks[i].DType() == dtype {
				// Remove from pool
				storage := blocks[i]
				p.blocks[sizeClass] = append(blocks[:i], blocks[i+1:]...)
				return storage, nil
			}
		}
	}

	// Allocate new
	return NewStorage(numel, dtype), nil
}

// Free returns storage to the pool for reuse.
func (p *PoolAllocator) Free(s tendo.Storage) {
	storage, ok := s.(*Storage)
	if !ok {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	sizeClass := roundUpPow2(storage.Len())
	p.blocks[sizeClass] = append(p.blocks[sizeClass], storage)
}

// Clear releases all cached memory.
func (p *PoolAllocator) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.blocks = make(map[int][]*Storage)
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

// Compile-time check.
var _ tendo.PoolAllocator = (*PoolAllocator)(nil)
