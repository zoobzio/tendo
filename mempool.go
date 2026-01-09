package tendo

import (
	"context"
	"sync"

	"github.com/zoobzio/capitan"
)

// PoolAllocator is the interface for backend-specific pool allocation.
// Each backend (CPU, CUDA, etc.) implements this interface to provide
// memory pooling for its device type.
type PoolAllocator interface {
	// Alloc allocates storage of the given size and dtype.
	Alloc(numel int, dtype DType) (Storage, error)

	// Free returns storage to the pool.
	Free(s Storage)

	// Clear releases all cached memory.
	Clear()
}

// Pool manages memory allocation and reuse for tensor storage.
// It maintains separate pools for CPU and each CUDA device.
type Pool struct {
	emitCtx       context.Context
	cpu           *cpuPool
	cuda          map[int]*cudaPool
	stats         PoolStats
	mu            sync.Mutex
	maxCacheBytes int64
}

// PoolStats tracks memory pool statistics.
type PoolStats struct {
	CUDAAllocations   map[int]int64
	CUDADeallocations map[int]int64
	CUDABytesInUse    map[int]int64
	CUDABytesCached   map[int]int64
	CPUAllocations    int64
	CPUDeallocations  int64
	CPUBytesInUse     int64
	CPUBytesCached    int64
}

// cpuPool manages CPU memory blocks.
type cpuPool struct {
	// blocks maps size class to available blocks
	// Size classes are powers of 2 for simplicity
	blocks map[int][]Storage
}

// cudaPool manages CUDA memory blocks for a single device.
type cudaPool struct {
	blocks map[int][]uintptr
	device int
}

// NewPool creates a new memory pool.
func NewPool() *Pool {
	return &Pool{
		cpu: &cpuPool{
			blocks: make(map[int][]Storage),
		},
		cuda: make(map[int]*cudaPool),
		stats: PoolStats{
			CUDAAllocations:   make(map[int]int64),
			CUDADeallocations: make(map[int]int64),
			CUDABytesInUse:    make(map[int]int64),
			CUDABytesCached:   make(map[int]int64),
		},
	}
}

// SetContext sets the context used for emitting pool events.
func (p *Pool) SetContext(ctx context.Context) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.emitCtx = ctx
}

// SetMaxCacheSize sets the maximum bytes to cache before eviction.
// Set to 0 for unlimited (default). When the cache exceeds this limit,
// oldest blocks are evicted until under the limit.
func (p *Pool) SetMaxCacheSize(bytes int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.maxCacheBytes = bytes
	if bytes > 0 {
		p.evictIfNeeded()
	}
}

// MaxCacheSize returns the current maximum cache size (0 = unlimited).
func (p *Pool) MaxCacheSize() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.maxCacheBytes
}

// totalCachedBytes returns total cached bytes (CPU + all CUDA devices).
// Caller must hold the mutex.
func (p *Pool) totalCachedBytes() int64 {
	total := p.stats.CPUBytesCached
	for _, bytes := range p.stats.CUDABytesCached {
		total += bytes
	}
	return total
}

// evictIfNeeded evicts cached blocks if over the limit.
// Caller must hold the mutex.
func (p *Pool) evictIfNeeded() {
	if p.maxCacheBytes <= 0 {
		return
	}

	// Evict CPU blocks first (FIFO from smallest size class)
	for p.totalCachedBytes() > p.maxCacheBytes && p.stats.CPUBytesCached > 0 {
		evicted := false
		for sizeClass, blocks := range p.cpu.blocks {
			if len(blocks) > 0 {
				// Remove oldest block (first in slice)
				block := blocks[0]
				p.cpu.blocks[sizeClass] = blocks[1:]
				p.stats.CPUBytesCached -= int64(block.Size())
				evicted = true
				break
			}
		}
		if !evicted {
			break
		}
	}

	// Then evict CUDA blocks if still over limit
	alloc, hasAlloc := GetMemoryAllocator(CUDA)
	for p.totalCachedBytes() > p.maxCacheBytes {
		evicted := false
		for device, pool := range p.cuda {
			for sizeClass, blocks := range pool.blocks {
				if len(blocks) > 0 {
					ptr := blocks[0]
					pool.blocks[sizeClass] = blocks[1:]
					if hasAlloc {
						_ = alloc.Free(ptr) //nolint:errcheck // Best-effort eviction, no recovery possible
					}
					// Estimate bytes freed (sizeClass * 4 for float32)
					p.stats.CUDABytesCached[device] -= int64(sizeClass * 4)
					evicted = true
					break
				}
			}
			if evicted {
				break
			}
		}
		if !evicted {
			break
		}
	}
}

// AllocCPU allocates or retrieves a CPU storage of the given size.
func (p *Pool) AllocCPU(numel int, dtype DType) *CPUStorage {
	p.mu.Lock()
	defer p.mu.Unlock()

	sizeClass := roundUpPow2(numel)

	// Check for cached block with matching dtype
	if blocks, ok := p.cpu.blocks[sizeClass]; ok && len(blocks) > 0 {
		// Search for a block with matching dtype
		for i := len(blocks) - 1; i >= 0; i-- {
			if cpu, ok := blocks[i].(*CPUStorage); ok {
				if cpu.Len() >= numel && cpu.DType() == dtype {
					// Remove from pool
					p.cpu.blocks[sizeClass] = append(blocks[:i], blocks[i+1:]...)
					p.stats.CPUBytesCached -= int64(cpu.Size())
					p.stats.CPUBytesInUse += int64(numel * dtypeSize(dtype))
					return cpu
				}
			}
		}
	}

	// Allocate new
	storage := NewCPUStorage(numel, dtype)
	p.stats.CPUAllocations++
	p.stats.CPUBytesInUse += int64(storage.Size())

	if p.emitCtx != nil {
		capitan.Emit(p.emitCtx, PoolAlloc,
			KeyBytes.Field(storage.Size()),
			KeyDevice.Field(Device{Type: CPU}),
		)
	}

	return storage
}

// FreeCPU returns a CPU storage to the pool for reuse.
func (p *Pool) FreeCPU(storage *CPUStorage) {
	p.mu.Lock()
	defer p.mu.Unlock()

	sizeClass := roundUpPow2(storage.Len())
	p.cpu.blocks[sizeClass] = append(p.cpu.blocks[sizeClass], storage)

	p.stats.CPUDeallocations++
	p.stats.CPUBytesInUse -= int64(storage.Size())
	p.stats.CPUBytesCached += int64(storage.Size())

	if p.emitCtx != nil {
		capitan.Emit(p.emitCtx, PoolFree,
			KeyBytes.Field(storage.Size()),
			KeyDevice.Field(Device{Type: CPU}),
		)
	}

	// Evict if over cache limit
	p.evictIfNeeded()
}

// AllocCUDA allocates or retrieves CUDA memory of the given size.
// Returns the device pointer and an error if CUDA is unavailable.
func (p *Pool) AllocCUDA(numel int, dtype DType, device int) (uintptr, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Ensure device pool exists
	if _, ok := p.cuda[device]; !ok {
		p.cuda[device] = &cudaPool{
			device: device,
			blocks: make(map[int][]uintptr),
		}
	}

	elemSize := dtypeSize(dtype)
	sizeClass := roundUpPow2(numel)
	pool := p.cuda[device]

	// Check for cached block
	if blocks, ok := pool.blocks[sizeClass]; ok && len(blocks) > 0 {
		ptr := blocks[len(blocks)-1]
		pool.blocks[sizeClass] = blocks[:len(blocks)-1]

		p.stats.CUDABytesCached[device] -= int64(sizeClass * elemSize)
		p.stats.CUDABytesInUse[device] += int64(numel * elemSize)
		return ptr, nil
	}

	// Allocate new via CUDA backend
	alloc, ok := GetMemoryAllocator(CUDA)
	if !ok {
		return 0, ErrNoBackend
	}
	ptr, err := alloc.Malloc(sizeClass * elemSize)
	if err != nil {
		return 0, err
	}

	p.stats.CUDAAllocations[device]++
	p.stats.CUDABytesInUse[device] += int64(numel * elemSize)

	if p.emitCtx != nil {
		capitan.Emit(p.emitCtx, PoolAlloc,
			KeyBytes.Field(numel*elemSize),
			KeyDevice.Field(Device{Type: CUDA, Index: device}),
		)
	}

	return ptr, nil
}

// FreeCUDA returns CUDA memory to the pool for reuse.
func (p *Pool) FreeCUDA(ptr uintptr, numel int, dtype DType, device int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, ok := p.cuda[device]; !ok {
		p.cuda[device] = &cudaPool{
			device: device,
			blocks: make(map[int][]uintptr),
		}
	}

	elemSize := dtypeSize(dtype)
	sizeClass := roundUpPow2(numel)
	pool := p.cuda[device]
	pool.blocks[sizeClass] = append(pool.blocks[sizeClass], ptr)

	p.stats.CUDADeallocations[device]++
	p.stats.CUDABytesInUse[device] -= int64(numel * elemSize)
	p.stats.CUDABytesCached[device] += int64(sizeClass * elemSize)

	if p.emitCtx != nil {
		capitan.Emit(p.emitCtx, PoolFree,
			KeyBytes.Field(numel*elemSize),
			KeyDevice.Field(Device{Type: CUDA, Index: device}),
		)
	}

	// Evict if over cache limit
	p.evictIfNeeded()
}

// Clear releases all cached memory back to the system.
func (p *Pool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Clear CPU cache (Go GC will handle it)
	p.cpu.blocks = make(map[int][]Storage)
	p.stats.CPUBytesCached = 0

	// Clear CUDA caches
	alloc, hasAlloc := GetMemoryAllocator(CUDA)
	for device, pool := range p.cuda {
		for _, blocks := range pool.blocks {
			for _, ptr := range blocks {
				if hasAlloc {
					_ = alloc.Free(ptr) //nolint:errcheck // Best-effort cleanup, no recovery possible
				}
			}
		}
		pool.blocks = make(map[int][]uintptr)
		p.stats.CUDABytesCached[device] = 0
	}
}

// Stats returns current pool statistics.
func (p *Pool) Stats() PoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Copy maps to avoid race
	stats := p.stats
	stats.CUDAAllocations = make(map[int]int64)
	stats.CUDADeallocations = make(map[int]int64)
	stats.CUDABytesInUse = make(map[int]int64)
	stats.CUDABytesCached = make(map[int]int64)
	for k, v := range p.stats.CUDAAllocations {
		stats.CUDAAllocations[k] = v
	}
	for k, v := range p.stats.CUDADeallocations {
		stats.CUDADeallocations[k] = v
	}
	for k, v := range p.stats.CUDABytesInUse {
		stats.CUDABytesInUse[k] = v
	}
	for k, v := range p.stats.CUDABytesCached {
		stats.CUDABytesCached[k] = v
	}
	return stats
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

// dtypeSize returns the size in bytes for a dtype.
func dtypeSize(dtype DType) int {
	switch dtype {
	case Float32:
		return 4
	case Float16, BFloat16:
		return 2
	case Int64:
		return 8
	default:
		return 4
	}
}

// Default pool instance.
var defaultPool = NewPool()

// DefaultPool returns the default memory pool.
func DefaultPool() *Pool {
	return defaultPool
}

// SetDefaultPool sets the default memory pool.
func SetDefaultPool(p *Pool) {
	defaultPool = p
}
