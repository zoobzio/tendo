package tendo

import (
	"testing"
)

func TestPoolBasic(t *testing.T) {
	t.Run("NewPool", func(t *testing.T) {
		p := NewPool()
		if p == nil {
			t.Fatal("NewPool returned nil")
		}
		if p.MaxCacheSize() != 0 {
			t.Errorf("expected default max cache size 0, got %d", p.MaxCacheSize())
		}
	})

	t.Run("AllocCPU and FreeCPU", func(t *testing.T) {
		p := NewPool()

		// Allocate
		storage := p.AllocCPU(100, Float32)
		if storage == nil {
			t.Fatal("AllocCPU returned nil")
		}
		if storage.Len() != 100 {
			t.Errorf("expected len 100, got %d", storage.Len())
		}

		stats := p.Stats()
		if stats.CPUAllocations != 1 {
			t.Errorf("expected 1 allocation, got %d", stats.CPUAllocations)
		}

		// Free back to pool
		p.FreeCPU(storage)

		stats = p.Stats()
		if stats.CPUDeallocations != 1 {
			t.Errorf("expected 1 deallocation, got %d", stats.CPUDeallocations)
		}
		if stats.CPUBytesCached <= 0 {
			t.Error("expected cached bytes > 0")
		}

		// Reallocate should reuse
		storage2 := p.AllocCPU(100, Float32)
		if storage2 == nil {
			t.Fatal("second AllocCPU returned nil")
		}

		stats = p.Stats()
		// Should still be 1 allocation since we reused from cache
		if stats.CPUAllocations != 1 {
			t.Errorf("expected 1 allocation (reuse), got %d", stats.CPUAllocations)
		}
	})
}

func TestPoolEviction(t *testing.T) {
	t.Run("SetMaxCacheSize", func(t *testing.T) {
		p := NewPool()
		p.SetMaxCacheSize(1000)
		if p.MaxCacheSize() != 1000 {
			t.Errorf("expected max cache size 1000, got %d", p.MaxCacheSize())
		}
	})

	t.Run("Eviction on limit", func(t *testing.T) {
		p := NewPool()

		// Allocate and free several blocks
		for i := 0; i < 10; i++ {
			storage := p.AllocCPU(100, Float32) // 400 bytes each
			p.FreeCPU(storage)
		}

		stats := p.Stats()
		cachedBefore := stats.CPUBytesCached
		if cachedBefore == 0 {
			t.Fatal("expected some cached bytes before setting limit")
		}

		// Set a small limit - should trigger eviction
		p.SetMaxCacheSize(500)

		stats = p.Stats()
		cachedAfter := stats.CPUBytesCached
		if cachedAfter > 500 {
			t.Errorf("expected cached bytes <= 500 after eviction, got %d", cachedAfter)
		}
	})

	t.Run("Eviction on FreeCPU", func(t *testing.T) {
		p := NewPool()
		p.SetMaxCacheSize(500) // Set limit first

		// Allocate and free - should evict to stay under limit
		for i := 0; i < 10; i++ {
			storage := p.AllocCPU(100, Float32) // 400 bytes each, rounded to 512
			p.FreeCPU(storage)
		}

		stats := p.Stats()
		if stats.CPUBytesCached > 500 {
			t.Errorf("expected cached bytes <= 500, got %d", stats.CPUBytesCached)
		}
	})

	t.Run("Unlimited cache (0)", func(t *testing.T) {
		p := NewPool()
		p.SetMaxCacheSize(0) // Unlimited

		// Allocate and free many blocks
		for i := 0; i < 100; i++ {
			storage := p.AllocCPU(100, Float32)
			p.FreeCPU(storage)
		}

		stats := p.Stats()
		// With unlimited, nothing should be evicted
		if stats.CPUBytesCached == 0 {
			t.Error("expected cached bytes > 0 with unlimited cache")
		}
	})
}

func TestPoolClear(t *testing.T) {
	t.Run("Clear releases cache", func(t *testing.T) {
		p := NewPool()

		// Fill cache
		for i := 0; i < 10; i++ {
			storage := p.AllocCPU(100, Float32)
			p.FreeCPU(storage)
		}

		stats := p.Stats()
		if stats.CPUBytesCached == 0 {
			t.Fatal("expected cached bytes > 0 before clear")
		}

		// Clear
		p.Clear()

		stats = p.Stats()
		if stats.CPUBytesCached != 0 {
			t.Errorf("expected 0 cached bytes after clear, got %d", stats.CPUBytesCached)
		}
	})
}
