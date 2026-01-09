package benchmarks

import (
	"context"
	"testing"

	"github.com/zoobzio/tendo"
)

func BenchmarkTensorCreation(b *testing.B) {
	b.Run("Zeros_Small", func(b *testing.B) {
		for b.Loop() {
			t, err := tendo.Zeros(64, 64)
			if err != nil {
				b.Fatal(err)
			}
			_ = t
		}
	})

	b.Run("Zeros_Medium", func(b *testing.B) {
		for b.Loop() {
			t, err := tendo.Zeros(256, 256)
			if err != nil {
				b.Fatal(err)
			}
			_ = t
		}
	})

	b.Run("Zeros_Large", func(b *testing.B) {
		for b.Loop() {
			t, err := tendo.Zeros(1024, 1024)
			if err != nil {
				b.Fatal(err)
			}
			_ = t
		}
	})

	b.Run("RandN_Small", func(b *testing.B) {
		for b.Loop() {
			t, err := tendo.RandN(64, 64)
			if err != nil {
				b.Fatal(err)
			}
			_ = t
		}
	})

	b.Run("RandN_Medium", func(b *testing.B) {
		for b.Loop() {
			t, err := tendo.RandN(256, 256)
			if err != nil {
				b.Fatal(err)
			}
			_ = t
		}
	})
}

func BenchmarkTensorClone(b *testing.B) {
	sizes := []struct {
		name string
		size int
	}{
		{"64x64", 64},
		{"256x256", 256},
		{"1024x1024", 1024},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			t, err := tendo.RandN(size.size, size.size)
			if err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for b.Loop() {
				_ = t.Clone()
			}
		})
	}
}

func BenchmarkTensorContiguous(b *testing.B) {
	ctx := context.Background()

	b.Run("Already_Contiguous", func(b *testing.B) {
		t, err := tendo.RandN(256, 256)
		if err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for b.Loop() {
			_ = t.Contiguous()
		}
	})

	b.Run("Transposed", func(b *testing.B) {
		t, err := tendo.RandN(256, 256)
		if err != nil {
			b.Fatal(err)
		}
		transposed, err := tendo.NewTranspose(0, 1).Process(ctx, t)
		if err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for b.Loop() {
			_ = transposed.Contiguous()
		}
	})
}

func BenchmarkMemoryPool(b *testing.B) {
	b.Run("Alloc_Free_CPU", func(b *testing.B) {
		pool := tendo.NewPool()
		b.ResetTimer()
		for b.Loop() {
			s := pool.AllocCPU(1024, tendo.Float32)
			pool.FreeCPU(s)
		}
	})

	b.Run("Alloc_Only", func(b *testing.B) {
		pool := tendo.NewPool()
		b.ResetTimer()
		for b.Loop() {
			_ = pool.AllocCPU(1024, tendo.Float32)
		}
	})
}
