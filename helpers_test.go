package tendo

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/zoobzio/pipz"
)

func newTestTensor() *Tensor {
	return NewTensor(NewCPUStorageFromSlice([]float32{1, 2, 3, 4}, Float32), []int{2, 2}, nil)
}

// mustCPUData extracts float32 data from a CPU tensor, panicking if not CPU storage.
func mustCPUData(t *Tensor) []float32 {
	accessor, ok := t.storage.(CPUDataAccessor)
	if !ok {
		panic("expected CPU storage")
	}
	return accessor.Data()
}

func TestSequence(t *testing.T) {
	tensor := newTestTensor()

	seq := Sequence("pipeline",
		Op(pipz.NewIdentity("double", "Double values"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
			data := mustCPUData(t)
			doubled := make([]float32, len(data))
			for i, v := range data {
				doubled[i] = v * 2
			}
			return NewTensor(NewCPUStorageFromSlice(doubled, Float32), t.Shape(), nil), nil
		}),
		Op(pipz.NewIdentity("add-one", "Add one"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
			data := mustCPUData(t)
			added := make([]float32, len(data))
			for i, v := range data {
				added[i] = v + 1
			}
			return NewTensor(NewCPUStorageFromSlice(added, Float32), t.Shape(), nil), nil
		}),
	)

	result, err := seq.Process(context.Background(), tensor)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// (1*2)+1 = 3, (2*2)+1 = 5, etc.
	data := mustCPUData(result)
	expected := []float32{3, 5, 7, 9}
	for i, v := range data {
		if v != expected[i] {
			t.Errorf("at index %d: expected %v, got %v", i, expected[i], v)
		}
	}
}

func TestFilter(t *testing.T) {
	t.Run("executes processor when predicate true", func(t *testing.T) {
		tensor := newTestTensor()

		filter := Filter("contiguous-only",
			func(ctx context.Context, t *Tensor) bool {
				return t.IsContiguous()
			},
			Op(pipz.NewIdentity("process", "Process tensor"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
				data := mustCPUData(t)
				doubled := make([]float32, len(data))
				for i, v := range data {
					doubled[i] = v * 2
				}
				return NewTensor(NewCPUStorageFromSlice(doubled, Float32), t.Shape(), nil), nil
			}),
		)

		result, err := filter.Process(context.Background(), tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := mustCPUData(result)
		if data[0] != 2 {
			t.Errorf("expected doubled tensor, got %v", data)
		}
	})

	t.Run("passes through when predicate false", func(t *testing.T) {
		tensor := newTestTensor()

		filter := Filter("large-only",
			func(ctx context.Context, t *Tensor) bool {
				return t.Numel() > 100
			},
			Op(pipz.NewIdentity("process", "Process tensor"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
				data := mustCPUData(t)
				doubled := make([]float32, len(data))
				for i, v := range data {
					doubled[i] = v * 2
				}
				return NewTensor(NewCPUStorageFromSlice(doubled, Float32), t.Shape(), nil), nil
			}),
		)

		result, err := filter.Process(context.Background(), tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should be unchanged
		data := mustCPUData(result)
		if data[0] != 1 {
			t.Errorf("expected original tensor, got %v", data)
		}
	})
}

func TestSwitch(t *testing.T) {
	tensor := newTestTensor()

	router := Switch("device-router", func(ctx context.Context, t *Tensor) string {
		return t.Device().Type.String()
	})
	router.AddRoute("cpu", Op(pipz.NewIdentity("cpu-path", "CPU path"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
		data := mustCPUData(t)
		doubled := make([]float32, len(data))
		for i, v := range data {
			doubled[i] = v * 2
		}
		return NewTensor(NewCPUStorageFromSlice(doubled, Float32), t.Shape(), nil), nil
	}))

	result, err := router.Process(context.Background(), tensor)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data := mustCPUData(result)
	if data[0] != 2 {
		t.Errorf("expected doubled tensor, got %v", data)
	}
}

func TestFallback(t *testing.T) {
	tensor := newTestTensor()

	fallback := Fallback("resilient",
		Op(pipz.NewIdentity("primary", "Primary processor"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
			return t, errors.New("primary failed")
		}),
		Op(pipz.NewIdentity("backup", "Backup processor"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
			data := mustCPUData(t)
			doubled := make([]float32, len(data))
			for i, v := range data {
				doubled[i] = v * 2
			}
			return NewTensor(NewCPUStorageFromSlice(doubled, Float32), t.Shape(), nil), nil
		}),
	)

	result, err := fallback.Process(context.Background(), tensor)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data := mustCPUData(result)
	if data[0] != 2 {
		t.Errorf("expected backup result, got %v", data)
	}
}

func TestTimeout(t *testing.T) {
	t.Run("completes within timeout", func(t *testing.T) {
		tensor := newTestTensor()

		timeout := Timeout("bounded", Op(pipz.NewIdentity("fast", "Fast processor"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
			data := mustCPUData(t)
			doubled := make([]float32, len(data))
			for i, v := range data {
				doubled[i] = v * 2
			}
			return NewTensor(NewCPUStorageFromSlice(doubled, Float32), t.Shape(), nil), nil
		}), time.Second)

		result, err := timeout.Process(context.Background(), tensor)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data := mustCPUData(result)
		if data[0] != 2 {
			t.Errorf("expected doubled tensor, got %v", data)
		}
	})

	t.Run("fails on timeout", func(t *testing.T) {
		tensor := newTestTensor()

		timeout := Timeout("bounded", Op(pipz.NewIdentity("slow", "Slow processor"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
			select {
			case <-time.After(500 * time.Millisecond):
				return t, nil
			case <-ctx.Done():
				return t, ctx.Err()
			}
		}), 10*time.Millisecond)

		_, err := timeout.Process(context.Background(), tensor)
		if err == nil {
			t.Error("expected timeout error")
		}
	})
}
