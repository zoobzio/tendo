package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// To returns a Chainable that moves the tensor to the specified device.
// If already on the target device, returns the tensor unchanged.
func To(device Device) pipz.Chainable[*Tensor] {
	return pipz.Apply(pipz.NewIdentity("to", "Device transfer"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
		if t.Device() == device {
			return t, nil
		}

		fromDevice := t.Device()
		result, err := transferTo(t, device)
		if err != nil {
			return nil, fmt.Errorf("to: %w", err)
		}

		emitWithTrace(ctx, TensorTransfer,
			KeyInput.Field(t),
			KeyOutput.Field(result),
			KeyFromDevice.Field(fromDevice),
			KeyToDevice.Field(device),
		)

		return result, nil
	})
}

// ToCPU returns a Chainable that moves the tensor to CPU.
func ToCPU() pipz.Chainable[*Tensor] {
	return To(Device{Type: CPU, Index: 0})
}

// ToGPU returns a Chainable that moves the tensor to the specified GPU.
func ToGPU(index int) pipz.Chainable[*Tensor] {
	return To(Device{Type: CUDA, Index: index})
}

// MakeContiguous returns a Chainable that makes the tensor contiguous in memory.
// If already contiguous, returns the tensor unchanged.
func MakeContiguous() pipz.Chainable[*Tensor] {
	return pipz.Transform(pipz.NewIdentity("contiguous", "Make contiguous"), func(_ context.Context, t *Tensor) *Tensor {
		return t.Contiguous()
	})
}

// transferTo moves tensor data to the target device.
func transferTo(t *Tensor, device Device) (*Tensor, error) {
	// Ensure tensor is contiguous before transfer
	src := t.Contiguous()

	// CPU to CPU: clone
	if src.Device().Type == CPU && device.Type == CPU {
		return src.Clone(), nil
	}

	// CPU to CUDA
	if src.Device().Type == CPU && device.Type == CUDA {
		cpuStorage, ok := src.storage.(CPUDataAccessor)
		if !ok {
			return nil, &DeviceError{Expected: CPU, Got: src.Device().Type}
		}

		// Get CUDA backend
		backend, ok := GetBackend(CUDA)
		if !ok {
			return nil, ErrCUDANotAvailable
		}

		// Create CUDA storage and copy data
		cudaStorage, err := backend.NewStorageFromSlice(cpuStorage.Data(), src.DType(), device.Index)
		if err != nil {
			return nil, fmt.Errorf("cpu→cuda: %w", err)
		}

		return NewTensor(cudaStorage, src.Shape(), nil), nil
	}

	// CUDA to CPU
	if src.Device().Type == CUDA && device.Type == CPU {
		copier, ok := src.storage.(HostCopier)
		if !ok {
			return nil, &DeviceError{Expected: CUDA, Got: src.Device().Type}
		}

		// Copy data from GPU to host
		data, err := copier.CopyToHost()
		if err != nil {
			return nil, fmt.Errorf("cuda→cpu: %w", err)
		}

		// Create CPU storage from the data
		cpuStorage := NewCPUStorageFromSlice(data, src.DType())
		return NewTensor(cpuStorage, src.Shape(), nil), nil
	}

	// CUDA to CUDA (same or different device)
	if src.Device().Type == CUDA && device.Type == CUDA {
		// Same device: just clone
		if src.Device().Index == device.Index {
			cloned := src.storage.Clone()
			if cloned == nil {
				return nil, &CUDAError{Message: "failed to clone CUDA storage"}
			}
			return NewTensor(cloned, src.Shape(), nil), nil
		}

		// Different device: copy via host (peer-to-peer would be faster but more complex)
		copier, ok := src.storage.(HostCopier)
		if !ok {
			return nil, &DeviceError{Expected: CUDA, Got: src.Device().Type}
		}

		data, err := copier.CopyToHost()
		if err != nil {
			return nil, fmt.Errorf("cuda→cuda: copy to host: %w", err)
		}

		// Get CUDA backend
		backend, ok := GetBackend(CUDA)
		if !ok {
			return nil, &DeviceError{Expected: CUDA, Got: CPU}
		}

		dstStorage, err := backend.NewStorageFromSlice(data, src.DType(), device.Index)
		if err != nil {
			return nil, fmt.Errorf("cuda→cuda: copy to device: %w", err)
		}

		return NewTensor(dstStorage, src.Shape(), nil), nil
	}

	return nil, &DeviceError{Expected: src.Device().Type, Got: device.Type}
}

// Pin returns a Chainable that pins the tensor's memory for faster CPU-GPU transfer.
// Only applicable to CPU tensors. No-op if already pinned or on GPU.
func Pin() pipz.Chainable[*Tensor] {
	return pipz.Transform(pipz.NewIdentity("pin", "Pin memory"), func(_ context.Context, t *Tensor) *Tensor {
		// TODO: Implement pinned memory when CUDA is available
		// This would require cudaHostAlloc instead of regular malloc
		return t
	})
}

// Unpin returns a Chainable that unpins the tensor's memory.
func Unpin() pipz.Chainable[*Tensor] {
	return pipz.Transform(pipz.NewIdentity("unpin", "Unpin memory"), func(_ context.Context, t *Tensor) *Tensor {
		// TODO: Implement unpinning when CUDA is available
		return t
	})
}

// Sync returns a Chainable that synchronizes the tensor's device.
// For CUDA tensors, this blocks until all pending operations complete.
// For CPU tensors, this is a no-op.
func Sync() pipz.Chainable[*Tensor] {
	return pipz.Apply(pipz.NewIdentity("sync", "Device synchronization"), func(ctx context.Context, t *Tensor) (*Tensor, error) {
		if syncer, ok := t.storage.(Syncer); ok {
			if err := syncer.Sync(); err != nil {
				return nil, fmt.Errorf("sync: %w", err)
			}
		}
		return t, nil
	})
}
