package tendo

import (
	"testing"
)

func TestErrNotImplemented(t *testing.T) {
	err := &ErrNotImplemented{
		Op:      "Conv3d",
		Backend: "cpu",
	}

	expected := "Conv3d not implemented for cpu backend"
	if err.Error() != expected {
		t.Errorf("Error() = %q, want %q", err.Error(), expected)
	}
}

// TestBackendInterfaces verifies that interface composition is correct.
// This is a compile-time check that the interfaces are properly defined.
func TestBackendInterfaces(t *testing.T) {
	// These are compile-time checks - if they compile, the interfaces are valid
	var _ interface {
		StorageOps
		DeviceInfo
		TensorFactory
	} = CoreBackend(nil)

	var _ interface {
		UnaryOps
		BinaryOps
	} = ArithmeticBackend(nil)

	var _ interface {
		MatrixOps
	} = MatrixBackend(nil)

	var _ interface {
		ReduceOps
	} = ReduceBackend(nil)

	var _ interface {
		CompareOps
	} = CompareBackend(nil)

	// Full backend should compose all interfaces
	var _ interface {
		CoreBackend
		ArithmeticBackend
		MatrixBackend
		NeuralBackend
		ReduceBackend
		CompareBackend
	} = Backend(nil)
}

// TestStorageInterface verifies Storage interface is properly defined.
func TestStorageInterface(t *testing.T) {
	// CPUStorage should implement Storage
	var _ Storage = (*CPUStorage)(nil)

	// CPUStorage should implement optional interfaces
	var _ CPUDataAccessor = (*CPUStorage)(nil)
	var _ CPUInt64DataAccessor = (*CPUStorage)(nil)
	var _ PoolableStorage = (*CPUStorage)(nil)
}
