package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/capitan"
	"github.com/zoobzio/pipz"
)

// Observability signals for device transfer.
var (
	OpTransfer = capitan.NewSignal("tendo.op.transfer", "Device transfer")
)

// Transfer is a chainable operator that moves a tensor to a different backend.
// This is used for explicit device-to-device transfers in pipelines.
type Transfer struct {
	target   Backend
	identity pipz.Identity
}

// NewTransfer creates a Transfer operator that moves tensors to the target backend.
func NewTransfer(target Backend) *Transfer {
	return &Transfer{
		identity: IdentityTransfer,
		target:   target,
	}
}

// Process transfers the tensor to the target backend.
func (t *Transfer) Process(ctx context.Context, tensor *Tensor) (*Tensor, error) {
	// If already on the target device, return as-is
	if tensor.Device().Type == t.target.DeviceType() {
		return tensor, nil
	}

	result, err := t.target.CopyFrom(tensor)
	if err != nil {
		return nil, fmt.Errorf("transfer: %w", err)
	}

	emitWithTrace(ctx, OpTransfer,
		KeyInput.Field(tensor),
		KeyOutput.Field(result),
		KeySourceDevice.Field(tensor.Device()),
		KeyTargetDevice.Field(result.Device()),
	)

	return result, nil
}

// Identity returns the operator identity.
func (t *Transfer) Identity() pipz.Identity { return t.identity }

// Schema returns the operator schema node.
func (t *Transfer) Schema() pipz.Node {
	return pipz.Node{Identity: t.identity, Type: "operator"}
}

// Close releases any resources held by this operator.
func (t *Transfer) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Transfer)(nil)

// Signal field keys for transfer.
var (
	KeySourceDevice = capitan.NewKey[Device]("source_device", VariantDevice)
	KeyTargetDevice = capitan.NewKey[Device]("target_device", VariantDevice)
)
