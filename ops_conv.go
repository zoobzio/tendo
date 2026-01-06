package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/capitan"
	"github.com/zoobzio/pipz"
)

// Conv2dConfig holds configuration for 2D convolution.
type Conv2dConfig struct {
	Padding  [2]int // [padH, padW]
	Stride   [2]int // [strideH, strideW]
	Dilation [2]int // [dilationH, dilationW]
	Groups   int    // number of groups for grouped convolution
}

// DefaultConv2dConfig returns default convolution config.
func DefaultConv2dConfig() Conv2dConfig {
	return Conv2dConfig{
		Padding:  [2]int{0, 0},
		Stride:   [2]int{1, 1},
		Dilation: [2]int{1, 1},
		Groups:   1,
	}
}

// Observability signals for convolution.
var (
	OpConv2d = capitan.NewSignal("tendo.op.conv2d", "2D convolution")
)

// Conv2d is a chainable operator that applies 2D convolution.
// Input shape: [N, C_in, H, W]
// Weight shape: [C_out, C_in/groups, kH, kW]
// Output shape: [N, C_out, H_out, W_out].
type Conv2d struct { //nolint:govet // field alignment is less important than readability
	backend  ConvOps
	weight   *Tensor
	config   Conv2dConfig
	identity pipz.Identity
}

// NewConv2d creates a Conv2d operator.
func NewConv2d(backend ConvOps, weight *Tensor, config Conv2dConfig) *Conv2d {
	return &Conv2d{
		identity: IdentityConv2d,
		backend:  backend,
		weight:   weight,
		config:   config,
	}
}

// NewConv2dSimple is a convenience constructor for standard convolution.
// Applies same padding to keep spatial dimensions.
func NewConv2dSimple(backend ConvOps, weight *Tensor, stride int) *Conv2d {
	kH := weight.Shape()[2]
	kW := weight.Shape()[3]
	padH := (kH - 1) / 2
	padW := (kW - 1) / 2

	return &Conv2d{
		identity: IdentityConv2d,
		backend:  backend,
		weight:   weight,
		config: Conv2dConfig{
			Padding:  [2]int{padH, padW},
			Stride:   [2]int{stride, stride},
			Dilation: [2]int{1, 1},
			Groups:   1,
		},
	}
}

// Process applies 2D convolution.
func (c *Conv2d) Process(ctx context.Context, input *Tensor) (*Tensor, error) {
	result, err := c.backend.Conv2d(ctx, input, c.weight, c.config.Padding, c.config.Stride, c.config.Dilation, c.config.Groups)
	if err != nil {
		return nil, fmt.Errorf("conv2d: %w", err)
	}

	emitWithTrace(ctx, OpConv2d,
		KeyInputA.Field(input),
		KeyInputB.Field(c.weight),
		KeyOutput.Field(result),
		KeyShape.Field(result.Shape()),
		KeyPadding.Field(c.config.Padding),
		KeyConvStride.Field(c.config.Stride),
		KeyDilation.Field(c.config.Dilation),
		KeyGroups.Field(c.config.Groups),
	)


	return result, nil
}

// Identity returns the operator identity.
func (c *Conv2d) Identity() pipz.Identity { return c.identity }

// Schema returns the operator schema.
func (c *Conv2d) Schema() pipz.Node { return pipz.Node{Identity: c.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (c *Conv2d) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Conv2d)(nil)

// Conv2dOutputShape computes the output shape for a conv2d operation.
func Conv2dOutputShape(inputShape, weightShape []int, config Conv2dConfig) []int {
	n := inputShape[0]
	outC := weightShape[0]
	inH := inputShape[2]
	inW := inputShape[3]
	kH := weightShape[2]
	kW := weightShape[3]

	outH := (inH+2*config.Padding[0]-config.Dilation[0]*(kH-1)-1)/config.Stride[0] + 1
	outW := (inW+2*config.Padding[1]-config.Dilation[1]*(kW-1)-1)/config.Stride[1] + 1

	return []int{n, outC, outH, outW}
}

// ConvTranspose2dConfig holds configuration for 2D transposed convolution.
type ConvTranspose2dConfig struct {
	Padding       [2]int // [padH, padW]
	OutputPadding [2]int // [outPadH, outPadW]
	Stride        [2]int // [strideH, strideW]
	Dilation      [2]int // [dilationH, dilationW]
	Groups        int    // number of groups for grouped convolution
}

// DefaultConvTranspose2dConfig returns default transposed convolution config.
func DefaultConvTranspose2dConfig() ConvTranspose2dConfig {
	return ConvTranspose2dConfig{
		Padding:       [2]int{0, 0},
		OutputPadding: [2]int{0, 0},
		Stride:        [2]int{1, 1},
		Dilation:      [2]int{1, 1},
		Groups:        1,
	}
}

// ConvTranspose2d is a chainable operator that applies 2D transposed convolution.
// Also known as deconvolution or fractionally-strided convolution.
// Input shape: [N, C_in, H, W]
// Weight shape: [C_in, C_out/groups, kH, kW]
// Output shape: [N, C_out, H_out, W_out].
type ConvTranspose2d struct { //nolint:govet // field alignment is less important than readability
	backend  ConvOps
	weight   *Tensor
	config   ConvTranspose2dConfig
	identity pipz.Identity
}

// NewConvTranspose2d creates a ConvTranspose2d operator.
func NewConvTranspose2d(backend ConvOps, weight *Tensor, config ConvTranspose2dConfig) *ConvTranspose2d {
	return &ConvTranspose2d{
		identity: IdentityConvTranspose2d,
		backend:  backend,
		weight:   weight,
		config:   config,
	}
}

// Process applies 2D transposed convolution.
func (c *ConvTranspose2d) Process(ctx context.Context, input *Tensor) (*Tensor, error) {
	result, err := c.backend.ConvTranspose2d(ctx, input, c.weight, c.config.Padding, c.config.OutputPadding, c.config.Stride, c.config.Dilation, c.config.Groups)
	if err != nil {
		return nil, fmt.Errorf("convtranspose2d: %w", err)
	}

	emitWithTrace(ctx, OpConvTranspose2d,
		KeyInputA.Field(input),
		KeyInputB.Field(c.weight),
		KeyOutput.Field(result),
		KeyShape.Field(result.Shape()),
		KeyPadding.Field(c.config.Padding),
		KeyOutputPadding.Field(c.config.OutputPadding),
		KeyConvStride.Field(c.config.Stride),
		KeyDilation.Field(c.config.Dilation),
		KeyGroups.Field(c.config.Groups),
	)


	return result, nil
}

// Identity returns the operator identity.
func (c *ConvTranspose2d) Identity() pipz.Identity { return c.identity }

// Schema returns the operator schema.
func (c *ConvTranspose2d) Schema() pipz.Node { return pipz.Node{Identity: c.identity, Type: "operator"} }

// Close releases any resources held by this operator.
func (c *ConvTranspose2d) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*ConvTranspose2d)(nil)
