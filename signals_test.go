package tendo

import (
	"testing"
)

func TestSignals_Exist(t *testing.T) {
	// Verify lifecycle signals are defined
	signals := []struct {
		name   string
		signal interface{}
	}{
		{"TensorCreated", TensorCreated},
		{"TensorFreed", TensorFreed},
		{"TensorTransfer", TensorTransfer},
	}

	for _, s := range signals {
		if s.signal == nil {
			t.Errorf("Signal %s should not be nil", s.name)
		}
	}
}

func TestOpSignals_Exist(t *testing.T) {
	// Verify operation signals are defined
	opSignals := []struct {
		name   string
		signal interface{}
	}{
		// Element-wise
		{"OpAdd", OpAdd},
		{"OpSub", OpSub},
		{"OpMul", OpMul},
		{"OpDiv", OpDiv},
		{"OpNeg", OpNeg},
		{"OpAbs", OpAbs},
		{"OpExp", OpExp},
		{"OpLog", OpLog},
		{"OpSqrt", OpSqrt},
		{"OpSquare", OpSquare},
		{"OpPow", OpPow},

		// Matrix
		{"OpMatMul", OpMatMul},
		{"OpTranspose", OpTranspose},

		// Reduction
		{"OpSum", OpSum},
		{"OpMean", OpMean},
		{"OpMax", OpMax},
		{"OpMin", OpMin},

		// Activation
		{"OpReLU", OpReLU},
		{"OpSigmoid", OpSigmoid},
		{"OpTanh", OpTanh},
		{"OpSoftmax", OpSoftmax},
	}

	for _, s := range opSignals {
		if s.signal == nil {
			t.Errorf("Signal %s should not be nil", s.name)
		}
	}
}

func TestIdentities_Exist(t *testing.T) {
	// Verify operation identities are defined
	identities := []struct {
		name     string
		identity interface{}
	}{
		// Element-wise
		{"IdentityAdd", IdentityAdd},
		{"IdentitySub", IdentitySub},
		{"IdentityMul", IdentityMul},
		{"IdentityDiv", IdentityDiv},

		// Activation
		{"IdentityReLU", IdentityReLU},
		{"IdentitySigmoid", IdentitySigmoid},
		{"IdentityTanh", IdentityTanh},
		{"IdentitySoftmax", IdentitySoftmax},

		// Matrix
		{"IdentityMatMul", IdentityMatMul},
		{"IdentityTranspose", IdentityTranspose},

		// Reduction
		{"IdentitySum", IdentitySum},
		{"IdentityMean", IdentityMean},
	}

	for _, id := range identities {
		if id.identity == nil {
			t.Errorf("Identity %s should not be nil", id.name)
		}
	}
}

func TestKeys_Exist(t *testing.T) {
	// Verify typed keys are defined
	keys := []struct {
		name string
		key  interface{}
	}{
		{"KeyInput", KeyInput},
		{"KeyInputA", KeyInputA},
		{"KeyInputB", KeyInputB},
		{"KeyOutput", KeyOutput},
		{"KeyShape", KeyShape},
		{"KeyDevice", KeyDevice},
		{"KeyDim", KeyDim},
		{"KeyDims", KeyDims},
	}

	for _, k := range keys {
		if k.key == nil {
			t.Errorf("Key %s should not be nil", k.name)
		}
	}
}

func TestVariants_Defined(t *testing.T) {
	// Verify custom variants are defined
	variants := []struct {
		name    string
		variant interface{}
	}{
		{"VariantTensor", VariantTensor},
		{"VariantTensorList", VariantTensorList},
		{"VariantShape", VariantShape},
		{"VariantDevice", VariantDevice},
	}

	for _, v := range variants {
		if v.variant == "" {
			t.Errorf("Variant %s should not be empty", v.name)
		}
	}
}
