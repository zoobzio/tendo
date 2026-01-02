package tendo

import (
	"context"
	"fmt"

	"github.com/zoobzio/pipz"
)

// Add is a chainable operator that performs element-wise addition.
type Add struct {
	backend BinaryOps
	other   *Tensor
}

// NewAdd creates an Add operator.
func NewAdd(backend BinaryOps, other *Tensor) *Add {
	return &Add{backend: backend, other: other}
}

// Process performs element-wise addition.
func (a *Add) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := a.backend.Add(ctx, t, a.other)
	if err != nil {
		return nil, fmt.Errorf("add: %w", err)
	}

	emitWithTrace(ctx, OpAdd,
		KeyInputA.Field(t),
		KeyInputB.Field(a.other),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "add", nil)

	return out, nil
}

// Name returns the operator name.
func (a *Add) Name() pipz.Name { return "add" }

// Close releases any resources held by this operator.
func (a *Add) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Add)(nil)

// Sub is a chainable operator that performs element-wise subtraction.
type Sub struct {
	backend BinaryOps
	other   *Tensor
}

// NewSub creates a Sub operator.
func NewSub(backend BinaryOps, other *Tensor) *Sub {
	return &Sub{backend: backend, other: other}
}

// Process performs element-wise subtraction.
func (s *Sub) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.Sub(ctx, t, s.other)
	if err != nil {
		return nil, fmt.Errorf("sub: %w", err)
	}

	emitWithTrace(ctx, OpSub,
		KeyInputA.Field(t),
		KeyInputB.Field(s.other),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "sub", nil)

	return out, nil
}

// Name returns the operator name.
func (s *Sub) Name() pipz.Name { return "sub" }

// Close releases any resources held by this operator.
func (s *Sub) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Sub)(nil)

// Mul is a chainable operator that performs element-wise multiplication.
type Mul struct {
	backend BinaryOps
	other   *Tensor
}

// NewMul creates a Mul operator.
func NewMul(backend BinaryOps, other *Tensor) *Mul {
	return &Mul{backend: backend, other: other}
}

// Process performs element-wise multiplication.
func (m *Mul) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := m.backend.Mul(ctx, t, m.other)
	if err != nil {
		return nil, fmt.Errorf("mul: %w", err)
	}

	emitWithTrace(ctx, OpMul,
		KeyInputA.Field(t),
		KeyInputB.Field(m.other),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "mul", map[string]*Tensor{"a": t, "b": m.other})

	return out, nil
}

// Name returns the operator name.
func (m *Mul) Name() pipz.Name { return "mul" }

// Close releases any resources held by this operator.
func (m *Mul) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Mul)(nil)

// Div is a chainable operator that performs element-wise division.
type Div struct {
	backend BinaryOps
	other   *Tensor
}

// NewDiv creates a Div operator.
func NewDiv(backend BinaryOps, other *Tensor) *Div {
	return &Div{backend: backend, other: other}
}

// Process performs element-wise division.
func (d *Div) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := d.backend.Div(ctx, t, d.other)
	if err != nil {
		return nil, fmt.Errorf("div: %w", err)
	}

	emitWithTrace(ctx, OpDiv,
		KeyInputA.Field(t),
		KeyInputB.Field(d.other),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "div", map[string]*Tensor{"a": t, "b": d.other})

	return out, nil
}

// Name returns the operator name.
func (d *Div) Name() pipz.Name { return "div" }

// Close releases any resources held by this operator.
func (d *Div) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Div)(nil)

// Neg is a chainable operator that negates each element.
type Neg struct {
	backend UnaryOps
}

// NewNeg creates a Neg operator.
func NewNeg(backend UnaryOps) *Neg {
	return &Neg{backend: backend}
}

// Process negates each element.
func (n *Neg) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := n.backend.Neg(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("neg: %w", err)
	}

	emitWithTrace(ctx, OpNeg,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "neg", nil)

	return out, nil
}

// Name returns the operator name.
func (n *Neg) Name() pipz.Name { return "neg" }

// Close releases any resources held by this operator.
func (n *Neg) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Neg)(nil)

// Abs is a chainable operator that takes the absolute value of each element.
type Abs struct {
	backend UnaryOps
}

// NewAbs creates an Abs operator.
func NewAbs(backend UnaryOps) *Abs {
	return &Abs{backend: backend}
}

// Process computes the absolute value of each element.
func (a *Abs) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := a.backend.Abs(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("abs: %w", err)
	}

	emitWithTrace(ctx, OpAbs,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "abs", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (a *Abs) Name() pipz.Name { return "abs" }

// Close releases any resources held by this operator.
func (a *Abs) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Abs)(nil)

// Exp is a chainable operator that computes e^x for each element.
type Exp struct {
	backend UnaryOps
}

// NewExp creates an Exp operator.
func NewExp(backend UnaryOps) *Exp {
	return &Exp{backend: backend}
}

// Process computes e^x for each element.
func (e *Exp) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := e.backend.Exp(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("exp: %w", err)
	}

	emitWithTrace(ctx, OpExp,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "exp", map[string]*Tensor{"output": out})

	return out, nil
}

// Name returns the operator name.
func (e *Exp) Name() pipz.Name { return "exp" }

// Close releases any resources held by this operator.
func (e *Exp) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Exp)(nil)

// Log is a chainable operator that computes the natural logarithm of each element.
type Log struct {
	backend UnaryOps
}

// NewLog creates a Log operator.
func NewLog(backend UnaryOps) *Log {
	return &Log{backend: backend}
}

// Process computes the natural logarithm of each element.
func (l *Log) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := l.backend.Log(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("log: %w", err)
	}

	emitWithTrace(ctx, OpLog,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "log", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (l *Log) Name() pipz.Name { return "log" }

// Close releases any resources held by this operator.
func (l *Log) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Log)(nil)

// Sqrt is a chainable operator that computes the square root of each element.
type Sqrt struct {
	backend UnaryOps
}

// NewSqrt creates a Sqrt operator.
func NewSqrt(backend UnaryOps) *Sqrt {
	return &Sqrt{backend: backend}
}

// Process computes the square root of each element.
func (s *Sqrt) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.Sqrt(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("sqrt: %w", err)
	}

	emitWithTrace(ctx, OpSqrt,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "sqrt", map[string]*Tensor{"output": out})

	return out, nil
}

// Name returns the operator name.
func (s *Sqrt) Name() pipz.Name { return "sqrt" }

// Close releases any resources held by this operator.
func (s *Sqrt) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Sqrt)(nil)

// Square is a chainable operator that squares each element.
type Square struct {
	backend UnaryOps
}

// NewSquare creates a Square operator.
func NewSquare(backend UnaryOps) *Square {
	return &Square{backend: backend}
}

// Process squares each element.
func (s *Square) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.Square(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("square: %w", err)
	}

	emitWithTrace(ctx, OpSquare,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "square", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (s *Square) Name() pipz.Name { return "square" }

// Close releases any resources held by this operator.
func (s *Square) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Square)(nil)

// Sign is a chainable operator that computes the sign of each element.
// Returns -1 for negative, 0 for zero, 1 for positive.
type Sign struct {
	backend UnaryOps
}

// NewSign creates a Sign operator.
func NewSign(backend UnaryOps) *Sign {
	return &Sign{backend: backend}
}

// Process computes the sign of each element.
func (s *Sign) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.Sign(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("sign: %w", err)
	}

	emitWithTrace(ctx, OpSign,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	// Sign is non-differentiable but we still propagate tape for graph structure
	propagateTape(t, out, "sign", nil)

	return out, nil
}

// Name returns the operator name.
func (s *Sign) Name() pipz.Name { return "sign" }

// Close releases any resources held by this operator.
func (s *Sign) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Sign)(nil)

// Pow is a chainable operator that raises each element to a power.
type Pow struct {
	backend BinaryOps
	exp     float32
}

// NewPow creates a Pow operator.
func NewPow(backend BinaryOps, exp float32) *Pow {
	return &Pow{backend: backend, exp: exp}
}

// Process raises each element to the power.
func (p *Pow) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := p.backend.Pow(ctx, t, p.exp)
	if err != nil {
		return nil, fmt.Errorf("pow: %w", err)
	}

	emitWithTrace(ctx, OpPow,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyScalar.Field(p.exp),
	)

	propagateTape(t, out, "pow", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (p *Pow) Name() pipz.Name { return "pow" }

// Close releases any resources held by this operator.
func (p *Pow) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Pow)(nil)

// Clamp is a chainable operator that clamps values to a range.
type Clamp struct {
	backend CompareOps
	min     float32
	max     float32
}

// NewClamp creates a Clamp operator.
func NewClamp(backend CompareOps, minVal, maxVal float32) *Clamp {
	return &Clamp{backend: backend, min: minVal, max: maxVal}
}

// Process clamps values to [min, max].
func (c *Clamp) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := c.backend.Clamp(ctx, t, c.min, c.max)
	if err != nil {
		return nil, fmt.Errorf("clamp: %w", err)
	}

	emitWithTrace(ctx, OpClamp,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyMin.Field(c.min),
		KeyMax.Field(c.max),
	)

	propagateTape(t, out, "clamp", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (c *Clamp) Name() pipz.Name { return "clamp" }

// Close releases any resources held by this operator.
func (c *Clamp) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Clamp)(nil)

// Where is a chainable operator that selects elements based on a condition.
// For each element: if condition > 0, select from input; otherwise select from other.
type Where struct {
	backend   CompareOps
	condition *Tensor
	other     *Tensor
}

// NewWhere creates a Where operator.
func NewWhere(backend CompareOps, condition, other *Tensor) *Where {
	return &Where{backend: backend, condition: condition, other: other}
}

// Process selects elements based on the condition.
func (w *Where) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := w.backend.Where(ctx, w.condition, t, w.other)
	if err != nil {
		return nil, fmt.Errorf("where: %w", err)
	}

	emitWithTrace(ctx, OpWhere,
		KeyInput.Field(t),
		KeyCondition.Field(w.condition),
		KeyOther.Field(w.other),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "where", map[string]*Tensor{"condition": w.condition})

	return out, nil
}

// Name returns the operator name.
func (w *Where) Name() pipz.Name { return "where" }

// Close releases any resources held by this operator.
func (w *Where) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Where)(nil)

// Sin is a chainable operator that computes the sine of each element.
type Sin struct {
	backend UnaryOps
}

// NewSin creates a Sin operator.
func NewSin(backend UnaryOps) *Sin {
	return &Sin{backend: backend}
}

// Process computes the sine of each element.
func (s *Sin) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := s.backend.Sin(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("sin: %w", err)
	}

	emitWithTrace(ctx, OpSin,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "sin", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (s *Sin) Name() pipz.Name { return "sin" }

// Close releases any resources held by this operator.
func (s *Sin) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Sin)(nil)

// Cos is a chainable operator that computes the cosine of each element.
type Cos struct {
	backend UnaryOps
}

// NewCos creates a Cos operator.
func NewCos(backend UnaryOps) *Cos {
	return &Cos{backend: backend}
}

// Process computes the cosine of each element.
func (c *Cos) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := c.backend.Cos(ctx, t)
	if err != nil {
		return nil, fmt.Errorf("cos: %w", err)
	}

	emitWithTrace(ctx, OpCos,
		KeyInput.Field(t),
		KeyOutput.Field(out),
	)

	propagateTape(t, out, "cos", map[string]*Tensor{"input": t})

	return out, nil
}

// Name returns the operator name.
func (c *Cos) Name() pipz.Name { return "cos" }

// Close releases any resources held by this operator.
func (c *Cos) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Cos)(nil)

// Tril is a chainable operator that returns the lower triangular part of a 2D tensor.
// Elements above diagonal + k are zeroed.
// k=0: main diagonal, k<0: below, k>0: above.
type Tril struct {
	backend CompareOps
	k       int
}

// NewTril creates a Tril operator.
func NewTril(backend CompareOps, k int) *Tril {
	return &Tril{backend: backend, k: k}
}

// Process returns the lower triangular part of the input.
func (tr *Tril) Process(ctx context.Context, t *Tensor) (*Tensor, error) {
	out, err := tr.backend.Tril(ctx, t, tr.k)
	if err != nil {
		return nil, fmt.Errorf("tril: %w", err)
	}

	emitWithTrace(ctx, OpTril,
		KeyInput.Field(t),
		KeyOutput.Field(out),
		KeyK.Field(tr.k),
	)

	propagateTape(t, out, "tril", nil)

	return out, nil
}

// Name returns the operator name.
func (tr *Tril) Name() pipz.Name { return "tril" }

// Close releases any resources held by this operator.
func (tr *Tril) Close() error { return nil }

var _ pipz.Chainable[*Tensor] = (*Tril)(nil)

// shapesEqual returns true if two shapes are identical.
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
