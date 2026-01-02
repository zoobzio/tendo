package cpu

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// blasMatMul2D performs C = A @ B using BLAS Gemm.
// Assumes row-major layout and contiguous data.
// A is m×k, B is k×n, C is m×n.
func blasMatMul2D(dataA, dataB, dataC []float32, m, k, n int) {
	a := blas32.General{Rows: m, Cols: k, Stride: k, Data: dataA}
	b := blas32.General{Rows: k, Cols: n, Stride: n, Data: dataB}
	c := blas32.General{Rows: m, Cols: n, Stride: n, Data: dataC}

	// C = 1.0 * A * B + 0.0 * C
	blas32.Gemm(blas.NoTrans, blas.NoTrans, 1.0, a, b, 0.0, c)
}
