#!/usr/bin/env python3
"""
Read a square matrix from an Excel file and compute a blocked LU factorization
with GPU-accelerated trailing updates using CUDA through CuPy.

Design notes
------------
- Excel I/O is handled on the CPU with pandas/openpyxl.
- The blocked LU algorithm keeps the expensive panel solves and Schur-complement
  updates on the GPU using CUDA libraries underneath CuPy.
- The diagonal block factorization is done on the host for each small block.
  This is a common hybrid strategy: the diagonal panel is comparatively small,
  while the large trailing updates exploit the GPU efficiently.
- This implementation performs LU *without pivoting*. It is fast and simple,
  but it requires matrices that do not encounter zero / near-zero pivots.

Example
-------
python excel_block_lu_cuda.py matrix.xlsx --sheet 0 --block-size 64 --dtype float64

Input format
------------
- The selected sheet must contain only the matrix values.
- No headers are assumed.
- The matrix must be square.

Output
------
Creates an Excel workbook containing:
- input_matrix
- combined_lu  (lower triangle stores L without unit diagonal, upper stores U)
- L
- U
- summary
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "CuPy is required. Install a CUDA-matched build such as\n"
        "  pip install cupy-cuda12x pandas openpyxl\n"
        f"Original import error: {exc}"
    )


ArrayPair = Tuple[np.ndarray, np.ndarray]


def load_matrix_from_excel(path: Path, sheet: str | int) -> np.ndarray:
    """Load a square numeric matrix from an Excel workbook."""
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    if df.empty:
        raise ValueError("The selected Excel sheet is empty.")

    matrix = df.to_numpy(dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains NaN or infinite values.")

    return matrix



def lu_no_pivot_cpu(block: np.ndarray, pivot_tol: float = 1e-12) -> ArrayPair:
    """
    Compute an LU factorization of a small dense block using Doolittle's method.

    Returns L and U where diag(L) = 1.

    Notes:
    - No pivoting is performed.
    - Raises if a zero / near-zero pivot is found.
    """
    n = block.shape[0]
    if block.shape[0] != block.shape[1]:
        raise ValueError("Diagonal block must be square.")

    # Work in the block's dtype to preserve float32 / float64 choice.
    block = np.array(block, copy=True)
    L = np.eye(n, dtype=block.dtype)
    U = np.zeros((n, n), dtype=block.dtype)

    for j in range(n):
        for i in range(j + 1):
            U[i, j] = block[i, j] - L[i, :i] @ U[:i, j]

        pivot = U[j, j]
        if abs(float(pivot)) < pivot_tol:
            raise np.linalg.LinAlgError(
                f"Near-zero pivot encountered at local index {j}: {pivot}. "
                "This no-pivot LU implementation cannot continue safely."
            )

        for i in range(j + 1, n):
            L[i, j] = (block[i, j] - L[i, :j] @ U[:j, j]) / pivot

    return L, U



def block_lu_gpu(
        matrix: np.ndarray,
        block_size: int,
        dtype: np.dtype,
        pivot_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Hybrid blocked LU factorization.

    The matrix is stored in combined-LU form on the GPU:
    - strict lower triangle -> entries of L
    - upper triangle        -> entries of U
    - diag(L) is implicit and equal to 1

    Returns
    -------
    combined_lu_host, L_host, U_host, rel_reconstruction_error, elapsed_seconds
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    n = matrix.shape[0]
    host_matrix = np.asarray(matrix, dtype=dtype)

    # Upload once to the GPU.
    A = cp.asarray(host_matrix)
    A_original = A.copy()

    start = time.perf_counter()

    for k in range(0, n, block_size):
        b = min(block_size, n - k)
        k_end = k + b

        # Factor the current diagonal block on the CPU.
        diag_block = cp.asnumpy(A[k:k_end, k:k_end])
        L11, U11 = lu_no_pivot_cpu(diag_block, pivot_tol=pivot_tol)

        # Store the factored block back in combined form.
        combined_diag = np.tril(L11, k=-1) + U11
        A[k:k_end, k:k_end] = cp.asarray(combined_diag)

        if k_end >= n:
            continue

        # Move the small factors to the GPU.
        L11_gpu = cp.asarray(L11)
        U11_gpu = cp.asarray(U11)

        # U12 = L11^{-1} * A12
        A12 = A[k:k_end, k_end:n]
        U12 = solve_triangular(
            L11_gpu,
            A12,
            lower=True,
            unit_diagonal=True,
            overwrite_b=False,
            check_finite=False,
        )
        A[k:k_end, k_end:n] = U12

        # L21 = A21 * U11^{-1}
        # Solve transposed system: (U11^T)(L21^T) = A21^T
        A21 = A[k_end:n, k:k_end]
        L21 = solve_triangular(
            U11_gpu.T,
            A21.T,
            lower=True,
            unit_diagonal=False,
            overwrite_b=False,
            check_finite=False,
        ).T
        A[k_end:n, k:k_end] = L21

        # Schur complement update: A22 -= L21 @ U12
        # This is the dominant cost and is heavily optimized through cuBLAS.
        A[k_end:n, k_end:n] -= L21 @ U12

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

    L = cp.tril(A, k=-1) + cp.eye(n, dtype=A.dtype)
    U = cp.triu(A)

    denom = cp.linalg.norm(A_original)
    if float(denom.item()) == 0.0:
        rel_error = 0.0
    else:
        rel_error = float((cp.linalg.norm(L @ U - A_original) / denom).item())

    return cp.asnumpy(A), cp.asnumpy(L), cp.asnumpy(U), rel_error, elapsed



def save_results_to_excel(
        output_path: Path,
        input_matrix: np.ndarray,
        combined_lu: np.ndarray,
        L: np.ndarray,
        U: np.ndarray,
        block_size: int,
        dtype_name: str,
        rel_error: float,
        elapsed: float,
) -> None:
    """Write the factorization results to an Excel workbook."""
    summary = pd.DataFrame(
        {
            "metric": [
                "matrix_size",
                "block_size",
                "dtype",
                "relative_reconstruction_error",
                "elapsed_seconds",
            ],
            "value": [
                int(input_matrix.shape[0]),
                int(block_size),
                dtype_name,
                rel_error,
                elapsed,
            ],
        }
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(input_matrix).to_excel(
            writer, index=False, header=False, sheet_name="input_matrix"
        )
        pd.DataFrame(combined_lu).to_excel(
            writer, index=False, header=False, sheet_name="combined_lu"
        )
        pd.DataFrame(L).to_excel(writer, index=False, header=False, sheet_name="L")
        pd.DataFrame(U).to_excel(writer, index=False, header=False, sheet_name="U")
        summary.to_excel(writer, index=False, sheet_name="summary")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blocked LU factorization for a matrix stored in an Excel file, accelerated with CUDA via CuPy."
    )
    parser.add_argument("input", type=Path, help="Path to the input .xlsx file")
    parser.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or zero-based sheet index (default: 0)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Block size for the blocked LU algorithm (default: 64)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Floating-point type used on CPU/GPU (default: float64)",
    )
    parser.add_argument(
        "--pivot-tol",
        type=float,
        default=1e-12,
        help="Abort if a pivot magnitude is smaller than this threshold (default: 1e-12)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .xlsx path (default: <input_stem>_lu_output.xlsx)",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() not in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        raise SystemExit("Input must be an Excel workbook such as .xlsx")

    sheet = args.sheet
    # Convert purely numeric strings to int so users can pass --sheet 0 on the CLI.
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    dtype = np.float32 if args.dtype == "float32" else np.float64
    output_path = args.output or input_path.with_name(f"{input_path.stem}_lu_output.xlsx")

    try:
        matrix = load_matrix_from_excel(input_path, sheet=sheet)
        combined_lu, L, U, rel_error, elapsed = block_lu_gpu(
            matrix=matrix,
            block_size=args.block_size,
            dtype=dtype,
            pivot_tol=args.pivot_tol,
        )
        save_results_to_excel(
            output_path=output_path,
            input_matrix=matrix.astype(dtype, copy=False),
            combined_lu=combined_lu,
            L=L,
            U=U,
            block_size=args.block_size,
            dtype_name=args.dtype,
            rel_error=rel_error,
            elapsed=elapsed,
        )
    except Exception as exc:
        raise SystemExit(f"Factorization failed: {exc}") from exc

    print(f"Input workbook : {input_path}")
    print(f"Output workbook: {output_path}")
    print(f"Matrix size    : {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"Block size     : {args.block_size}")
    print(f"Dtype          : {args.dtype}")
    print(f"Relative error : {rel_error:.6e}")
    print(f"Elapsed time   : {elapsed:.6f} s")
    print("Done.")



if __name__ == "__main__":
    main()
