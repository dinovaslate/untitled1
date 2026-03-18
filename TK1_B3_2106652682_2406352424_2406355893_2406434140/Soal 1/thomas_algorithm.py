#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from algorithm_memory import AlgorithmMemoryTracker, snapshot_memory


DEFAULT_PIVOT_TOL = 1e-12


def extract_four_diagonal_bands(
    matrix: np.ndarray,
    *,
    tol: float = DEFAULT_PIVOT_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square.")

    n = A.shape[0]
    lower = np.diag(A, k=-1).astype(np.float64, copy=True)
    diagonal = np.diag(A, k=0).astype(np.float64, copy=True)
    upper_1 = np.diag(A, k=1).astype(np.float64, copy=True)
    upper_2 = np.diag(A, k=2).astype(np.float64, copy=True)

    row_idx, col_idx = np.indices(A.shape)
    allowed = (
        (row_idx == col_idx)
        | (row_idx == col_idx + 1)
        | (col_idx == row_idx + 1)
        | (col_idx == row_idx + 2)
    )
    if np.any(np.abs(A[~allowed]) > tol):
        raise ValueError("matrix is not 4-diagonal with lower bandwidth 1 and upper bandwidth 2.")

    return lower, diagonal, upper_1, upper_2


def factorize_four_diagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper_1: np.ndarray,
    upper_2: np.ndarray,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
    memory_tracker: AlgorithmMemoryTracker | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower_arr = np.asarray(lower, dtype=np.float64)
    diagonal_arr = np.asarray(diagonal, dtype=np.float64)
    upper1_arr = np.asarray(upper_1, dtype=np.float64)
    upper2_arr = np.asarray(upper_2, dtype=np.float64)

    n = diagonal_arr.shape[0]
    if n == 0:
        raise ValueError("diagonal must be non-empty.")
    if lower_arr.shape != (max(n - 1, 0),):
        raise ValueError("lower must have shape (n - 1,).")
    if upper1_arr.shape != (max(n - 1, 0),):
        raise ValueError("upper_1 must have shape (n - 1,).")
    if upper2_arr.shape != (max(n - 2, 0),):
        raise ValueError("upper_2 must have shape (n - 2,).")

    multipliers = np.zeros(max(n - 1, 0), dtype=np.float64)
    u_diagonal = diagonal_arr.copy()
    u_upper_1 = upper1_arr.copy()
    u_upper_2 = upper2_arr.copy()
    snapshot_memory(
        memory_tracker,
        lower_arr,
        diagonal_arr,
        upper1_arr,
        upper2_arr,
        multipliers,
        u_diagonal,
        u_upper_1,
        u_upper_2,
        label="factorize_four_diagonal:start",
    )

    if abs(float(u_diagonal[0])) < pivot_tol:
        raise np.linalg.LinAlgError(
            f"Near-zero pivot encountered at index 0: {u_diagonal[0]}."
        )

    for i in range(n - 1):
        multipliers[i] = lower_arr[i] / u_diagonal[i]
        u_diagonal[i + 1] -= multipliers[i] * u_upper_1[i]
        if i + 1 < n - 1:
            u_upper_1[i + 1] -= multipliers[i] * u_upper_2[i]
        if abs(float(u_diagonal[i + 1])) < pivot_tol:
            raise np.linalg.LinAlgError(
                f"Near-zero pivot encountered at index {i + 1}: {u_diagonal[i + 1]}."
            )

    snapshot_memory(
        memory_tracker,
        lower_arr,
        diagonal_arr,
        upper1_arr,
        upper2_arr,
        multipliers,
        u_diagonal,
        u_upper_1,
        u_upper_2,
        label="factorize_four_diagonal:final",
    )
    return multipliers, u_diagonal, u_upper_1, u_upper_2


def solve_four_diagonal_factored(
    multipliers: np.ndarray,
    u_diagonal: np.ndarray,
    u_upper_1: np.ndarray,
    u_upper_2: np.ndarray,
    rhs: np.ndarray,
    memory_tracker: AlgorithmMemoryTracker | None = None,
) -> np.ndarray:
    rhs_arr = np.array(rhs, copy=True, dtype=np.float64)
    squeeze = rhs_arr.ndim == 1
    if squeeze:
        rhs_arr = rhs_arr.reshape(-1, 1)

    n = u_diagonal.shape[0]
    if rhs_arr.shape[0] != n:
        raise ValueError("rhs dimension does not match factorized matrix.")

    y = np.empty_like(rhs_arr)
    snapshot_memory(
        memory_tracker,
        multipliers,
        u_diagonal,
        u_upper_1,
        u_upper_2,
        rhs_arr,
        y,
        label="solve_four_diagonal_factored:start",
    )
    y[0] = rhs_arr[0]
    for i in range(1, n):
        y[i] = rhs_arr[i] - multipliers[i - 1] * y[i - 1]

    x = np.empty_like(rhs_arr)
    x[-1] = y[-1] / u_diagonal[-1]
    if n >= 2:
        x[-2] = (y[-2] - u_upper_1[-1] * x[-1]) / u_diagonal[-2]
    for i in range(n - 3, -1, -1):
        x[i] = (
            y[i]
            - u_upper_1[i] * x[i + 1]
            - u_upper_2[i] * x[i + 2]
        ) / u_diagonal[i]

    snapshot_memory(
        memory_tracker,
        multipliers,
        u_diagonal,
        u_upper_1,
        u_upper_2,
        rhs_arr,
        y,
        x,
        label="solve_four_diagonal_factored:final",
    )
    return x.ravel() if squeeze else x


def solve_four_diagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper_1: np.ndarray,
    upper_2: np.ndarray,
    rhs: np.ndarray,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
    memory_tracker: AlgorithmMemoryTracker | None = None,
) -> np.ndarray:
    factors = factorize_four_diagonal(
        lower,
        diagonal,
        upper_1,
        upper_2,
        pivot_tol=pivot_tol,
        memory_tracker=memory_tracker,
    )
    return solve_four_diagonal_factored(*factors, rhs, memory_tracker=memory_tracker)


def four_diagonal_matvec(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper_1: np.ndarray,
    upper_2: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    lower_arr = np.asarray(lower, dtype=np.float64)
    diagonal_arr = np.asarray(diagonal, dtype=np.float64)
    upper1_arr = np.asarray(upper_1, dtype=np.float64)
    upper2_arr = np.asarray(upper_2, dtype=np.float64)
    vector_arr = np.asarray(vector, dtype=np.float64)

    result = diagonal_arr * vector_arr
    if vector_arr.shape[0] > 1:
        result[:-1] += upper1_arr * vector_arr[1:]
        result[1:] += lower_arr * vector_arr[:-1]
    if vector_arr.shape[0] > 2:
        result[:-2] += upper2_arr * vector_arr[2:]
    return result


def four_diagonal_inf_norm(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper_1: np.ndarray,
    upper_2: np.ndarray,
) -> float:
    lower_arr = np.asarray(lower, dtype=np.float64)
    diagonal_arr = np.asarray(diagonal, dtype=np.float64)
    upper1_arr = np.asarray(upper_1, dtype=np.float64)
    upper2_arr = np.asarray(upper_2, dtype=np.float64)

    row_sums = np.abs(diagonal_arr)
    if diagonal_arr.shape[0] > 1:
        row_sums[:-1] += np.abs(upper1_arr)
        row_sums[1:] += np.abs(lower_arr)
    if diagonal_arr.shape[0] > 2:
        row_sums[:-2] += np.abs(upper2_arr)
    return float(np.max(row_sums))


def lu_relative_error_inf_four_diagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper_1: np.ndarray,
    upper_2: np.ndarray,
    multipliers: np.ndarray,
    u_diagonal: np.ndarray,
    u_upper_1: np.ndarray,
    u_upper_2: np.ndarray,
) -> float:
    n = diagonal.shape[0]

    reconstructed_main = np.array(u_diagonal, copy=True, dtype=np.float64)
    if n > 1:
        reconstructed_main[1:] += multipliers * u_upper_1

    reconstructed_upper1 = np.array(u_upper_1, copy=True, dtype=np.float64)
    if n > 2:
        reconstructed_upper1[1:] += multipliers[:-1] * u_upper_2

    reconstructed_upper2 = np.array(u_upper_2, copy=True, dtype=np.float64)
    reconstructed_lower = multipliers * u_diagonal[:-1] if n > 1 else np.empty(0, dtype=np.float64)

    diff_main = reconstructed_main - np.asarray(diagonal, dtype=np.float64)
    diff_upper1 = reconstructed_upper1 - np.asarray(upper_1, dtype=np.float64)
    diff_upper2 = reconstructed_upper2 - np.asarray(upper_2, dtype=np.float64)
    diff_lower = reconstructed_lower - np.asarray(lower, dtype=np.float64)

    row_sums = np.abs(diff_main)
    if n > 1:
        row_sums[:-1] += np.abs(diff_upper1)
        row_sums[1:] += np.abs(diff_lower)
    if n > 2:
        row_sums[:-2] += np.abs(diff_upper2)

    numerator = float(np.max(row_sums))
    denominator = four_diagonal_inf_norm(lower, diagonal, upper_1, upper_2)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def normalized_residual_inf_four_diagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper_1: np.ndarray,
    upper_2: np.ndarray,
    x: np.ndarray,
    rhs: np.ndarray,
) -> float:
    numerator = np.linalg.norm(
        four_diagonal_matvec(lower, diagonal, upper_1, upper_2, x) - rhs,
        ord=np.inf,
    )
    denominator = (
        four_diagonal_inf_norm(lower, diagonal, upper_1, upper_2) * np.linalg.norm(x, ord=np.inf)
        + np.linalg.norm(rhs, ord=np.inf)
    )
    if float(denominator) == 0.0:
        return 0.0 if float(numerator) == 0.0 else float("inf")
    return float(numerator / denominator)


def factorize_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
    memory_tracker: AlgorithmMemoryTracker | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower_arr = np.asarray(lower, dtype=np.float64)
    diagonal_arr = np.asarray(diagonal, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)

    n = diagonal_arr.shape[0]
    if lower_arr.shape != (max(n - 1, 0),):
        raise ValueError("lower must have shape (n - 1,).")
    if upper_arr.shape != (max(n - 1, 0),):
        raise ValueError("upper must have shape (n - 1,).")
    if n == 0:
        raise ValueError("diagonal must be non-empty.")

    multipliers = np.zeros(max(n - 1, 0), dtype=np.float64)
    u_diagonal = diagonal_arr.copy()
    u_upper = upper_arr.copy()
    snapshot_memory(
        memory_tracker,
        lower_arr,
        diagonal_arr,
        upper_arr,
        multipliers,
        u_diagonal,
        u_upper,
        label="factorize_tridiagonal:start",
    )

    if abs(float(u_diagonal[0])) < pivot_tol:
        raise np.linalg.LinAlgError(
            f"Near-zero pivot encountered at index 0: {u_diagonal[0]}."
        )

    for i in range(n - 1):
        multipliers[i] = lower_arr[i] / u_diagonal[i]
        u_diagonal[i + 1] -= multipliers[i] * u_upper[i]
        if abs(float(u_diagonal[i + 1])) < pivot_tol:
            raise np.linalg.LinAlgError(
                f"Near-zero pivot encountered at index {i + 1}: {u_diagonal[i + 1]}."
            )

    snapshot_memory(
        memory_tracker,
        lower_arr,
        diagonal_arr,
        upper_arr,
        multipliers,
        u_diagonal,
        u_upper,
        label="factorize_tridiagonal:final",
    )
    return multipliers, u_diagonal, u_upper


def solve_tridiagonal_factored(
    multipliers: np.ndarray,
    u_diagonal: np.ndarray,
    u_upper: np.ndarray,
    rhs: np.ndarray,
    memory_tracker: AlgorithmMemoryTracker | None = None,
) -> np.ndarray:
    rhs_arr = np.array(rhs, copy=True, dtype=np.float64)
    squeeze = rhs_arr.ndim == 1
    if squeeze:
        rhs_arr = rhs_arr.reshape(-1, 1)

    n = u_diagonal.shape[0]
    if rhs_arr.shape[0] != n:
        raise ValueError("rhs dimension does not match factorized matrix.")

    y = np.empty_like(rhs_arr)
    snapshot_memory(
        memory_tracker,
        multipliers,
        u_diagonal,
        u_upper,
        rhs_arr,
        y,
        label="solve_tridiagonal_factored:start",
    )
    y[0] = rhs_arr[0]
    for i in range(1, n):
        y[i] = rhs_arr[i] - multipliers[i - 1] * y[i - 1]

    x = np.empty_like(rhs_arr)
    x[-1] = y[-1] / u_diagonal[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - u_upper[i] * x[i + 1]) / u_diagonal[i]

    snapshot_memory(
        memory_tracker,
        multipliers,
        u_diagonal,
        u_upper,
        rhs_arr,
        y,
        x,
        label="solve_tridiagonal_factored:final",
    )
    return x.ravel() if squeeze else x


def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
    memory_tracker: AlgorithmMemoryTracker | None = None,
) -> np.ndarray:
    factors = factorize_tridiagonal(
        lower,
        diagonal,
        upper,
        pivot_tol=pivot_tol,
        memory_tracker=memory_tracker,
    )
    return solve_tridiagonal_factored(*factors, rhs, memory_tracker=memory_tracker)


def tridiagonal_matvec(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    lower_arr = np.asarray(lower, dtype=np.float64)
    diagonal_arr = np.asarray(diagonal, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    vector_arr = np.asarray(vector, dtype=np.float64)

    result = diagonal_arr * vector_arr
    if vector_arr.shape[0] > 1:
        result[:-1] += upper_arr * vector_arr[1:]
        result[1:] += lower_arr * vector_arr[:-1]
    return result


def tridiagonal_inf_norm(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
) -> float:
    lower_arr = np.asarray(lower, dtype=np.float64)
    diagonal_arr = np.asarray(diagonal, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)

    row_sums = np.abs(diagonal_arr)
    if diagonal_arr.shape[0] > 1:
        row_sums[:-1] += np.abs(upper_arr)
        row_sums[1:] += np.abs(lower_arr)
    return float(np.max(row_sums))


def lu_relative_error_inf_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    multipliers: np.ndarray,
    u_diagonal: np.ndarray,
    u_upper: np.ndarray,
) -> float:
    n = diagonal.shape[0]

    diff_main = np.array(diagonal, copy=True, dtype=np.float64)
    diff_main[0] = u_diagonal[0] - diagonal[0]
    if n > 1:
        diff_main[1:] = u_diagonal[1:] + multipliers * u_upper - diagonal[1:]

    diff_lower = multipliers * u_diagonal[:-1] - lower if n > 1 else np.empty(0, dtype=np.float64)
    diff_upper = u_upper - upper if n > 1 else np.empty(0, dtype=np.float64)

    row_sums = np.abs(diff_main)
    if n > 1:
        row_sums[:-1] += np.abs(diff_upper)
        row_sums[1:] += np.abs(diff_lower)

    numerator = float(np.max(row_sums))
    denominator = tridiagonal_inf_norm(lower, diagonal, upper)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def normalized_residual_inf_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    x: np.ndarray,
    rhs: np.ndarray,
) -> float:
    numerator = np.linalg.norm(tridiagonal_matvec(lower, diagonal, upper, x) - rhs, ord=np.inf)
    denominator = (
        tridiagonal_inf_norm(lower, diagonal, upper) * np.linalg.norm(x, ord=np.inf)
        + np.linalg.norm(rhs, ord=np.inf)
    )
    if float(denominator) == 0.0:
        return 0.0 if float(numerator) == 0.0 else float("inf")
    return float(numerator / denominator)


__all__ = [
    "DEFAULT_PIVOT_TOL",
    "extract_four_diagonal_bands",
    "factorize_four_diagonal",
    "factorize_tridiagonal",
    "four_diagonal_inf_norm",
    "four_diagonal_matvec",
    "lu_relative_error_inf_four_diagonal",
    "lu_relative_error_inf_tridiagonal",
    "normalized_residual_inf_four_diagonal",
    "normalized_residual_inf_tridiagonal",
    "solve_four_diagonal",
    "solve_four_diagonal_factored",
    "solve_tridiagonal",
    "solve_tridiagonal_factored",
    "tridiagonal_inf_norm",
    "tridiagonal_matvec",
]
