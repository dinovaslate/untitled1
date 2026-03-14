#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


DEFAULT_PIVOT_TOL = 1e-12


def factorize_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
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

    return multipliers, u_diagonal, u_upper


def solve_tridiagonal_factored(
    multipliers: np.ndarray,
    u_diagonal: np.ndarray,
    u_upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    rhs_arr = np.array(rhs, copy=True, dtype=np.float64)
    squeeze = rhs_arr.ndim == 1
    if squeeze:
        rhs_arr = rhs_arr.reshape(-1, 1)

    n = u_diagonal.shape[0]
    if rhs_arr.shape[0] != n:
        raise ValueError("rhs dimension does not match factorized matrix.")

    y = np.empty_like(rhs_arr)
    y[0] = rhs_arr[0]
    for i in range(1, n):
        y[i] = rhs_arr[i] - multipliers[i - 1] * y[i - 1]

    x = np.empty_like(rhs_arr)
    x[-1] = y[-1] / u_diagonal[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - u_upper[i] * x[i + 1]) / u_diagonal[i]

    return x.ravel() if squeeze else x


def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
) -> np.ndarray:
    factors = factorize_tridiagonal(lower, diagonal, upper, pivot_tol=pivot_tol)
    return solve_tridiagonal_factored(*factors, rhs)


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
    "factorize_tridiagonal",
    "lu_relative_error_inf_tridiagonal",
    "normalized_residual_inf_tridiagonal",
    "solve_tridiagonal",
    "solve_tridiagonal_factored",
    "tridiagonal_inf_norm",
    "tridiagonal_matvec",
]
