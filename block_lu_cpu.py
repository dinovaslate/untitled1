#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


DEFAULT_PIVOT_TOL = 1e-12


def block_size_for_cpu(n: int) -> int:
    if n <= 32:
        return 4
    if n <= 256:
        return 8
    if n <= 1024:
        return 16
    return 32


def lu_no_pivot_cpu(
    block: np.ndarray,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    n = block.shape[0]
    if block.shape[0] != block.shape[1]:
        raise ValueError("Diagonal block must be square.")

    A = np.array(block, copy=True, dtype=np.float64)
    for k in range(n):
        pivot = A[k, k]
        if abs(float(pivot)) < pivot_tol:
            raise np.linalg.LinAlgError(
                f"Near-zero pivot encountered at local index {k}: {pivot}."
            )
        if k + 1 >= n:
            continue
        A[k + 1 :, k] /= pivot
        A[k + 1 :, k + 1 :] -= np.outer(A[k + 1 :, k], A[k, k + 1 :])

    L = np.tril(A, k=-1) + np.eye(n, dtype=A.dtype)
    U = np.triu(A)
    return L, U


def solve_lower_triangular(
    lower: np.ndarray,
    rhs: np.ndarray,
    *,
    unit_diagonal: bool,
) -> np.ndarray:
    rhs_arr = np.array(rhs, copy=True, dtype=np.float64)
    squeeze = rhs_arr.ndim == 1
    if squeeze:
        rhs_arr = rhs_arr.reshape(-1, 1)

    n = lower.shape[0]
    result = np.zeros_like(rhs_arr)

    for i in range(n):
        value = rhs_arr[i].copy()
        if i > 0:
            value -= lower[i, :i] @ result[:i]
        if not unit_diagonal:
            value /= lower[i, i]
        result[i] = value

    return result.ravel() if squeeze else result


def solve_upper_triangular(upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    rhs_arr = np.array(rhs, copy=True, dtype=np.float64)
    squeeze = rhs_arr.ndim == 1
    if squeeze:
        rhs_arr = rhs_arr.reshape(-1, 1)

    n = upper.shape[0]
    result = np.zeros_like(rhs_arr)

    for i in range(n - 1, -1, -1):
        value = rhs_arr[i].copy()
        if i + 1 < n:
            value -= upper[i, i + 1 :] @ result[i + 1 :]
        value /= upper[i, i]
        result[i] = value

    return result.ravel() if squeeze else result


def solve_via_lu_cpu(L: np.ndarray, U: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    y = solve_lower_triangular(L, rhs, unit_diagonal=True)
    return solve_upper_triangular(U, y)


def block_lu_cpu(
    matrix: np.ndarray,
    block_size: int,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    A = np.array(matrix, copy=True, dtype=np.float64)
    n = A.shape[0]

    for k in range(0, n, block_size):
        b = min(block_size, n - k)
        k_end = k + b

        L11, U11 = lu_no_pivot_cpu(A[k:k_end, k:k_end], pivot_tol=pivot_tol)
        A[k:k_end, k:k_end] = np.tril(L11, k=-1) + U11

        if k_end >= n:
            continue

        A12 = A[k:k_end, k_end:n]
        U12 = solve_lower_triangular(L11, A12, unit_diagonal=True)
        A[k:k_end, k_end:n] = U12

        A21 = A[k_end:n, k:k_end]
        L21 = solve_lower_triangular(U11.T, A21.T, unit_diagonal=False).T
        A[k_end:n, k:k_end] = L21

        A[k_end:n, k_end:n] -= L21 @ U12

    L = np.tril(A, k=-1) + np.eye(n, dtype=A.dtype)
    U = np.triu(A)
    return A, L, U


__all__ = [
    "DEFAULT_PIVOT_TOL",
    "block_lu_cpu",
    "block_size_for_cpu",
    "lu_no_pivot_cpu",
    "solve_lower_triangular",
    "solve_upper_triangular",
    "solve_via_lu_cpu",
]
