#!/usr/bin/env python3
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from block_lu_cpu import DEFAULT_PIVOT_TOL, lu_no_pivot_cpu


GPU_WARMED_UP = False


def parse_cuda_version(name: str) -> tuple[int, ...]:
    version_text = name.lstrip("vV")
    try:
        return tuple(int(part) for part in version_text.split("."))
    except ValueError:
        return (0,)


def prepend_process_path(path: Path) -> None:
    if not path.exists():
        return

    current = os.environ.get("Path") or os.environ.get("PATH") or ""
    current_parts = [part for part in current.split(os.pathsep) if part]
    normalized_current = {
        os.path.normcase(os.path.normpath(part))
        for part in current_parts
    }
    normalized_candidate = os.path.normcase(os.path.normpath(str(path)))
    if normalized_candidate in normalized_current:
        return

    updated = str(path) if not current else f"{path}{os.pathsep}{current}"
    os.environ["Path"] = updated
    os.environ["PATH"] = updated


def find_cuda_toolkit_root() -> Path | None:
    explicit = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path

    default_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if default_base.exists():
        candidates = [
            entry for entry in default_base.iterdir()
            if entry.is_dir() and entry.name.lower().startswith("v")
        ]
        if candidates:
            return max(candidates, key=lambda entry: parse_cuda_version(entry.name))

    return None


def configure_cuda_environment() -> Path | None:
    root = find_cuda_toolkit_root()
    if root is None:
        return None

    os.environ.setdefault("CUDA_PATH", str(root))
    os.environ.setdefault("CUDA_HOME", str(root))
    prepend_process_path(root / "bin")
    prepend_process_path(root / "libnvvp")
    return root


def block_size_for_gpu(n: int) -> int:
    if n <= 32:
        return max(2, n)
    if n <= 256:
        return 16
    if n <= 1024:
        return 32
    return 64


def warm_up_gpu_backend() -> None:
    global GPU_WARMED_UP
    if GPU_WARMED_UP:
        return

    configure_cuda_environment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp  # type: ignore
        from cupyx.scipy.linalg import solve_triangular  # type: ignore

    A = cp.eye(4, dtype=cp.float64)
    rhs = cp.ones((4, 1), dtype=cp.float64)
    _ = solve_triangular(
        A,
        rhs,
        lower=True,
        unit_diagonal=False,
        overwrite_b=False,
        check_finite=False,
    )
    _ = A @ A
    cp.cuda.Stream.null.synchronize()
    GPU_WARMED_UP = True


def block_lu_gpu_device(
    matrix: np.ndarray,
    block_size: int,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
):
    configure_cuda_environment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp  # type: ignore
        from cupyx.scipy.linalg import solve_triangular  # type: ignore

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    host_matrix = np.asarray(matrix, dtype=np.float64)
    A = cp.asarray(host_matrix)
    n = host_matrix.shape[0]

    for k in range(0, n, block_size):
        b = min(block_size, n - k)
        k_end = k + b

        diag_block = cp.asnumpy(A[k:k_end, k:k_end])
        L11, U11 = lu_no_pivot_cpu(diag_block, pivot_tol=pivot_tol)
        A[k:k_end, k:k_end] = cp.asarray(np.tril(L11, k=-1) + U11)

        if k_end >= n:
            continue

        L11_gpu = cp.asarray(L11)
        U11_gpu = cp.asarray(U11)

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

        A[k_end:n, k_end:n] -= L21 @ U12

    cp.cuda.Stream.null.synchronize()
    L = cp.tril(A, k=-1) + cp.eye(n, dtype=A.dtype)
    U = cp.triu(A)
    return A, L, U


def block_lu_gpu(
    matrix: np.ndarray,
    block_size: int,
    pivot_tol: float = DEFAULT_PIVOT_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    configure_cuda_environment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp  # type: ignore

    combined_lu_gpu, L_gpu, U_gpu = block_lu_gpu_device(
        matrix=matrix,
        block_size=block_size,
        pivot_tol=pivot_tol,
    )
    return cp.asnumpy(combined_lu_gpu), cp.asnumpy(L_gpu), cp.asnumpy(U_gpu)


def solve_via_lu_gpu_device(L, U, rhs: np.ndarray):
    configure_cuda_environment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp  # type: ignore
        from cupyx.scipy.linalg import solve_triangular  # type: ignore

    rhs_gpu = cp.asarray(np.asarray(rhs, dtype=np.float64))
    y = solve_triangular(
        L,
        rhs_gpu,
        lower=True,
        unit_diagonal=True,
        overwrite_b=False,
        check_finite=False,
    )
    x = solve_triangular(
        U,
        y,
        lower=False,
        unit_diagonal=False,
        overwrite_b=False,
        check_finite=False,
    )
    return x


def solve_via_lu_gpu(L, U, rhs: np.ndarray) -> np.ndarray:
    configure_cuda_environment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp  # type: ignore

    return cp.asnumpy(solve_via_lu_gpu_device(L, U, rhs))


__all__ = [
    "GPU_WARMED_UP",
    "block_lu_gpu",
    "block_lu_gpu_device",
    "block_size_for_gpu",
    "configure_cuda_environment",
    "find_cuda_toolkit_root",
    "parse_cuda_version",
    "prepend_process_path",
    "solve_via_lu_gpu",
    "solve_via_lu_gpu_device",
    "warm_up_gpu_backend",
]
