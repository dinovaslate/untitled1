#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ctypes
from ctypes import wintypes

import numpy as np
import psutil


SIZE_DIR_PATTERN = re.compile(r"ukuran\s+(\d+)\s+x\s+(\d+)", re.IGNORECASE)
DEFAULT_PIVOT_TOL = 1e-12
GPU_WARMED_UP = False


class PdhError(RuntimeError):
    pass


class PdhFmtCounterValueDouble(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class WindowsCounterSampler:
    PDH_FMT_DOUBLE = 0x00000200

    def __init__(self, counter_paths: dict[str, str]):
        self._pdh = ctypes.WinDLL("pdh")
        self._query = ctypes.c_void_p()
        self._counters: dict[str, ctypes.c_void_p] = {}
        self._counter_paths = dict(counter_paths)
        self._bind_functions()
        self._open_query()
        try:
            for name, path in self._counter_paths.items():
                self._counters[name] = self._add_counter(path)
            self._collect()
        except Exception:
            self.close()
            raise

    def _bind_functions(self) -> None:
        self._open_query_fn = self._pdh.PdhOpenQueryW
        self._open_query_fn.argtypes = [
            wintypes.LPCWSTR,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._open_query_fn.restype = wintypes.DWORD

        self._close_query_fn = self._pdh.PdhCloseQuery
        self._close_query_fn.argtypes = [ctypes.c_void_p]
        self._close_query_fn.restype = wintypes.DWORD

        add_english = getattr(self._pdh, "PdhAddEnglishCounterW", None)
        add_localized = self._pdh.PdhAddCounterW

        if add_english is not None:
            add_english.argtypes = [
                ctypes.c_void_p,
                wintypes.LPCWSTR,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            add_english.restype = wintypes.DWORD
        add_localized.argtypes = [
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        add_localized.restype = wintypes.DWORD

        self._add_counter_fn = add_english or add_localized
        self._add_counter_fallback_fn = add_localized if add_english is not None else None

        self._collect_fn = self._pdh.PdhCollectQueryData
        self._collect_fn.argtypes = [ctypes.c_void_p]
        self._collect_fn.restype = wintypes.DWORD

        self._format_fn = self._pdh.PdhGetFormattedCounterValue
        self._format_fn.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(PdhFmtCounterValueDouble),
        ]
        self._format_fn.restype = wintypes.DWORD

    def _check_status(self, status: int, action: str) -> None:
        if status != 0:
            raise PdhError(f"{action} failed with PDH status 0x{status:08X}")

    def _open_query(self) -> None:
        status = self._open_query_fn(None, None, ctypes.byref(self._query))
        self._check_status(status, "PdhOpenQueryW")

    def _add_counter(self, path: str) -> ctypes.c_void_p:
        counter = ctypes.c_void_p()
        status = self._add_counter_fn(self._query, path, None, ctypes.byref(counter))
        if status != 0 and self._add_counter_fallback_fn is not None:
            status = self._add_counter_fallback_fn(self._query, path, None, ctypes.byref(counter))
        self._check_status(status, f"Add counter {path}")
        return counter

    def _collect(self) -> None:
        status = self._collect_fn(self._query)
        self._check_status(status, "PdhCollectQueryData")

    def sample(self) -> dict[str, float]:
        self._collect()
        values: dict[str, float] = {}
        value_type = wintypes.DWORD()
        for name, counter in self._counters.items():
            value = PdhFmtCounterValueDouble()
            status = self._format_fn(counter, self.PDH_FMT_DOUBLE, ctypes.byref(value_type), ctypes.byref(value))
            self._check_status(status, f"Read counter {self._counter_paths[name]}")
            values[name] = float(value.doubleValue)
        return values

    def close(self) -> None:
        if self._query:
            self._close_query_fn(self._query)
            self._query = ctypes.c_void_p()

    def __enter__(self) -> WindowsCounterSampler:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class BackendInfo:
    backend: str
    note: str
    gpu_name: str | None = None


class PerformanceMonitor:
    def __init__(self, backend_info: BackendInfo, sample_interval: float = 0.05):
        self.backend_info = backend_info
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        self.cpu_count = max(psutil.cpu_count(logical=True) or 1, 1)
        self.monitor_gpu = backend_info.backend == "gpu" and shutil.which("nvidia-smi") is not None
        self._cache_sampler: WindowsCounterSampler | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._samples: list[dict[str, float]] = []
        self._start_wall = 0.0
        self._end_wall = 0.0
        self._start_cpu = 0.0
        self._end_cpu = 0.0

    def start(self) -> None:
        if self._started:
            return
        counter_paths = {
            "cache_copy_read_hits_percent": r"\Cache\Copy Read Hits %",
            "cache_data_map_hits_percent": r"\Cache\Data Map Hits %",
        }
        try:
            self._cache_sampler = WindowsCounterSampler(counter_paths)
        except Exception:
            self._cache_sampler = None
        self.process.cpu_percent(None)
        self._start_wall = time.perf_counter()
        cpu_times = self.process.cpu_times()
        self._start_cpu = float(cpu_times.user + cpu_times.system)
        self._thread = threading.Thread(target=self._run, name="performance-monitor", daemon=True)
        self._started = True
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._collect_sample()
            time.sleep(self.sample_interval)

    def _collect_sample(self) -> None:
        sample: dict[str, float] = {
            "rss_bytes": float(self.process.memory_info().rss),
            "cpu_percent_total_capacity": float(self.process.cpu_percent(None) / self.cpu_count),
        }
        if self._cache_sampler is not None:
            try:
                sample.update(self._cache_sampler.sample())
            except Exception:
                pass
        if self.monitor_gpu:
            gpu_snapshot = query_nvidia_smi()
            if gpu_snapshot is not None:
                sample["gpu_utilization_percent"] = gpu_snapshot["gpu_utilization_percent"]
                sample["gpu_memory_used_mb"] = gpu_snapshot["gpu_memory_used_mb"]
        self._samples.append(sample)

    def stop(self) -> dict[str, Any]:
        if not self._started:
            return {}
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._collect_sample()
        self._end_wall = time.perf_counter()
        cpu_times = self.process.cpu_times()
        self._end_cpu = float(cpu_times.user + cpu_times.system)
        if self._cache_sampler is not None:
            self._cache_sampler.close()
        metrics = self._build_metrics()
        self._started = False
        return metrics

    def _values(self, key: str, *, skip_first: bool = False) -> list[float]:
        samples = self._samples[1:] if skip_first and len(self._samples) > 1 else self._samples
        return [sample[key] for sample in samples if key in sample and math.isfinite(sample[key])]

    def _avg(self, key: str, *, skip_first: bool = False) -> float | None:
        values = self._values(key, skip_first=skip_first)
        if not values:
            return None
        return float(sum(values) / len(values))

    def _max(self, key: str) -> float | None:
        values = [sample[key] for sample in self._samples if key in sample and math.isfinite(sample[key])]
        if not values:
            return None
        return float(max(values))

    def _build_metrics(self) -> dict[str, Any]:
        wall_seconds = max(self._end_wall - self._start_wall, 0.0)
        cpu_time_seconds = max(self._end_cpu - self._start_cpu, 0.0)
        avg_cpu_percent = None
        if wall_seconds > 0:
            avg_cpu_percent = float(cpu_time_seconds / wall_seconds * 100.0 / self.cpu_count)

        copy_hit = self._avg("cache_copy_read_hits_percent", skip_first=True)
        data_map_hit = self._avg("cache_data_map_hits_percent", skip_first=True)
        hit_components = [value for value in [copy_hit, data_map_hit] if value is not None]
        cache_hit_proxy = float(max(hit_components)) if hit_components else None

        return {
            "monitor_window_seconds": wall_seconds,
            "cpu_time_seconds": cpu_time_seconds,
            "cpu_usage_avg_percent_total_capacity": avg_cpu_percent,
            "cpu_usage_peak_percent_total_capacity": self._max("cpu_percent_total_capacity"),
            "memory_peak_mb": bytes_to_mb(self._max("rss_bytes")),
            "cache_copy_read_hits_percent_avg": copy_hit,
            "cache_data_map_hits_percent_avg": data_map_hit,
            "cache_hit_percent_proxy_avg": cache_hit_proxy,
            "cache_miss_percent_proxy_avg": None if cache_hit_proxy is None else float(100.0 - cache_hit_proxy),
            "gpu_usage_avg_percent": self._avg("gpu_utilization_percent") if self.monitor_gpu else 0.0,
            "gpu_usage_peak_percent": self._max("gpu_utilization_percent") if self.monitor_gpu else 0.0,
            "gpu_memory_peak_mb": self._max("gpu_memory_used_mb") if self.monitor_gpu else 0.0,
            "sample_count": len(self._samples),
        }


def bytes_to_mb(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value / (1024.0 * 1024.0))


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


def query_nvidia_smi() -> dict[str, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    line = completed.stdout.strip().splitlines()
    if not line:
        return None

    try:
        gpu_util, memory_used = [float(part.strip()) for part in line[0].split(",")[:2]]
    except Exception:
        return None

    return {
        "gpu_utilization_percent": gpu_util,
        "gpu_memory_used_mb": memory_used,
    }


def detect_backend() -> BackendInfo:
    cuda_root = configure_cuda_environment()
    gpu_name = None
    if shutil.which("nvidia-smi") is not None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            gpu_name = completed.stdout.strip().splitlines()[0].strip() or None
        except Exception:
            gpu_name = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp  # type: ignore

        _ = cp.asarray(np.array([1.0], dtype=np.float64)) + 1.0
        _ = float(cp.asnumpy(_)[0])
        return BackendInfo(
            backend="gpu",
            note=(
                f"GPU path active via CuPy {cp.__version__}"
                + (f" using CUDA at {cuda_root}." if cuda_root is not None else ".")
            ),
            gpu_name=gpu_name,
        )
    except Exception as exc:
        reason = " ".join(f"{exc.__class__.__name__}: {exc}".split())
        note = "GPU path unavailable, CPU fallback used"
        if reason:
            note = f"{note} ({reason})"
        return BackendInfo(backend="cpu", note=note, gpu_name=gpu_name)


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


def lu_no_pivot_cpu(block: np.ndarray, pivot_tol: float = DEFAULT_PIVOT_TOL) -> tuple[np.ndarray, np.ndarray]:
    n = block.shape[0]
    if block.shape[0] != block.shape[1]:
        raise ValueError("Diagonal block must be square.")

    # Use in-place Gaussian elimination to reduce Python-loop overhead.
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


def block_lu_cpu(
    matrix: np.ndarray,
    block_size: int,
    pivot_tol: float,
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


def block_lu_gpu_device(
    matrix: np.ndarray,
    block_size: int,
    pivot_tol: float,
):
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
    pivot_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as cp  # type: ignore

    combined_lu_gpu, L_gpu, U_gpu = block_lu_gpu_device(
        matrix=matrix,
        block_size=block_size,
        pivot_tol=pivot_tol,
    )
    return cp.asnumpy(combined_lu_gpu), cp.asnumpy(L_gpu), cp.asnumpy(U_gpu)


def solve_via_lu(L: np.ndarray, U: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    y = solve_lower_triangular(L, rhs, unit_diagonal=True)
    return solve_upper_triangular(U, y)


def solve_via_lu_gpu_device(L, U, rhs: np.ndarray):
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


def relative_error_inf(actual: np.ndarray, reference: np.ndarray) -> float:
    numerator = np.linalg.norm(actual - reference, ord=np.inf)
    denominator = np.linalg.norm(reference, ord=np.inf)
    if float(denominator) == 0.0:
        return 0.0 if float(numerator) == 0.0 else float("inf")
    return float(numerator / denominator)


def normalized_residual_inf(A: np.ndarray, x: np.ndarray, rhs: np.ndarray) -> float:
    numerator = np.linalg.norm(A @ x - rhs, ord=np.inf)
    denominator = (
        np.linalg.norm(A, ord=np.inf) * np.linalg.norm(x, ord=np.inf)
        + np.linalg.norm(rhs, ord=np.inf)
    )
    if float(denominator) == 0.0:
        return 0.0 if float(numerator) == 0.0 else float("inf")
    return float(numerator / denominator)


def block_size_for(n: int, backend: str) -> int:
    if backend == "gpu":
        if n <= 32:
            return max(2, n)
        if n <= 256:
            return 16
        if n <= 1024:
            return 32
        return 64

    if n <= 32:
        return 4
    if n <= 256:
        return 8
    if n <= 1024:
        return 16
    return 32


def load_case(case_dir: Path, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.loadtxt(case_dir / f"A_{size}.csv", delimiter=",", dtype=np.float64)
    b = np.loadtxt(case_dir / f"b_{size}.csv", delimiter=",", dtype=np.float64)
    c = np.loadtxt(case_dir / f"c_{size}.csv", delimiter=",", dtype=np.float64)
    return A, b, c


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.18e")


def save_vector_csv(path: Path, vector: np.ndarray) -> None:
    np.savetxt(path, vector.reshape(-1, 1), delimiter=",", fmt="%.18e")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if not math.isfinite(numeric) else numeric
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def write_case_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        f"# {metrics['size_label']}",
        "",
        f"- Backend: `{metrics['backend']}`",
        f"- Strategy: `{metrics['execution_strategy']}`",
        f"- Catatan backend: {metrics['backend_note']}",
        f"- Block size: `{metrics['block_size']}`",
        f"- Relative error `||LU - A||inf / ||A||inf`: `{metrics['lu_relative_error_inf']:.6e}`",
        f"- Relative error `x_b` vs `numpy.linalg.solve`: `{metrics['x_b_relative_error_inf_vs_numpy']:.6e}`",
        f"- Relative error `x_c` vs `numpy.linalg.solve`: `{metrics['x_c_relative_error_inf_vs_numpy']:.6e}`",
        f"- Normalized residual `Ax=b`: `{metrics['x_b_normalized_residual_inf']:.6e}`",
        f"- Normalized residual `Ax=c`: `{metrics['x_c_normalized_residual_inf']:.6e}`",
        f"- Elapsed time algoritma: `{metrics['elapsed_seconds_algorithm']:.6f}` s",
        f"- CPU time proses: `{metrics['cpu_time_seconds']:.6f}` s",
        f"- CPU usage rata-rata: `{format_optional(metrics['cpu_usage_avg_percent_total_capacity'])}` %",
        f"- Peak memory: `{format_optional(metrics['memory_peak_mb'])}` MB",
        f"- GPU usage puncak: `{format_optional(metrics['gpu_usage_peak_percent'])}` %",
        f"- Peak GPU memory: `{format_optional(metrics['gpu_memory_peak_mb'])}` MB",
        f"- Cache hit proxy rata-rata: `{format_optional(metrics['cache_hit_percent_proxy_avg'])}` %",
        f"- Cache miss proxy rata-rata: `{format_optional(metrics['cache_miss_percent_proxy_avg'])}` %",
        "",
        "Catatan:",
        "- Error solusi `x` dihitung terhadap `numpy.linalg.solve` karena solusi eksak tidak tersedia di dataset.",
        "- Cache hit/miss diambil sebagai proxy dari Windows Cache Manager (`Copy Read Hits %` dan `Data Map Hits %`), bukan hardware CPU cache per proses. Proxy memakai hit tertinggi agar counter yang sedang tidak aktif tidak terbaca sebagai miss palsu.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def discover_cases(input_root: Path, requested_sizes: list[int] | None) -> list[tuple[int, Path]]:
    cases: list[tuple[int, Path]] = []
    requested = set(requested_sizes or [])

    for entry in sorted(input_root.iterdir()):
        if not entry.is_dir():
            continue
        match = SIZE_DIR_PATTERN.fullmatch(entry.name.strip())
        if not match:
            continue
        rows, cols = int(match.group(1)), int(match.group(2))
        if rows != cols:
            continue
        if requested and rows not in requested:
            continue
        cases.append((rows, entry))

    if requested:
        found = {size for size, _ in cases}
        missing = sorted(requested - found)
        if missing:
            missing_text = ", ".join(str(size) for size in missing)
            raise SystemExit(f"Ukuran tidak ditemukan: {missing_text}")

    return sorted(cases, key=lambda item: item[0])


def write_root_readme(path: Path, backend_info: BackendInfo) -> None:
    content = f"""# Hasil Block LU

Struktur folder:
- `ukuran_<n>x<n>/L.csv`
- `ukuran_<n>x<n>/U.csv`
- `ukuran_<n>x<n>/x_b.csv`
- `ukuran_<n>x<n>/x_c.csv`
- `ukuran_<n>x<n>/metrics.json`
- `ukuran_<n>x<n>/laporan.md`
- `ringkasan.csv`

Alasan struktur:
- Empat file utama tetap dipisah supaya mudah dicek atau dipakai ulang.
- Metrik dipisah ke `metrics.json` dan `laporan.md` agar data matriks tidak bercampur dengan narasi.
- Nama folder memakai format `ukuran_<n>x<n>` supaya konsisten dan mudah diolah skrip.

Definisi metrik:
- `lu_relative_error_inf`: `||LU - A||inf / ||A||inf`
- `x_*_relative_error_inf_vs_numpy`: error relatif solusi LU terhadap `numpy.linalg.solve`
- `x_*_normalized_residual_inf`: `||Ax - rhs||inf / (||A||inf ||x||inf + ||rhs||inf)`
- `cache_*`: proxy Windows Cache Manager, bukan hardware CPU cache per proses. Nilai proxy memakai hit tertinggi dari `Copy Read Hits %` dan `Data Map Hits %` agar counter yang tidak aktif tidak terbaca sebagai miss palsu.
- `execution_strategy`: menandakan path optimasi yang dipakai. CPU memakai block size yang lebih kecil agar locality panel lebih baik, sedangkan GPU menyelesaikan forward/back substitution tetap di device untuk mengurangi transfer host-device.

Backend:
- `{backend_info.backend}`
- {backend_info.note}
"""
    path.write_text(content, encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "size",
        "size_label",
        "block_size",
        "backend",
        "execution_strategy",
        "lu_relative_error_inf",
        "x_b_relative_error_inf_vs_numpy",
        "x_c_relative_error_inf_vs_numpy",
        "x_b_normalized_residual_inf",
        "x_c_normalized_residual_inf",
        "elapsed_seconds_algorithm",
        "cpu_time_seconds",
        "cpu_usage_avg_percent_total_capacity",
        "memory_peak_mb",
        "gpu_usage_peak_percent",
        "gpu_memory_peak_mb",
        "cache_hit_percent_proxy_avg",
        "cache_miss_percent_proxy_avg",
        "backend_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: to_builtin(row.get(field)) for field in fieldnames})


def run_case(
    size: int,
    case_dir: Path,
    output_root: Path,
    backend_info: BackendInfo,
    pivot_tol: float,
) -> dict[str, Any]:
    A, b, c = load_case(case_dir, size)
    block_size = block_size_for(size, backend_info.backend)
    if backend_info.backend == "gpu":
        warm_up_gpu_backend()

    monitor = PerformanceMonitor(backend_info)
    monitor.start()

    start_algorithm = time.perf_counter()
    if backend_info.backend == "gpu":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp  # type: ignore

        combined_lu_gpu, L_gpu, U_gpu = block_lu_gpu_device(
            A,
            block_size=block_size,
            pivot_tol=pivot_tol,
        )
        x_b_gpu = solve_via_lu_gpu_device(L_gpu, U_gpu, b)
        x_c_gpu = solve_via_lu_gpu_device(L_gpu, U_gpu, c)
        cp.cuda.Stream.null.synchronize()
        combined_lu = cp.asnumpy(combined_lu_gpu)
        L = cp.asnumpy(L_gpu)
        U = cp.asnumpy(U_gpu)
        x_b = cp.asnumpy(x_b_gpu)
        x_c = cp.asnumpy(x_c_gpu)
        execution_strategy = "optimized_block_lu_gpu"
    else:
        combined_lu, L, U = block_lu_cpu(A, block_size=block_size, pivot_tol=pivot_tol)
        x_b = solve_via_lu(L, U, b)
        x_c = solve_via_lu(L, U, c)
        execution_strategy = "optimized_block_lu_cpu"
    elapsed_algorithm = time.perf_counter() - start_algorithm
    runtime_metrics = monitor.stop()

    lu_relative_error = relative_error_inf(L @ U, A)
    x_b_reference = np.linalg.solve(A, b)
    x_c_reference = np.linalg.solve(A, c)

    metrics: dict[str, Any] = {
        "size": size,
        "size_label": f"ukuran_{size}x{size}",
        "input_directory": str(case_dir),
        "output_directory": str(output_root / f"ukuran_{size}x{size}"),
        "backend": backend_info.backend,
        "backend_note": backend_info.note,
        "gpu_name": backend_info.gpu_name,
        "block_size": block_size,
        "execution_strategy": execution_strategy,
        "pivot_tolerance": pivot_tol,
        "lu_relative_error_inf": lu_relative_error,
        "x_b_relative_error_inf_vs_numpy": relative_error_inf(x_b, x_b_reference),
        "x_c_relative_error_inf_vs_numpy": relative_error_inf(x_c, x_c_reference),
        "x_b_normalized_residual_inf": normalized_residual_inf(A, x_b, b),
        "x_c_normalized_residual_inf": normalized_residual_inf(A, x_c, c),
        "elapsed_seconds_algorithm": elapsed_algorithm,
        "combined_lu_shape": list(combined_lu.shape),
    }
    metrics.update(runtime_metrics)

    case_output_dir = output_root / f"ukuran_{size}x{size}"
    case_output_dir.mkdir(parents=True, exist_ok=True)
    save_matrix_csv(case_output_dir / "L.csv", L)
    save_matrix_csv(case_output_dir / "U.csv", U)
    save_vector_csv(case_output_dir / "x_b.csv", x_b)
    save_vector_csv(case_output_dir / "x_c.csv", x_c)
    write_json(case_output_dir / "metrics.json", metrics)
    write_case_report(case_output_dir / "laporan.md", metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jalankan block LU factorization untuk semua matriks CSV dan tulis hasil ke folder hasil."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.cwd(),
        help="Folder root yang berisi directory 'ukuran * x *'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd() / "hasil",
        help="Folder output untuk hasil.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=None,
        help="Filter ukuran tertentu, contoh: --sizes 32 128 512",
    )
    parser.add_argument(
        "--pivot-tol",
        type=float,
        default=DEFAULT_PIVOT_TOL,
        help="Ambang pivot minimum untuk LU tanpa pivoting.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Pilih backend eksekusi. Default: auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    detected_backend = detect_backend()
    if args.backend == "auto":
        backend_info = detected_backend
    elif args.backend == "cpu":
        backend_info = BackendInfo(
            backend="cpu",
            note=(
                f"CPU forced by CLI override. Auto-detected backend was "
                f"{detected_backend.backend}: {detected_backend.note}"
            ),
            gpu_name=detected_backend.gpu_name,
        )
    else:
        if detected_backend.backend != "gpu":
            raise SystemExit(
                "Backend GPU diminta, tetapi GPU path tidak siap. "
                f"Auto-detection note: {detected_backend.note}"
            )
        backend_info = detected_backend
    cases = discover_cases(input_root, args.sizes)
    if not cases:
        raise SystemExit("Tidak ada kasus matriks yang ditemukan.")

    summary_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for size, case_dir in cases:
        try:
            metrics = run_case(
                size=size,
                case_dir=case_dir,
                output_root=output_root,
                backend_info=backend_info,
                pivot_tol=args.pivot_tol,
            )
            summary_rows.append(metrics)
            print(
                f"[OK] {size}x{size} | backend={metrics['backend']} | "
                f"elapsed={metrics['elapsed_seconds_algorithm']:.6f}s | "
                f"lu_err={metrics['lu_relative_error_inf']:.6e}"
            )
        except Exception as exc:
            message = f"{size}x{size}: {exc}"
            failures.append(message)
            print(f"[FAIL] {message}")

    write_summary_csv(output_root / "ringkasan.csv", summary_rows)
    write_root_readme(output_root / "README.md", backend_info)

    if failures:
        failure_path = output_root / "gagal.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"Ada {len(failures)} kasus gagal. Lihat {failure_path}")

    print(f"Semua hasil tersimpan di: {output_root}")


if __name__ == "__main__":
    main()
