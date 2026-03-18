#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from algorithm_memory import AlgorithmMemoryTracker, snapshot_memory
from block_lu_cpu import block_lu_cpu, block_size_for_cpu, solve_via_lu_cpu
from block_lu_gpu import (
    block_lu_gpu_device,
    block_size_for_gpu,
    solve_via_lu_gpu_device,
    warm_up_gpu_backend,
)
from generate_hasil_lu import (
    BackendInfo,
    PerformanceMonitor,
    detect_backend,
    discover_cases,
    load_case,
    normalized_residual_inf,
    relative_error_inf,
)
from thomas_algorithm import (
    extract_four_diagonal_bands,
    factorize_four_diagonal,
    four_diagonal_inf_norm,
    lu_relative_error_inf_four_diagonal,
    normalized_residual_inf_four_diagonal,
    solve_four_diagonal_factored,
)


PIVOT_TOL = 1e-12
DEFAULT_SIZES = [8, 16, 32, 128, 256, 512]
TIMING_BATCHES = 5
TARGET_BATCH_SECONDS = 0.30
MAX_BATCH_REPETITIONS = 4000
MIN_BATCH_REPETITIONS = 5


@dataclass(frozen=True)
class BandedCase:
    size: int
    case_dir: Path
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    diagonal: np.ndarray
    upper_1: np.ndarray
    upper_2: np.ndarray
    x_b_reference: np.ndarray
    x_c_reference: np.ndarray
    condition_number_1: float
    condition_number_2: float
    condition_number_inf: float


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    display_name: str
    note: str
    execute: Callable[[BandedCase], dict[str, Any]]
    profile_backend: str


def bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if not math.isfinite(numeric) else numeric
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 1000.0:.4f}"


def fmt_mb(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def fmt_sci(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3e}"


def load_banded_case(case_dir: Path, size: int) -> BandedCase:
    A, b, c = load_case(case_dir, size)
    lower, diagonal, upper_1, upper_2 = extract_four_diagonal_bands(A, tol=PIVOT_TOL)
    x_b_reference = np.linalg.solve(A, b)
    x_c_reference = np.linalg.solve(A, c)
    return BandedCase(
        size=size,
        case_dir=case_dir,
        A=A,
        b=b,
        c=c,
        lower=lower,
        diagonal=diagonal,
        upper_1=upper_1,
        upper_2=upper_2,
        x_b_reference=x_b_reference,
        x_c_reference=x_c_reference,
        condition_number_1=float(np.linalg.cond(A, 1)),
        condition_number_2=float(np.linalg.cond(A, 2)),
        condition_number_inf=float(np.linalg.cond(A, np.inf)),
    )


def execute_thomas4(case: BandedCase) -> dict[str, Any]:
    tracker = AlgorithmMemoryTracker()
    snapshot_memory(
        tracker,
        case.lower,
        case.diagonal,
        case.upper_1,
        case.upper_2,
        case.b,
        case.c,
        label="execute_thomas4:inputs",
    )
    multipliers, u_diagonal, u_upper_1, u_upper_2 = factorize_four_diagonal(
        case.lower,
        case.diagonal,
        case.upper_1,
        case.upper_2,
        pivot_tol=PIVOT_TOL,
        memory_tracker=tracker,
    )
    x_b = solve_four_diagonal_factored(
        multipliers,
        u_diagonal,
        u_upper_1,
        u_upper_2,
        case.b,
        memory_tracker=tracker,
    )
    snapshot_memory(
        tracker,
        case.lower,
        case.diagonal,
        case.upper_1,
        case.upper_2,
        case.b,
        case.c,
        multipliers,
        u_diagonal,
        u_upper_1,
        u_upper_2,
        x_b,
        label="execute_thomas4:after_x_b",
    )
    x_c = solve_four_diagonal_factored(
        multipliers,
        u_diagonal,
        u_upper_1,
        u_upper_2,
        case.c,
        memory_tracker=tracker,
    )
    snapshot_memory(
        tracker,
        case.lower,
        case.diagonal,
        case.upper_1,
        case.upper_2,
        case.b,
        case.c,
        multipliers,
        u_diagonal,
        u_upper_1,
        u_upper_2,
        x_b,
        x_c,
        label="execute_thomas4:final",
    )
    return {
        "storage_model": "4-diagonal compact",
        "algorithm_peak_memory_bytes": tracker.peak_bytes,
        "x_b": x_b,
        "x_c": x_c,
        "lu_relative_error_inf": lu_relative_error_inf_four_diagonal(
            case.lower,
            case.diagonal,
            case.upper_1,
            case.upper_2,
            multipliers,
            u_diagonal,
            u_upper_1,
            u_upper_2,
        ),
        "x_b_normalized_residual_inf": normalized_residual_inf_four_diagonal(
            case.lower,
            case.diagonal,
            case.upper_1,
            case.upper_2,
            x_b,
            case.b,
        ),
        "x_c_normalized_residual_inf": normalized_residual_inf_four_diagonal(
            case.lower,
            case.diagonal,
            case.upper_1,
            case.upper_2,
            x_c,
            case.c,
        ),
        "matrix_inf_norm": four_diagonal_inf_norm(
            case.lower,
            case.diagonal,
            case.upper_1,
            case.upper_2,
        ),
        "backend_display": "CPU",
        "block_size": None,
    }


def execute_block_lu(case: BandedCase, *, prefer_gpu: bool) -> dict[str, Any]:
    tracker = AlgorithmMemoryTracker()
    snapshot_memory(tracker, case.A, case.b, case.c, label="execute_block_lu:inputs")

    if prefer_gpu:
        import cupy as cp  # type: ignore

        block_size = block_size_for_gpu(case.size)
        combined_gpu, L_gpu, U_gpu = block_lu_gpu_device(
            case.A,
            block_size=block_size,
            pivot_tol=PIVOT_TOL,
            memory_tracker=tracker,
        )
        x_b_gpu = solve_via_lu_gpu_device(L_gpu, U_gpu, case.b, memory_tracker=tracker)
        snapshot_memory(
            tracker,
            case.A,
            case.b,
            case.c,
            combined_gpu,
            L_gpu,
            U_gpu,
            x_b_gpu,
            label="execute_block_lu_gpu:after_x_b",
        )
        x_c_gpu = solve_via_lu_gpu_device(L_gpu, U_gpu, case.c, memory_tracker=tracker)
        snapshot_memory(
            tracker,
            case.A,
            case.b,
            case.c,
            combined_gpu,
            L_gpu,
            U_gpu,
            x_b_gpu,
            x_c_gpu,
            label="execute_block_lu_gpu:final",
        )
        x_b = cp.asnumpy(x_b_gpu)
        x_c = cp.asnumpy(x_c_gpu)
        combined = cp.asnumpy(combined_gpu)
        L = cp.asnumpy(L_gpu)
        U = cp.asnumpy(U_gpu)
        backend_display = "CPU+GPU"
    else:
        block_size = block_size_for_cpu(case.size)
        combined, L, U = block_lu_cpu(
            case.A,
            block_size=block_size,
            pivot_tol=PIVOT_TOL,
            memory_tracker=tracker,
        )
        x_b = solve_via_lu_cpu(L, U, case.b, memory_tracker=tracker)
        snapshot_memory(
            tracker,
            case.A,
            case.b,
            case.c,
            combined,
            L,
            U,
            x_b,
            label="execute_block_lu_cpu:after_x_b",
        )
        x_c = solve_via_lu_cpu(L, U, case.c, memory_tracker=tracker)
        snapshot_memory(
            tracker,
            case.A,
            case.b,
            case.c,
            combined,
            L,
            U,
            x_b,
            x_c,
            label="execute_block_lu_cpu:final",
        )
        backend_display = "CPU"

    return {
        "storage_model": "dense",
        "algorithm_peak_memory_bytes": tracker.peak_bytes,
        "combined_lu": combined,
        "L": L,
        "U": U,
        "x_b": x_b,
        "x_c": x_c,
        "lu_relative_error_inf": relative_error_inf(L @ U, case.A),
        "x_b_normalized_residual_inf": normalized_residual_inf(case.A, x_b, case.b),
        "x_c_normalized_residual_inf": normalized_residual_inf(case.A, x_c, case.c),
        "backend_display": backend_display,
        "block_size": block_size,
    }


def choose_batch_repetitions(single_run_seconds: float) -> int:
    if single_run_seconds <= 0.0:
        return MIN_BATCH_REPETITIONS
    estimated = int(math.ceil(TARGET_BATCH_SECONDS / single_run_seconds))
    return max(MIN_BATCH_REPETITIONS, min(MAX_BATCH_REPETITIONS, estimated))


def benchmark_method(case: BandedCase, method: MethodSpec) -> dict[str, Any]:
    if method.slug == "block_lu" and method.profile_backend == "gpu" and case.size >= 512:
        warm_up_gpu_backend()

    pilot_start = time.perf_counter()
    pilot_result = method.execute(case)
    pilot_elapsed = time.perf_counter() - pilot_start
    batch_repetitions = choose_batch_repetitions(pilot_elapsed)

    timing_per_run: list[float] = []
    for _ in range(TIMING_BATCHES):
        start = time.perf_counter()
        last_result: dict[str, Any] | None = None
        for _ in range(batch_repetitions):
            last_result = method.execute(case)
        elapsed = time.perf_counter() - start
        timing_per_run.append(elapsed / batch_repetitions)
        del last_result

    monitor = PerformanceMonitor(
        BackendInfo(
            backend="gpu" if pilot_result.get("backend_display") == "CPU+GPU" else "cpu",
            note=method.note,
        )
    )
    monitor.start()
    profile_start = time.perf_counter()
    profile_result: dict[str, Any] | None = None
    for _ in range(batch_repetitions):
        profile_result = method.execute(case)
    profile_elapsed_total = time.perf_counter() - profile_start
    monitor_metrics = monitor.stop()

    if profile_result is None:
        raise RuntimeError(f"No profile result produced for method {method.slug}.")

    x_b = np.asarray(profile_result["x_b"], dtype=np.float64)
    x_c = np.asarray(profile_result["x_c"], dtype=np.float64)

    return {
        "method": method.slug,
        "method_display_name": method.display_name,
        "note": method.note,
        "repetitions_per_batch": batch_repetitions,
        "timing_batches": TIMING_BATCHES,
        "elapsed_compute_median_seconds": float(statistics.median(timing_per_run)),
        "elapsed_compute_mean_seconds": float(statistics.mean(timing_per_run)),
        "elapsed_profile_batch_seconds": profile_elapsed_total,
        "elapsed_profile_run_seconds": float(profile_elapsed_total / batch_repetitions),
        "algorithm_peak_memory_mb": bytes_to_mb(profile_result.get("algorithm_peak_memory_bytes")),
        "process_peak_memory_mb": monitor_metrics.get("memory_peak_mb"),
        "cpu_time_seconds": monitor_metrics.get("cpu_time_seconds"),
        "cpu_usage_avg_percent_total_capacity": monitor_metrics.get("cpu_usage_avg_percent_total_capacity"),
        "cpu_usage_peak_percent_total_capacity": monitor_metrics.get("cpu_usage_peak_percent_total_capacity"),
        "gpu_usage_avg_percent": monitor_metrics.get("gpu_usage_avg_percent"),
        "gpu_usage_peak_percent": monitor_metrics.get("gpu_usage_peak_percent"),
        "gpu_memory_peak_mb": monitor_metrics.get("gpu_memory_peak_mb"),
        "cache_hit_percent_proxy_avg": monitor_metrics.get("cache_hit_percent_proxy_avg"),
        "cache_miss_percent_proxy_avg": monitor_metrics.get("cache_miss_percent_proxy_avg"),
        "storage_model": profile_result.get("storage_model"),
        "backend_display": profile_result.get("backend_display"),
        "block_size": profile_result.get("block_size"),
        "lu_relative_error_inf": float(profile_result["lu_relative_error_inf"]),
        "x_b_relative_error_inf": relative_error_inf(x_b, case.x_b_reference),
        "x_c_relative_error_inf": relative_error_inf(x_c, case.x_c_reference),
        "x_b_normalized_residual_inf": float(profile_result["x_b_normalized_residual_inf"]),
        "x_c_normalized_residual_inf": float(profile_result["x_c_normalized_residual_inf"]),
        "x_b": x_b,
        "x_c": x_c,
    }


def build_wide_row(case: BandedCase, block_lu_row: dict[str, Any], thomas_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "size": case.size,
        "size_label": f"ukuran_{case.size}x{case.size}",
        "lower_bandwidth_p": 1,
        "upper_bandwidth_q": 2,
        "condition_number_1": case.condition_number_1,
        "condition_number_2": case.condition_number_2,
        "condition_number_inf": case.condition_number_inf,
        "block_lu_backend": block_lu_row["backend_display"],
        "block_lu_block_size": block_lu_row["block_size"],
        "block_lu_elapsed_median_seconds": block_lu_row["elapsed_compute_median_seconds"],
        "block_lu_elapsed_mean_seconds": block_lu_row["elapsed_compute_mean_seconds"],
        "block_lu_algorithm_peak_memory_mb": block_lu_row["algorithm_peak_memory_mb"],
        "block_lu_process_peak_memory_mb": block_lu_row["process_peak_memory_mb"],
        "block_lu_cpu_usage_avg_percent_total_capacity": block_lu_row["cpu_usage_avg_percent_total_capacity"],
        "block_lu_cpu_usage_peak_percent_total_capacity": block_lu_row["cpu_usage_peak_percent_total_capacity"],
        "block_lu_gpu_usage_peak_percent": block_lu_row["gpu_usage_peak_percent"],
        "block_lu_gpu_memory_peak_mb": block_lu_row["gpu_memory_peak_mb"],
        "block_lu_cache_hit_percent_proxy_avg": block_lu_row["cache_hit_percent_proxy_avg"],
        "block_lu_cache_miss_percent_proxy_avg": block_lu_row["cache_miss_percent_proxy_avg"],
        "block_lu_lu_relative_error_inf": block_lu_row["lu_relative_error_inf"],
        "block_lu_x_b_relative_error_inf": block_lu_row["x_b_relative_error_inf"],
        "block_lu_x_c_relative_error_inf": block_lu_row["x_c_relative_error_inf"],
        "block_lu_x_b_normalized_residual_inf": block_lu_row["x_b_normalized_residual_inf"],
        "block_lu_x_c_normalized_residual_inf": block_lu_row["x_c_normalized_residual_inf"],
        "thomas4_backend": thomas_row["backend_display"],
        "thomas4_elapsed_median_seconds": thomas_row["elapsed_compute_median_seconds"],
        "thomas4_elapsed_mean_seconds": thomas_row["elapsed_compute_mean_seconds"],
        "thomas4_algorithm_peak_memory_mb": thomas_row["algorithm_peak_memory_mb"],
        "thomas4_process_peak_memory_mb": thomas_row["process_peak_memory_mb"],
        "thomas4_cpu_usage_avg_percent_total_capacity": thomas_row["cpu_usage_avg_percent_total_capacity"],
        "thomas4_cpu_usage_peak_percent_total_capacity": thomas_row["cpu_usage_peak_percent_total_capacity"],
        "thomas4_gpu_usage_peak_percent": thomas_row["gpu_usage_peak_percent"],
        "thomas4_gpu_memory_peak_mb": thomas_row["gpu_memory_peak_mb"],
        "thomas4_cache_hit_percent_proxy_avg": thomas_row["cache_hit_percent_proxy_avg"],
        "thomas4_cache_miss_percent_proxy_avg": thomas_row["cache_miss_percent_proxy_avg"],
        "thomas4_lu_relative_error_inf": thomas_row["lu_relative_error_inf"],
        "thomas4_x_b_relative_error_inf": thomas_row["x_b_relative_error_inf"],
        "thomas4_x_c_relative_error_inf": thomas_row["x_c_relative_error_inf"],
        "thomas4_x_b_normalized_residual_inf": thomas_row["x_b_normalized_residual_inf"],
        "thomas4_x_c_normalized_residual_inf": thomas_row["x_c_normalized_residual_inf"],
        "thomas4_note": thomas_row["note"],
        "block_lu_note": block_lu_row["note"],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must be non-empty.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def write_vector_txt(path: Path, vector: np.ndarray) -> None:
    np.savetxt(path, np.asarray(vector, dtype=np.float64).reshape(-1, 1), fmt="%.18e")


def build_html(rows: list[dict[str, Any]], output_path: Path, summary_path: Path) -> None:
    runtime_rows: list[str] = []
    accuracy_rows: list[str] = []
    resource_rows: list[str] = []
    conditioning_rows: list[str] = []

    for row in rows:
        size_label = f"{int(row['size'])} x {int(row['size'])}"
        runtime_rows.append(
            "<tr>"
            f'<td><span class="size-pill">{size_label}</span></td>'
            f"<td>{html.escape(str(row['block_lu_backend']))}</td>"
            f"<td>{fmt_ms(row['block_lu_elapsed_median_seconds'])}</td>"
            f"<td>{fmt_ms(row['thomas4_elapsed_median_seconds'])}</td>"
            f"<td>{fmt_mb(row['block_lu_algorithm_peak_memory_mb'])}</td>"
            f"<td>{fmt_mb(row['thomas4_algorithm_peak_memory_mb'])}</td>"
            "</tr>"
        )
        accuracy_rows.append(
            "<tr>"
            f'<td><span class="size-pill">{size_label}</span></td>'
            f"<td>{fmt_sci(row['block_lu_lu_relative_error_inf'])}</td>"
            f"<td>{fmt_sci(row['block_lu_x_b_relative_error_inf'])}</td>"
            f"<td>{fmt_sci(row['block_lu_x_c_relative_error_inf'])}</td>"
            f"<td>{fmt_sci(row['thomas4_lu_relative_error_inf'])}</td>"
            f"<td>{fmt_sci(row['thomas4_x_b_relative_error_inf'])}</td>"
            f"<td>{fmt_sci(row['thomas4_x_c_relative_error_inf'])}</td>"
            "</tr>"
        )
        resource_rows.append(
            "<tr>"
            f'<td><span class="size-pill">{size_label}</span></td>'
            f"<td>{fmt_percent(row['block_lu_cpu_usage_avg_percent_total_capacity'])}</td>"
            f"<td>{fmt_percent(row['thomas4_cpu_usage_avg_percent_total_capacity'])}</td>"
            f"<td>{fmt_percent(row['block_lu_gpu_usage_peak_percent'])}</td>"
            f"<td>{fmt_percent(row['thomas4_gpu_usage_peak_percent'])}</td>"
            f"<td>{fmt_mb(row['block_lu_process_peak_memory_mb'])}</td>"
            f"<td>{fmt_mb(row['thomas4_process_peak_memory_mb'])}</td>"
            "</tr>"
        )
        conditioning_rows.append(
            "<tr>"
            f'<td><span class="size-pill">{size_label}</span></td>'
            f"<td>{fmt_sci(row['condition_number_1'])}</td>"
            f"<td>{fmt_sci(row['condition_number_2'])}</td>"
            f"<td>{fmt_sci(row['condition_number_inf'])}</td>"
            "</tr>"
        )

    fastest_row = min(rows, key=lambda item: item["thomas4_elapsed_median_seconds"])
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>4-Diagonal Benchmark on Given Data</title>
  <style>
    :root {{
      --bg: linear-gradient(135deg, #f7f1e9 0%, #fdfbf7 48%, #e7efe9 100%);
      --ink: #172031;
      --muted: #5f6a7d;
      --panel: rgba(255, 255, 255, 0.92);
      --line: rgba(23, 32, 49, 0.12);
      --accent: #0d6d5f;
      --accent-soft: rgba(13, 109, 95, 0.12);
      --shadow: 0 18px 60px rgba(23, 32, 49, 0.12);
      --radius: 24px;
      font-family: "Segoe UI", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--bg); }}
    .page {{ width: min(1180px, calc(100vw - 32px)); margin: 20px auto 48px; }}
    .hero, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); }}
    .hero {{ padding: 30px; }}
    .eyebrow {{ display: inline-flex; padding: 8px 12px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
    h1 {{ margin: 14px 0 10px; font-size: clamp(30px, 4vw, 52px); line-height: 1.05; }}
    .hero p, .panel p {{ color: var(--muted); line-height: 1.7; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; margin-top: 20px; }}
    .card {{ padding: 18px; border-radius: 20px; background: white; border: 1px solid var(--line); }}
    .card .label {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; }}
    .card .value {{ margin-top: 10px; font-size: 28px; font-weight: 700; }}
    .panel {{ margin-top: 22px; padding: 24px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 18px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    th, td {{ padding: 13px 12px; border-bottom: 1px solid rgba(23, 32, 49, 0.08); text-align: left; vertical-align: top; }}
    th {{ background: rgba(247, 241, 233, 0.75); color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; }}
    tr:last-child td {{ border-bottom: none; }}
    .size-pill {{ display: inline-flex; padding: 8px 12px; border-radius: 999px; background: rgba(23, 32, 49, 0.08); font-weight: 700; white-space: nowrap; }}
    .footer-note {{ margin-top: 18px; color: var(--muted); font-size: 14px; }}
    code {{ background: rgba(23, 32, 49, 0.08); border-radius: 6px; padding: 2px 6px; }}
    @media (max-width: 820px) {{
      .cards {{ grid-template-columns: 1fr; }}
      .page {{ width: min(100vw - 16px, 100%); }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <span class="eyebrow">Given 4-Diagonal Data</span>
      <h1>Block LU vs Modified Thomas on the Provided 8-512 Matrices</h1>
      <p>Halaman ini membandingkan Block LU dengan solver 4-diagonal bertipe Thomas pada matriks asli yang diberikan di folder <code>ukuran 8 x 8</code> sampai <code>ukuran 512 x 512</code>. Jadi angka di bawah ini tidak lagi berasal dari dataset tridiagonal sintetis, melainkan langsung dari pita asli dengan lower bandwidth <code>p = 1</code> dan upper bandwidth <code>q = 2</code>.</p>
      <div class="cards">
        <div class="card">
          <div class="label">Fastest Method</div>
          <div class="value">Thomas 4-Diagonal</div>
          <p>Pada semua ukuran yang diuji, solver 4-diagonal tetap lebih cepat daripada Block LU.</p>
        </div>
        <div class="card">
          <div class="label">Largest Tested Size</div>
          <div class="value">{int(rows[-1]["size"])} x {int(rows[-1]["size"])}</div>
          <p>Pada ukuran ini Block LU memakai backend {html.escape(str(rows[-1]["block_lu_backend"]))} sesuai policy laporan.</p>
        </div>
        <div class="card">
          <div class="label">Best Thomas Time</div>
          <div class="value">{fmt_ms(fastest_row["thomas4_elapsed_median_seconds"])} ms</div>
          <p>Ringkasan numerik lengkap disimpan di <code>{html.escape(str(summary_path.name))}</code>.</p>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Runtime and Algorithm Memory</h2>
      <p>Kolom memory di sini adalah peak core memory algoritma, yaitu ukuran buffer array utama yang benar-benar hidup selama factorization dan solve. View yang berbagi buffer tidak dihitung ganda.</p>
      <div class="table-wrap"><table><thead><tr><th>Size</th><th>Block LU Backend</th><th>Block LU Time (ms)</th><th>Thomas 4-Diagonal Time (ms)</th><th>Block LU Algo Mem (MB)</th><th>Thomas 4-Diagonal Algo Mem (MB)</th></tr></thead><tbody>{''.join(runtime_rows)}</tbody></table></div>
    </section>

    <section class="panel">
      <h2>Accuracy Snapshot</h2>
      <p>Setiap error solusi dibandingkan terhadap <code>numpy.linalg.solve</code> pada matriks asli yang sama. Error LU memakai norma infinity relatif terhadap matriks input.</p>
      <div class="table-wrap"><table><thead><tr><th>Size</th><th>BLU LU Err</th><th>BLU x_b Err</th><th>BLU x_c Err</th><th>Thomas LU Err</th><th>Thomas x_b Err</th><th>Thomas x_c Err</th></tr></thead><tbody>{''.join(accuracy_rows)}</tbody></table></div>
    </section>

    <section class="panel">
      <h2>Resource Snapshot</h2>
      <p>CPU dan GPU usage diambil dari batch profiling yang cukup panjang agar sampler tidak didominasi overhead timing pada satu eksekusi yang terlalu singkat.</p>
      <div class="table-wrap"><table><thead><tr><th>Size</th><th>BLU CPU Avg (%)</th><th>Thomas CPU Avg (%)</th><th>BLU GPU Peak (%)</th><th>Thomas GPU Peak (%)</th><th>BLU Process Peak Mem (MB)</th><th>Thomas Process Peak Mem (MB)</th></tr></thead><tbody>{''.join(resource_rows)}</tbody></table></div>
    </section>

    <section class="panel">
      <h2>Condition Numbers</h2>
      <p>Condition number dihitung langsung dari matriks asli ukuran 8 sampai 512 dan dipakai untuk membaca apakah error yang terukur masih konsisten dengan sensitivitas sistem.</p>
      <div class="table-wrap"><table><thead><tr><th>Size</th><th>cond_1(A)</th><th>cond_2(A)</th><th>cond_inf(A)</th></tr></thead><tbody>{''.join(conditioning_rows)}</tbody></table></div>
      <p class="footer-note">Generated by <code>benchmark_banded4_given_data.py</code>. Summary CSV: <code>{html.escape(str(summary_path))}</code>.</p>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Block LU and a 4-diagonal Thomas-type solver on the provided banded matrices."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("."),
        help="Root directory containing folders such as 'ukuran 8 x 8'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("hasil_banded4_aktual"),
        help="Output directory for CSV, JSON, HTML, and solution text files.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=DEFAULT_SIZES,
        help="Subset of sizes to process. Defaults to 8 16 32 128 256 512.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    solutions_dir = output_root / "solusi"
    solutions_dir.mkdir(parents=True, exist_ok=True)

    backend_info = detect_backend()
    prefer_gpu_for_block_lu = backend_info.backend == "gpu"

    block_lu_method = MethodSpec(
        slug="block_lu",
        display_name="Block LU",
        note="Dense blocked LU used as the general-purpose banded fallback.",
        execute=lambda case: execute_block_lu(
            case,
            prefer_gpu=(case.size >= 512 and prefer_gpu_for_block_lu),
        ),
        profile_backend="gpu" if prefer_gpu_for_block_lu else "cpu",
    )
    thomas_method = MethodSpec(
        slug="thomas4",
        display_name="Thomas 4-Diagonal",
        note="Compact solver for lower bandwidth 1 and upper bandwidth 2.",
        execute=execute_thomas4,
        profile_backend="cpu",
    )

    wide_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for size, case_dir in discover_cases(args.input_root, args.sizes):
        print(f"[LOAD] {size}x{size} from {case_dir}")
        case = load_banded_case(case_dir, size)

        print(f"[RUN] Block LU @ {size}x{size}")
        block_lu_row = benchmark_method(case, block_lu_method)
        print(
            f"[DONE] Block LU @ {size}x{size} -> "
            f"{block_lu_row['elapsed_compute_median_seconds']:.6f}s"
        )

        print(f"[RUN] Thomas 4-Diagonal @ {size}x{size}")
        thomas_row = benchmark_method(case, thomas_method)
        print(
            f"[DONE] Thomas 4-Diagonal @ {size}x{size} -> "
            f"{thomas_row['elapsed_compute_median_seconds']:.6f}s"
        )

        write_vector_txt(solutions_dir / f"block_lu_x_b_{size}x{size}.txt", block_lu_row["x_b"])
        write_vector_txt(solutions_dir / f"block_lu_x_c_{size}x{size}.txt", block_lu_row["x_c"])
        write_vector_txt(solutions_dir / f"thomas4_x_b_{size}x{size}.txt", thomas_row["x_b"])
        write_vector_txt(solutions_dir / f"thomas4_x_c_{size}x{size}.txt", thomas_row["x_c"])

        wide_rows.append(build_wide_row(case, block_lu_row, thomas_row))
        detail_rows.extend(
            [
                {"size": size, **{k: v for k, v in block_lu_row.items() if k not in {"x_b", "x_c"}}},
                {"size": size, **{k: v for k, v in thomas_row.items() if k not in {"x_b", "x_c"}}},
            ]
        )

    summary_csv_path = output_root / "ringkasan.csv"
    summary_json_path = output_root / "ringkasan.json"
    detail_json_path = output_root / "ringkasan_detail.json"
    html_path = output_root / "website_perbandingan_banded4_aktual.html"

    write_summary_csv(summary_csv_path, wide_rows)
    write_json(summary_json_path, {"rows": wide_rows})
    write_json(detail_json_path, {"rows": detail_rows, "backend_info": to_builtin(backend_info.__dict__)})
    build_html(wide_rows, html_path, summary_csv_path)

    print(f"[WRITE] {summary_csv_path}")
    print(f"[WRITE] {summary_json_path}")
    print(f"[WRITE] {detail_json_path}")
    print(f"[WRITE] {html_path}")


if __name__ == "__main__":
    main()
