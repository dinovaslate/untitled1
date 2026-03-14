#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from benchmark_lu_cpu_variants import (
    factorize_doolittle,
    factorize_gaussian_elimination,
)
from block_lu_cpu import (
    block_lu_cpu,
    block_size_for_cpu,
    solve_via_lu_cpu,
)
from block_lu_gpu import (
    block_lu_gpu_device,
    block_size_for_gpu,
    solve_via_lu_gpu_device,
    warm_up_gpu_backend,
)
from generate_hasil_lu import (
    BackendInfo,
    PerformanceMonitor,
    normalized_residual_inf,
    relative_error_inf,
)
from thomas_algorithm import (
    factorize_tridiagonal,
    lu_relative_error_inf_tridiagonal,
    normalized_residual_inf_tridiagonal,
    solve_tridiagonal_factored,
)


ALL_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
PIVOT_TOL = 1e-12


@dataclass(frozen=True)
class TridiagonalCase:
    size: int
    lower: np.ndarray
    diagonal: np.ndarray
    upper: np.ndarray
    b: np.ndarray
    c: np.ndarray
    dense_matrix: np.ndarray | None


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    display_name: str
    note: str
    family: str
    requires_dense: bool
    max_size: int
    repetitions: dict[int, int]
    execute: Callable[[TridiagonalCase], dict]


def build_tridiagonal_case(size: int, *, requires_dense: bool) -> TridiagonalCase:
    lower = np.full(size - 1, -1.0, dtype=np.float64)
    diagonal = np.full(size, 4.0, dtype=np.float64)
    upper = np.full(size - 1, -1.0, dtype=np.float64)

    b = np.full(size, 2.0, dtype=np.float64)
    b[0] = 3.0
    b[-1] = 3.0

    c = 2.0 * np.arange(1, size + 1, dtype=np.float64)
    c[0] = 2.0
    c[-1] = 3.0 * size + 1.0

    dense_matrix = None
    if requires_dense:
        dense_matrix = np.zeros((size, size), dtype=np.float64)
        indices = np.arange(size)
        dense_matrix[indices, indices] = diagonal
        dense_matrix[indices[1:], indices[:-1]] = lower
        dense_matrix[indices[:-1], indices[1:]] = upper

    return TridiagonalCase(
        size=size,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        b=b,
        c=c,
        dense_matrix=dense_matrix,
    )


def x_true_b(size: int) -> np.ndarray:
    return np.ones(size, dtype=np.float64)


def x_true_c(size: int) -> np.ndarray:
    return np.arange(1, size + 1, dtype=np.float64)


def execute_thomas(case: TridiagonalCase) -> dict:
    multipliers, u_diagonal, u_upper = factorize_tridiagonal(
        case.lower,
        case.diagonal,
        case.upper,
        pivot_tol=PIVOT_TOL,
    )
    x_b = solve_tridiagonal_factored(multipliers, u_diagonal, u_upper, case.b)
    x_c = solve_tridiagonal_factored(multipliers, u_diagonal, u_upper, case.c)
    return {
        "block_size": None,
        "storage_model": "tridiagonal compact",
        "lu_relative_error_inf": lu_relative_error_inf_tridiagonal(
            case.lower,
            case.diagonal,
            case.upper,
            multipliers,
            u_diagonal,
            u_upper,
        ),
        "x_b": x_b,
        "x_c": x_c,
    }


def execute_block_lu_cpu(case: TridiagonalCase) -> dict:
    if case.dense_matrix is None:
        raise ValueError("Dense matrix is required for Block LU CPU.")

    block_size = block_size_for_cpu(case.size)
    combined, L, U = block_lu_cpu(case.dense_matrix, block_size=block_size, pivot_tol=PIVOT_TOL)
    x_b = solve_via_lu_cpu(L, U, case.b)
    x_c = solve_via_lu_cpu(L, U, case.c)
    return {
        "block_size": block_size,
        "storage_model": "dense",
        "combined_lu": combined,
        "L": L,
        "U": U,
        "x_b": x_b,
        "x_c": x_c,
    }


def execute_block_lu_gpu(case: TridiagonalCase) -> dict:
    if case.dense_matrix is None:
        raise ValueError("Dense matrix is required for Block LU GPU.")

    import cupy as cp  # type: ignore

    block_size = block_size_for_gpu(case.size)
    combined_gpu, L_gpu, U_gpu = block_lu_gpu_device(
        case.dense_matrix,
        block_size=block_size,
        pivot_tol=PIVOT_TOL,
    )
    x_b_gpu = solve_via_lu_gpu_device(L_gpu, U_gpu, case.b)
    x_c_gpu = solve_via_lu_gpu_device(L_gpu, U_gpu, case.c)
    cp.cuda.Stream.null.synchronize()
    return {
        "block_size": block_size,
        "storage_model": "dense",
        "combined_lu": cp.asnumpy(combined_gpu),
        "L": cp.asnumpy(L_gpu),
        "U": cp.asnumpy(U_gpu),
        "x_b": cp.asnumpy(x_b_gpu),
        "x_c": cp.asnumpy(x_c_gpu),
    }


def execute_doolittle(case: TridiagonalCase) -> dict:
    if case.dense_matrix is None:
        raise ValueError("Dense matrix is required for Doolittle.")

    combined, L, U = factorize_doolittle(case.dense_matrix, block_size=0, pivot_tol=PIVOT_TOL)
    x_b = solve_via_lu_cpu(L, U, case.b)
    x_c = solve_via_lu_cpu(L, U, case.c)
    return {
        "block_size": None,
        "storage_model": "dense",
        "combined_lu": combined,
        "L": L,
        "U": U,
        "x_b": x_b,
        "x_c": x_c,
    }


def execute_gaussian(case: TridiagonalCase) -> dict:
    if case.dense_matrix is None:
        raise ValueError("Dense matrix is required for Gaussian elimination LU.")

    combined, L, U = factorize_gaussian_elimination(
        case.dense_matrix,
        block_size=0,
        pivot_tol=PIVOT_TOL,
    )
    x_b = solve_via_lu_cpu(L, U, case.b)
    x_c = solve_via_lu_cpu(L, U, case.c)
    return {
        "block_size": None,
        "storage_model": "dense",
        "combined_lu": combined,
        "L": L,
        "U": U,
        "x_b": x_b,
        "x_c": x_c,
    }


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        slug="thomas",
        display_name="Thomas Algorithm",
        note="Structure-aware tridiagonal LU on compact diagonals.",
        family="specialized",
        requires_dense=False,
        max_size=65536,
        repetitions={1024: 32, 2048: 24, 4096: 16, 8192: 12, 16384: 8, 32768: 6, 65536: 4},
        execute=execute_thomas,
    ),
    MethodSpec(
        slug="block_lu_cpu",
        display_name="Block LU CPU",
        note="Blocked no-pivot LU on CPU with backend-aware block size.",
        family="block",
        requires_dense=True,
        max_size=8192,
        repetitions={1024: 3, 2048: 2, 4096: 1, 8192: 1},
        execute=execute_block_lu_cpu,
    ),
    MethodSpec(
        slug="block_lu_gpu",
        display_name="Block LU GPU",
        note="Blocked no-pivot LU on GPU with on-device triangular solves.",
        family="block",
        requires_dense=True,
        max_size=8192,
        repetitions={1024: 3, 2048: 2, 4096: 2, 8192: 1},
        execute=execute_block_lu_gpu,
    ),
    MethodSpec(
        slug="doolittle",
        display_name="LU Doolittle",
        note="Classic full-matrix Doolittle LU without pivoting.",
        family="classic",
        requires_dense=True,
        max_size=2048,
        repetitions={1024: 2, 2048: 1},
        execute=execute_doolittle,
    ),
    MethodSpec(
        slug="gaussian_elimination",
        display_name="LU Gaussian Elimination",
        note="Classic full-matrix in-place Gaussian elimination LU without pivoting.",
        family="classic",
        requires_dense=True,
        max_size=4096,
        repetitions={1024: 2, 2048: 1, 4096: 1},
        execute=execute_gaussian,
    ),
)


def method_colors(slug: str) -> tuple[str, str]:
    return {
        "thomas": ("#2563eb", "rgba(59, 130, 246, 0.18)"),
        "block_lu_cpu": ("#b45309", "rgba(245, 158, 11, 0.18)"),
        "block_lu_gpu": ("#0f766e", "rgba(45, 212, 191, 0.18)"),
        "doolittle": ("#7c3aed", "rgba(168, 85, 247, 0.18)"),
        "gaussian_elimination": ("#166534", "rgba(34, 197, 94, 0.18)"),
    }[slug]


def skip_reason(method: MethodSpec, size: int) -> str:
    if size > method.max_size:
        if method.family == "classic":
            return "Skipped: dense classical LU becomes too slow past this size."
        if method.family == "block":
            return "Skipped: dense block path above this size would exceed practical RAM or benchmark time."
    return "Skipped by policy."


def benchmark_method(method: MethodSpec, size: int) -> dict:
    case = build_tridiagonal_case(size, requires_dense=method.requires_dense)
    repetitions = method.repetitions[size]

    if method.slug == "block_lu_gpu":
        warm_up_gpu_backend()

    method.execute(case)

    timings: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = method.execute(case)
        timings.append(time.perf_counter() - start)
        del result

    monitor_backend = "gpu" if method.slug == "block_lu_gpu" else "cpu"
    monitor = PerformanceMonitor(BackendInfo(backend=monitor_backend, note=method.note))
    monitor.start()
    profile_start = time.perf_counter()
    profile = method.execute(case)
    profile_elapsed = time.perf_counter() - profile_start
    metrics = monitor.stop()

    x_b_ref = x_true_b(size)
    x_c_ref = x_true_c(size)
    x_b = profile["x_b"]
    x_c = profile["x_c"]

    if "lu_relative_error_inf" in profile:
        lu_error = float(profile["lu_relative_error_inf"])
    else:
        if case.dense_matrix is None:
            raise ValueError("Dense LU error requested without dense matrix.")
        lu_error = relative_error_inf(profile["L"] @ profile["U"], case.dense_matrix)

    if case.dense_matrix is not None:
        x_b_residual = normalized_residual_inf(case.dense_matrix, x_b, case.b)
        x_c_residual = normalized_residual_inf(case.dense_matrix, x_c, case.c)
    else:
        x_b_residual = normalized_residual_inf_tridiagonal(case.lower, case.diagonal, case.upper, x_b, case.b)
        x_c_residual = normalized_residual_inf_tridiagonal(case.lower, case.diagonal, case.upper, x_c, case.c)

    return {
        "status": "measured",
        "size": size,
        "method": method.slug,
        "method_display_name": method.display_name,
        "family": method.family,
        "method_note": method.note,
        "repetitions": repetitions,
        "block_size": profile.get("block_size"),
        "storage_model": profile.get("storage_model", "dense" if method.requires_dense else "tridiagonal compact"),
        "elapsed_compute_median_seconds": float(statistics.median(timings)),
        "elapsed_compute_mean_seconds": float(statistics.mean(timings)),
        "elapsed_profile_run_seconds": profile_elapsed,
        "lu_relative_error_inf": lu_error,
        "x_b_relative_error_inf": relative_error_inf(x_b, x_b_ref),
        "x_c_relative_error_inf": relative_error_inf(x_c, x_c_ref),
        "x_b_normalized_residual_inf": x_b_residual,
        "x_c_normalized_residual_inf": x_c_residual,
        **metrics,
    }


def benchmark_or_skip(method: MethodSpec, size: int) -> dict:
    if size not in method.repetitions or size > method.max_size:
        return {
            "status": "skipped",
            "size": size,
            "method": method.slug,
            "method_display_name": method.display_name,
            "family": method.family,
            "method_note": method.note,
            "storage_model": "dense" if method.requires_dense else "tridiagonal compact",
            "skip_reason": skip_reason(method, size),
        }
    return benchmark_method(method, size)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "status",
        "size",
        "method",
        "method_display_name",
        "family",
        "storage_model",
        "repetitions",
        "block_size",
        "elapsed_compute_median_seconds",
        "elapsed_compute_mean_seconds",
        "elapsed_profile_run_seconds",
        "lu_relative_error_inf",
        "x_b_relative_error_inf",
        "x_c_relative_error_inf",
        "x_b_normalized_residual_inf",
        "x_c_normalized_residual_inf",
        "cpu_time_seconds",
        "cpu_usage_avg_percent_total_capacity",
        "cpu_usage_peak_percent_total_capacity",
        "memory_peak_mb",
        "gpu_usage_peak_percent",
        "gpu_memory_peak_mb",
        "cache_hit_percent_proxy_avg",
        "cache_miss_percent_proxy_avg",
        "skip_reason",
        "method_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "Skipped"
    if value >= 60:
        minutes = int(value // 60)
        seconds = value - minutes * 60
        return f"{minutes}m {seconds:.1f}s"
    return f"{value:.3f}s"


def fmt_sci(value: float | None) -> str:
    if value is None:
        return "Skipped"
    return f"{value:.3e}"


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def render_runtime_cell(row: dict | None) -> str:
    if row is None or row["status"] != "measured":
        reason = row.get("skip_reason", "No data.") if row else "No data."
        return f'<td><div class="skip-card"><strong>Skipped</strong><span>{html.escape(reason)}</span></div></td>'

    color, soft = method_colors(row["method"])
    detail_lines = [f"{fmt_percent(row.get('cpu_usage_avg_percent_total_capacity'))} avg CPU"]
    if row.get("block_size") is not None:
        detail_lines.append(f"block size {row['block_size']}")
    else:
        detail_lines.append(str(row.get("storage_model", "")))
    if row["method"] == "block_lu_gpu":
        detail_lines.append(f"{fmt_percent(row.get('gpu_usage_peak_percent'))} peak GPU")

    detail_html = "".join(f"<span>{html.escape(line)}</span>" for line in detail_lines)
    return (
        "<td>"
        f'<div class="metric-card" style="--metric-ink:{color}; --metric-soft:{soft};">'
        f"<strong>{fmt_seconds(row['elapsed_compute_median_seconds'])}</strong>"
        f"{detail_html}"
        "</div>"
        "</td>"
    )


def render_accuracy_cell(row: dict | None) -> str:
    if row is None or row["status"] != "measured":
        return '<td><div class="skip-card compact"><strong>Skipped</strong></div></td>'
    return (
        "<td>"
        '<div class="metric-card compact">'
        f"<strong>{fmt_sci(row['lu_relative_error_inf'])}</strong>"
        f"<span>x_b {fmt_sci(row['x_b_relative_error_inf'])}</span>"
        f"<span>x_c {fmt_sci(row['x_c_relative_error_inf'])}</span>"
        "</div>"
        "</td>"
    )


def build_html(rows: list[dict], output_path: Path, summary_path: Path) -> None:
    by_size: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_size.setdefault(int(row["size"]), {})[str(row["method"])] = row

    measured_rows = [row for row in rows if row["status"] == "measured"]
    dense_rows = [row for row in measured_rows if row["storage_model"] == "dense"]
    fastest_overall = min(measured_rows, key=lambda item: item["elapsed_compute_median_seconds"])
    fastest_dense = min(dense_rows, key=lambda item: item["elapsed_compute_median_seconds"])

    thomas_advantage_8192 = None
    if 8192 in by_size and by_size[8192].get("thomas", {}).get("status") == "measured":
        dense_8192 = [
            row for row in by_size[8192].values()
            if row.get("status") == "measured" and row.get("storage_model") == "dense"
        ]
        if dense_8192:
            best_dense_8192 = min(dense_8192, key=lambda row: row["elapsed_compute_median_seconds"])
            thomas_advantage_8192 = (
                best_dense_8192["elapsed_compute_median_seconds"]
                / by_size[8192]["thomas"]["elapsed_compute_median_seconds"]
            )

    method_order = [method.slug for method in METHODS]
    method_names = {method.slug: method.display_name for method in METHODS}

    runtime_rows_html: list[str] = []
    accuracy_rows_html: list[str] = []
    coverage_rows_html: list[str] = []

    for size in ALL_SIZES:
        group = by_size.get(size, {})
        runtime_rows_html.append(
            "<tr>"
            f'<td><span class="size-pill">{size} x {size}</span></td>'
            + "".join(render_runtime_cell(group.get(slug)) for slug in method_order)
            + "</tr>"
        )
        accuracy_rows_html.append(
            "<tr>"
            f'<td><span class="size-pill">{size} x {size}</span></td>'
            + "".join(render_accuracy_cell(group.get(slug)) for slug in method_order)
            + "</tr>"
        )

        notes = []
        for slug in method_order:
            row = group.get(slug)
            if row is None:
                notes.append(f"{method_names[slug]}: no data.")
            elif row["status"] == "skipped":
                notes.append(f"{method_names[slug]}: {row['skip_reason']}")
            else:
                notes.append(
                    f"{method_names[slug]}: measured, median {fmt_seconds(row['elapsed_compute_median_seconds'])}."
                )
        coverage_rows_html.append(
            "<tr>"
            f'<td><span class="size-pill">{size} x {size}</span></td>'
            f"<td>{'<br>'.join(html.escape(note) for note in notes)}</td>"
            "</tr>"
        )

    legend = "".join(
        f'<span class="legend-chip" style="--legend-ink:{method_colors(method.slug)[0]}; --legend-soft:{method_colors(method.slug)[1]};">{html.escape(method.display_name)}</span>'
        for method in METHODS
    )

    summary_cards = [
        (
            "Overall Fastest",
            fastest_overall["method_display_name"],
            f"{fastest_overall['size']} x {fastest_overall['size']} in {fmt_seconds(fastest_overall['elapsed_compute_median_seconds'])}.",
        ),
        (
            "Best Dense Method",
            fastest_dense["method_display_name"],
            f"{fastest_dense['size']} x {fastest_dense['size']} in {fmt_seconds(fastest_dense['elapsed_compute_median_seconds'])}.",
        ),
        (
            "8192 Thomas Advantage",
            f"{thomas_advantage_8192:.2f}x faster" if thomas_advantage_8192 is not None else "Not available",
            "Perbandingan terhadap metode dense tercepat yang masih bisa dibench pada ukuran 8192.",
        ),
        (
            "Coverage",
            "Thomas measured to 65536",
            "Dense methods tetap dibatasi karena mereka mematerialisasi matriks penuh, bukan storage pita tridiagonal.",
        ),
    ]

    cards_html = "".join(
        "<article class=\"summary-card\">"
        f"<div class=\"summary-title\">{html.escape(title)}</div>"
        f"<div class=\"summary-value\">{html.escape(value)}</div>"
        f"<div class=\"summary-subtext\">{html.escape(subtext)}</div>"
        "</article>"
        for title, value, subtext in summary_cards
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tridiagonal Solver Comparison</title>
  <style>
    :root {{
      --bg: #f6efe6;
      --panel: rgba(255, 252, 247, 0.95);
      --ink: #172031;
      --muted: #5b6474;
      --line: rgba(23, 32, 49, 0.12);
      --shadow: 0 20px 50px rgba(23, 32, 49, 0.14);
      --radius: 24px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 24%),
        radial-gradient(circle at top right, rgba(45, 212, 191, 0.18), transparent 25%),
        linear-gradient(180deg, #fffdf8 0%, var(--bg) 100%);
    }}
    .page {{
      width: min(1520px, calc(100vw - 28px));
      margin: 20px auto 40px;
    }}
    .hero {{
      padding: 34px;
      border-radius: 32px;
      color: white;
      background:
        linear-gradient(145deg, rgba(23, 32, 49, 0.96), rgba(37, 99, 235, 0.90)),
        linear-gradient(120deg, #172031, #2563eb);
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      display: inline-flex;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.12);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    h1 {{
      margin: 18px 0 10px;
      font-size: clamp(34px, 5vw, 60px);
      line-height: 0.95;
      max-width: 980px;
    }}
    .hero p {{
      margin: 0;
      max-width: 920px;
      color: rgba(255, 255, 255, 0.84);
      line-height: 1.7;
      font-size: 17px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .legend-chip {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--legend-soft);
      color: var(--legend-ink);
      font-weight: 700;
      font-size: 13px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 18px;
      margin-top: 22px;
    }}
    .summary-card {{
      padding: 20px;
      border-radius: 22px;
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.35);
      box-shadow: var(--shadow);
    }}
    .summary-title {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
    }}
    .summary-value {{
      margin-top: 10px;
      font-size: 28px;
      font-weight: 700;
    }}
    .summary-subtext {{
      margin-top: 8px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 14px;
    }}
    .panel {{
      margin-top: 24px;
      padding: 24px;
      border-radius: var(--radius);
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .panel h2 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    .panel p {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.68;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1420px;
    }}
    th, td {{
      padding: 14px 12px;
      border-bottom: 1px solid rgba(23, 32, 49, 0.08);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      background: rgba(246, 239, 230, 0.7);
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .size-pill {{
      display: inline-flex;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(23, 32, 49, 0.08);
      font-weight: 700;
      white-space: nowrap;
    }}
    .metric-card {{
      display: grid;
      gap: 6px;
      padding: 12px 14px;
      border-radius: 16px;
      background: var(--metric-soft, rgba(23, 32, 49, 0.05));
      border: 1px solid rgba(23, 32, 49, 0.08);
    }}
    .metric-card strong {{
      font-size: 18px;
      color: var(--metric-ink, var(--ink));
    }}
    .metric-card span {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .metric-card.compact strong {{
      font-size: 15px;
    }}
    .skip-card {{
      display: grid;
      gap: 6px;
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(148, 163, 184, 0.12);
      border: 1px dashed rgba(100, 116, 139, 0.4);
    }}
    .skip-card strong {{
      font-size: 14px;
      color: #475569;
    }}
    .skip-card span {{
      color: #64748b;
      font-size: 12px;
      line-height: 1.5;
    }}
    .skip-card.compact {{
      min-height: 74px;
      place-content: center;
    }}
    .footnote {{
      margin-top: 18px;
      color: var(--muted);
      line-height: 1.68;
      font-size: 14px;
    }}
    @media (max-width: 1160px) {{
      .summary-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 720px) {{
      .page {{
        width: min(100vw - 16px, 100%);
        margin: 8px auto 20px;
      }}
      .hero, .panel {{
        padding: 18px;
      }}
      .summary-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <span class="eyebrow">Tridiagonal Dataset Comparison</span>
      <h1>Thomas Algorithm vs Block LU and Dense LU Variants</h1>
      <p>Benchmark ini memakai sistem tridiagonal yang identik dengan folder <code>Ukuran N x N</code>: diagonal utama 4, diagonal atas dan bawah -1. Thomas algorithm diukur sebagai solver struktur-khusus dengan storage kompak, sementara Block LU, Doolittle, dan Gaussian elimination tetap memakai representasi dense. Jadi perbandingan ini sengaja menunjukkan dua hal sekaligus: baseline terbaik untuk matriks tridiagonal, dan seberapa mahal jika struktur itu diabaikan.</p>
      <div class="legend">{legend}</div>
    </section>

    <section class="summary-grid">{cards_html}</section>

    <section class="panel">
      <h2>Runtime Comparison</h2>
      <p>Angka utama memakai median runtime untuk ukuran yang diulang lebih dari sekali. Thomas algorithm tetap diukur sampai 65536 karena kompleksitasnya linear. Metode dense ditandai <em>Skipped</em> saat biaya komputasi dan memori sudah tidak proporsional.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Ukuran</th>
              <th>Thomas Algorithm</th>
              <th>Block LU CPU</th>
              <th>Block LU GPU</th>
              <th>LU Doolittle</th>
              <th>LU Gaussian Elimination</th>
            </tr>
          </thead>
          <tbody>
            {''.join(runtime_rows_html)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Accuracy Snapshot</h2>
      <p>Setiap sel menampilkan <code>||LU - A||inf / ||A||inf</code> pada baris pertama, lalu relative error solusi <code>x_b</code> dan <code>x_c</code> terhadap solusi exact. Untuk Thomas, faktor LU dihitung dalam bentuk pita tridiagonal, bukan matriks penuh.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Ukuran</th>
              <th>Thomas Algorithm</th>
              <th>Block LU CPU</th>
              <th>Block LU GPU</th>
              <th>LU Doolittle</th>
              <th>LU Gaussian Elimination</th>
            </tr>
          </thead>
          <tbody>
            {''.join(accuracy_rows_html)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Coverage and Interpretation</h2>
      <p>Thomas algorithm memang tidak apples-to-apples dengan LU dense umum, karena dia memanfaatkan struktur tridiagonal secara langsung. Justru itu poin pembandingnya: kalau matriks Anda benar-benar tridiagonal, Thomas adalah baseline yang seharusnya dikalahkan oleh metode umum hanya jika ada alasan khusus, misalnya kebutuhan GPU atau generalitas struktur.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Ukuran</th>
              <th>Status per metode</th>
            </tr>
          </thead>
          <tbody>
            {''.join(coverage_rows_html)}
          </tbody>
        </table>
      </div>
      <div class="footnote">
        Summary CSV: <code>{html.escape(str(summary_path))}</code><br>
        Policy: benchmark hanya menghitung factorization + solve <code>x_b</code> + solve <code>x_c</code>. I/O CSV tidak ikut dihitung. Dense methods tetap membangun matriks penuh di memori, sedangkan Thomas bekerja dengan tiga diagonal saja.
      </div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    output_dir = Path("hasil_tridiagonal_html_dengan_thomas")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for size in ALL_SIZES:
        for method in METHODS:
            print(f"[RUN] {method.display_name} @ {size}x{size}")
            row = benchmark_or_skip(method, size)
            rows.append(row)
            if row["status"] == "measured":
                print(
                    f"[DONE] {method.display_name} @ {size}x{size} -> "
                    f"{row['elapsed_compute_median_seconds']:.6f}s"
                )
            else:
                print(f"[SKIP] {method.display_name} @ {size}x{size} -> {row['skip_reason']}")

    summary_path = output_dir / "ringkasan.csv"
    json_path = output_dir / "ringkasan.json"
    html_path = output_dir / "website_perbandingan_tridiagonal_dengan_thomas.html"
    write_summary_csv(summary_path, rows)
    write_json(json_path, {"rows": rows})
    build_html(rows, html_path, summary_path)
    print(f"[WRITE] {summary_path}")
    print(f"[WRITE] {json_path}")
    print(f"[WRITE] {html_path}")


if __name__ == "__main__":
    main()
