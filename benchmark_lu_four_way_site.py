#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
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
from generate_hasil_lu import (
    BackendInfo,
    PerformanceMonitor,
    block_lu_cpu,
    block_lu_gpu_device,
    block_size_for,
    load_case,
    normalized_residual_inf,
    relative_error_inf,
    solve_via_lu,
    solve_via_lu_gpu_device,
    warm_up_gpu_backend,
)


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    display_name: str
    family: str
    note: str
    execute: Callable[[np.ndarray, np.ndarray, np.ndarray, int, float], dict]


def choose_repetitions(size: int) -> int:
    if size <= 16:
        return 24
    if size <= 32:
        return 16
    if size <= 128:
        return 8
    if size <= 256:
        return 5
    return 3


def execute_block_lu_cpu(A: np.ndarray, b: np.ndarray, c: np.ndarray, size: int, pivot_tol: float) -> dict:
    block_size = block_size_for(size, "cpu")
    combined, L, U = block_lu_cpu(A, block_size=block_size, pivot_tol=pivot_tol)
    x_b = solve_via_lu(L, U, b)
    x_c = solve_via_lu(L, U, c)
    return {
        "block_size": block_size,
        "combined_lu": combined,
        "L": L,
        "U": U,
        "x_b": x_b,
        "x_c": x_c,
    }


def execute_block_lu_gpu(A: np.ndarray, b: np.ndarray, c: np.ndarray, size: int, pivot_tol: float) -> dict:
    import cupy as cp  # type: ignore

    block_size = block_size_for(size, "gpu")
    combined_gpu, L_gpu, U_gpu = block_lu_gpu_device(A, block_size=block_size, pivot_tol=pivot_tol)
    x_b_gpu = solve_via_lu_gpu_device(L_gpu, U_gpu, b)
    x_c_gpu = solve_via_lu_gpu_device(L_gpu, U_gpu, c)
    cp.cuda.Stream.null.synchronize()
    return {
        "block_size": block_size,
        "combined_lu": cp.asnumpy(combined_gpu),
        "L": cp.asnumpy(L_gpu),
        "U": cp.asnumpy(U_gpu),
        "x_b": cp.asnumpy(x_b_gpu),
        "x_c": cp.asnumpy(x_c_gpu),
    }


def execute_doolittle(A: np.ndarray, b: np.ndarray, c: np.ndarray, size: int, pivot_tol: float) -> dict:
    del size
    combined, L, U = factorize_doolittle(A, block_size=0, pivot_tol=pivot_tol)
    x_b = solve_via_lu(L, U, b)
    x_c = solve_via_lu(L, U, c)
    return {
        "block_size": None,
        "combined_lu": combined,
        "L": L,
        "U": U,
        "x_b": x_b,
        "x_c": x_c,
    }


def execute_gaussian(A: np.ndarray, b: np.ndarray, c: np.ndarray, size: int, pivot_tol: float) -> dict:
    del size
    combined, L, U = factorize_gaussian_elimination(A, block_size=0, pivot_tol=pivot_tol)
    x_b = solve_via_lu(L, U, b)
    x_c = solve_via_lu(L, U, c)
    return {
        "block_size": None,
        "combined_lu": combined,
        "L": L,
        "U": U,
        "x_b": x_b,
        "x_c": x_c,
    }


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        slug="block_lu_cpu",
        display_name="Block LU CPU",
        family="block",
        note="Optimized blocked LU on CPU with backend-aware block size.",
        execute=execute_block_lu_cpu,
    ),
    MethodSpec(
        slug="block_lu_gpu",
        display_name="Block LU GPU",
        family="block",
        note="Optimized blocked LU on GPU with on-device forward/back substitution.",
        execute=execute_block_lu_gpu,
    ),
    MethodSpec(
        slug="doolittle",
        display_name="LU Doolittle",
        family="classic",
        note="Classic Doolittle LU without pivoting.",
        execute=execute_doolittle,
    ),
    MethodSpec(
        slug="gaussian_elimination",
        display_name="LU Gaussian Elimination",
        family="classic",
        note="In-place Gaussian elimination LU without pivoting.",
        execute=execute_gaussian,
    ),
)


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.floating):
        result = float(value)
        return None if not math.isfinite(result) else result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def benchmark_method(
    method: MethodSpec,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    size: int,
    pivot_tol: float,
) -> dict:
    repetitions = choose_repetitions(size)

    if method.slug == "block_lu_gpu":
        warm_up_gpu_backend()

    # Warm up the method itself so one-time setup does not dominate timing.
    method.execute(A, b, c, size, pivot_tol)

    timings: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = method.execute(A, b, c, size, pivot_tol)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        del result

    monitor_backend = BackendInfo(
        backend="gpu" if method.slug == "block_lu_gpu" else "cpu",
        note=f"Profile run for {method.display_name}.",
    )
    monitor = PerformanceMonitor(monitor_backend)
    monitor.start()
    profile_start = time.perf_counter()
    profile_result = method.execute(A, b, c, size, pivot_tol)
    profile_elapsed = time.perf_counter() - profile_start
    runtime_metrics = monitor.stop()

    x_b_reference = np.linalg.solve(A, b)
    x_c_reference = np.linalg.solve(A, c)

    L = profile_result["L"]
    U = profile_result["U"]
    x_b = profile_result["x_b"]
    x_c = profile_result["x_c"]
    combined = profile_result["combined_lu"]
    block_size = profile_result["block_size"]

    return {
        "size": size,
        "size_label": f"ukuran_{size}x{size}",
        "method": method.slug,
        "method_display_name": method.display_name,
        "family": method.family,
        "method_note": method.note,
        "block_size": block_size,
        "repetitions": repetitions,
        "elapsed_compute_median_seconds": float(statistics.median(timings)),
        "elapsed_compute_mean_seconds": float(statistics.mean(timings)),
        "elapsed_compute_min_seconds": float(min(timings)),
        "elapsed_compute_max_seconds": float(max(timings)),
        "elapsed_profile_run_seconds": profile_elapsed,
        "lu_relative_error_inf": relative_error_inf(L @ U, A),
        "x_b_relative_error_inf_vs_numpy": relative_error_inf(x_b, x_b_reference),
        "x_c_relative_error_inf_vs_numpy": relative_error_inf(x_c, x_c_reference),
        "x_b_normalized_residual_inf": normalized_residual_inf(A, x_b, b),
        "x_c_normalized_residual_inf": normalized_residual_inf(A, x_c, c),
        "combined_lu_shape": list(combined.shape),
        **runtime_metrics,
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "size",
        "size_label",
        "method",
        "method_display_name",
        "family",
        "block_size",
        "repetitions",
        "elapsed_compute_median_seconds",
        "elapsed_compute_mean_seconds",
        "elapsed_compute_min_seconds",
        "elapsed_compute_max_seconds",
        "elapsed_profile_run_seconds",
        "lu_relative_error_inf",
        "x_b_relative_error_inf_vs_numpy",
        "x_c_relative_error_inf_vs_numpy",
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
        "method_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: to_builtin(row.get(field)) for field in fieldnames})


def fmt_seconds(value: float) -> str:
    return f"{value:.6f} s"


def fmt_sci(value: float) -> str:
    return f"{value:.3e}"


def fmt_ratio(value: float) -> str:
    return f"{value:.2f}x"


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def method_color(slug: str) -> tuple[str, str]:
    return {
        "block_lu_cpu": ("#9a3412", "rgba(249, 115, 22, 0.16)"),
        "block_lu_gpu": ("#0369a1", "rgba(14, 165, 233, 0.16)"),
        "doolittle": ("#7c2d12", "rgba(245, 158, 11, 0.16)"),
        "gaussian_elimination": ("#166534", "rgba(34, 197, 94, 0.16)"),
    }[slug]


def build_html(rows: list[dict], output_path: Path, summary_path: Path) -> None:
    by_size: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_size.setdefault(int(row["size"]), {})[str(row["method"])] = row

    method_order = [method.slug for method in METHODS]
    method_names = {method.slug: method.display_name for method in METHODS}

    win_counts = {method.slug: 0 for method in METHODS}
    geometric_products = {method.slug: 1.0 for method in METHODS}
    runtime_row_html: list[str] = []
    accuracy_row_html: list[str] = []
    resource_row_html: list[str] = []

    for size in sorted(by_size):
        group = by_size[size]
        fastest_slug = min(method_order, key=lambda slug: group[slug]["elapsed_compute_median_seconds"])
        win_counts[fastest_slug] += 1

        max_time = max(group[slug]["elapsed_compute_median_seconds"] for slug in method_order)
        reference_block_cpu = group["block_lu_cpu"]["elapsed_compute_median_seconds"]
        for slug in method_order:
            geometric_products[slug] *= group[slug]["elapsed_compute_median_seconds"]

        def runtime_cell(slug: str) -> str:
            row = group[slug]
            color, soft = method_color(slug)
            width = 0.0 if max_time == 0 else (row["elapsed_compute_median_seconds"] / max_time) * 100.0
            marker = "best-badge" if slug == fastest_slug else "method-badge"
            return f"""
            <td>
              <div class="metric-cell">
                <span class="{marker}" style="--badge-ink:{color}; --badge-soft:{soft};">{html.escape(method_names[slug])}</span>
                <strong>{fmt_seconds(row["elapsed_compute_median_seconds"])}</strong>
                <div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%; --fill:{color};"></div></div>
              </div>
            </td>
            """

        runtime_row_html.append(
            f"""
            <tr>
              <td><span class="size-pill">{size} x {size}</span></td>
              {runtime_cell("block_lu_cpu")}
              {runtime_cell("block_lu_gpu")}
              {runtime_cell("doolittle")}
              {runtime_cell("gaussian_elimination")}
              <td>{fmt_ratio(group["block_lu_gpu"]["elapsed_compute_median_seconds"] / reference_block_cpu)}</td>
              <td>{fmt_ratio(group["doolittle"]["elapsed_compute_median_seconds"] / reference_block_cpu)}</td>
              <td>{fmt_ratio(group["gaussian_elimination"]["elapsed_compute_median_seconds"] / reference_block_cpu)}</td>
            </tr>
            """
        )

        accuracy_row_html.append(
            f"""
            <tr>
              <td><span class="size-pill">{size} x {size}</span></td>
              <td>{fmt_sci(group["block_lu_cpu"]["lu_relative_error_inf"])}</td>
              <td>{fmt_sci(group["block_lu_gpu"]["lu_relative_error_inf"])}</td>
              <td>{fmt_sci(group["doolittle"]["lu_relative_error_inf"])}</td>
              <td>{fmt_sci(group["gaussian_elimination"]["lu_relative_error_inf"])}</td>
              <td>{fmt_sci(group["block_lu_cpu"]["x_b_relative_error_inf_vs_numpy"])}</td>
              <td>{fmt_sci(group["block_lu_gpu"]["x_b_relative_error_inf_vs_numpy"])}</td>
              <td>{fmt_sci(group["doolittle"]["x_b_relative_error_inf_vs_numpy"])}</td>
              <td>{fmt_sci(group["gaussian_elimination"]["x_b_relative_error_inf_vs_numpy"])}</td>
            </tr>
            """
        )

        resource_row_html.append(
            f"""
            <tr>
              <td><span class="size-pill">{size} x {size}</span></td>
              <td>{group["block_lu_cpu"]["block_size"]}</td>
              <td>{group["block_lu_gpu"]["block_size"]}</td>
              <td>{fmt_percent(group["block_lu_gpu"]["gpu_usage_peak_percent"])}</td>
              <td>{group["block_lu_cpu"]["memory_peak_mb"]:.2f} MB</td>
              <td>{group["block_lu_gpu"]["memory_peak_mb"]:.2f} MB</td>
              <td>{group["doolittle"]["memory_peak_mb"]:.2f} MB</td>
              <td>{group["gaussian_elimination"]["memory_peak_mb"]:.2f} MB</td>
            </tr>
            """
        )

    method_count = len(method_order)
    size_count = len(by_size)
    geo_mean_rows = []
    for slug in method_order:
        geo = geometric_products[slug] ** (1.0 / size_count)
        geo_mean_rows.append((slug, geo))
    overall_best_slug, overall_best_geo = min(geo_mean_rows, key=lambda item: item[1])
    most_wins_slug, most_wins_count = max(win_counts.items(), key=lambda item: item[1])

    summary_cards = []
    for title, value, subtext in [
        (
            "Most Wins",
            method_names[most_wins_slug],
            f"Menang di {most_wins_count} dari {size_count} ukuran.",
        ),
        (
            "Best Geometric Mean",
            method_names[overall_best_slug],
            f"Geometric mean runtime {fmt_seconds(overall_best_geo)}.",
        ),
        (
            "Policy",
            "Unified End-to-End Median",
            "Timing membandingkan factorization + solve x_b + solve x_c, tanpa file I/O ke disk.",
        ),
        (
            "GPU Note",
            "No-Pivot GPU Path",
            "GPU memakai jalur no-pivot yang sama, dan hasil tetap dimaterialisasi ke host agar validasi numerik konsisten.",
        ),
    ]:
        summary_cards.append(
            f"""
            <article class="summary-card">
              <div class="summary-title">{html.escape(title)}</div>
              <div class="summary-value">{html.escape(value)}</div>
              <div class="summary-subtext">{html.escape(subtext)}</div>
            </article>
            """
        )

    legend_items = []
    for slug in method_order:
        color, soft = method_color(slug)
        legend_items.append(
            f'<span class="legend-chip" style="--legend-ink:{color}; --legend-soft:{soft};">{html.escape(method_names[slug])}</span>'
        )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LU Method Arena</title>
  <style>
    :root {{
      --bg: #f4efe5;
      --ink: #152033;
      --muted: #5f6675;
      --panel: rgba(255, 253, 248, 0.94);
      --panel-strong: rgba(255, 251, 244, 0.98);
      --line: rgba(21, 32, 51, 0.12);
      --shadow: 0 24px 62px rgba(21, 32, 51, 0.14);
      --radius: 26px;
      --accent: #7c3aed;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(249, 115, 22, 0.18), transparent 26%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 26%),
        linear-gradient(180deg, #fffdf9 0%, var(--bg) 100%);
    }}

    .page {{
      width: min(1320px, calc(100vw - 28px));
      margin: 20px auto 44px;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      padding: 34px;
      border-radius: 34px;
      background:
        linear-gradient(145deg, rgba(21, 32, 51, 0.94), rgba(124, 58, 237, 0.88)),
        linear-gradient(120deg, #152033, #7c3aed);
      color: white;
      box-shadow: var(--shadow);
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px);
      background-size: 26px 26px;
      mask-image: linear-gradient(180deg, rgba(0,0,0,0.8), transparent);
      pointer-events: none;
    }}

    .eyebrow {{
      position: relative;
      z-index: 1;
      display: inline-flex;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.12);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .hero h1 {{
      position: relative;
      z-index: 1;
      margin: 18px 0 12px;
      font-size: clamp(34px, 5vw, 62px);
      line-height: 0.95;
      max-width: 920px;
    }}

    .hero p {{
      position: relative;
      z-index: 1;
      max-width: 820px;
      margin: 0;
      color: rgba(255,255,255,0.84);
      line-height: 1.72;
      font-size: 17px;
    }}

    .legend {{
      position: relative;
      z-index: 1;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 20px;
    }}

    .legend-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 12px;
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
      border: 1px solid rgba(255,255,255,0.38);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
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
      line-height: 1.62;
      font-size: 14px;
    }}

    .panel {{
      margin-top: 24px;
      padding: 24px;
      border-radius: var(--radius);
      background: var(--panel-strong);
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
      border-radius: 20px;
      border: 1px solid var(--line);
      background: white;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1160px;
    }}

    th, td {{
      padding: 14px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}

    thead th {{
      background: #fafbfc;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    tbody tr:nth-child(even) {{
      background: #fcfdfd;
    }}

    .size-pill {{
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      font-weight: 700;
      font-size: 13px;
    }}

    .metric-cell {{
      display: grid;
      gap: 7px;
      min-width: 165px;
    }}

    .method-badge, .best-badge {{
      display: inline-flex;
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      color: var(--badge-ink);
      background: var(--badge-soft);
    }}

    .best-badge {{
      box-shadow: inset 0 0 0 1px rgba(21, 32, 51, 0.08);
    }}

    .bar-track {{
      height: 8px;
      border-radius: 999px;
      background: #e5e7eb;
      overflow: hidden;
    }}

    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: var(--fill);
    }}

    .footnote {{
      margin-top: 18px;
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(124, 58, 237, 0.08);
      color: #5b21b6;
      line-height: 1.72;
    }}

    .source-list {{
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }}

    code {{
      padding: 2px 6px;
      border-radius: 8px;
      background: rgba(21,32,51,0.06);
      font-family: Consolas, monospace;
      font-size: 0.92em;
    }}

    @media (max-width: 980px) {{
      .page {{
        width: min(100vw - 16px, 100%);
        margin: 8px auto 24px;
      }}

      .hero, .panel {{
        padding: 18px;
        border-radius: 22px;
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
      <div class="eyebrow">Benchmark Website • Four-Way LU Comparison</div>
      <h1>Block LU CPU vs Block LU GPU vs Doolittle vs Gaussian Elimination</h1>
      <p>
        Website ini membandingkan empat metode LU pada dataset matriks yang sama. Semua angka runtime memakai policy
        yang seragam: <strong>factorization + solve x_b + solve x_c</strong>, diulang beberapa kali per ukuran, lalu
        diambil median. File I/O ke disk tidak ikut dihitung, tetapi materialisasi hasil yang diperlukan untuk validasi
        tetap menjadi bagian dari pipeline benchmark.
      </p>
      <div class="legend">
        {"".join(legend_items)}
      </div>
    </section>

    <section class="summary-grid">
      {"".join(summary_cards)}
    </section>

    <section class="panel">
      <h2>Runtime Arena</h2>
      <p>
        Setiap sel menampilkan median runtime metode tersebut pada ukuran matriks terkait. Bar di dalam sel
        dinormalisasi terhadap runtime paling lambat pada baris yang sama agar perbedaan relatif langsung terlihat.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Matrix Size</th>
              <th>Block LU CPU</th>
              <th>Block LU GPU</th>
              <th>LU Doolittle</th>
              <th>LU Gaussian</th>
              <th>GPU / Block CPU</th>
              <th>Doolittle / Block CPU</th>
              <th>Gaussian / Block CPU</th>
            </tr>
          </thead>
          <tbody>
            {"".join(runtime_row_html)}
          </tbody>
        </table>
      </div>
      <div class="footnote">
        Perbandingan ini sengaja menggunakan benchmark unified baru, bukan mencampur report lama yang policy timing-nya
        berbeda. Dengan begitu, angka antar-metode bisa dibaca sebagai perbandingan langsung.
      </div>
    </section>

    <section class="panel">
      <h2>Accuracy Check</h2>
      <p>
        Relative error dihitung dengan infinity norm. Error solusi <code>x_b</code> dibandingkan terhadap referensi
        <code>numpy.linalg.solve</code>. Untuk dataset ini, empat metode tetap berada pada skala akurasi yang sangat kecil.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Matrix Size</th>
              <th>LU Err Block CPU</th>
              <th>LU Err Block GPU</th>
              <th>LU Err Doolittle</th>
              <th>LU Err Gaussian</th>
              <th>x_b Err Block CPU</th>
              <th>x_b Err Block GPU</th>
              <th>x_b Err Doolittle</th>
              <th>x_b Err Gaussian</th>
            </tr>
          </thead>
          <tbody>
            {"".join(accuracy_row_html)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Resource Snapshot</h2>
      <p>
        Tabel ini memberi konteks resource untuk satu profile run per metode. Nilai ini bukan median, melainkan snapshot
        representatif yang dipakai sekaligus untuk validasi numerik.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Matrix Size</th>
              <th>Block Size CPU</th>
              <th>Block Size GPU</th>
              <th>GPU Peak Util</th>
              <th>Peak Mem Block CPU</th>
              <th>Peak Mem Block GPU</th>
              <th>Peak Mem Doolittle</th>
              <th>Peak Mem Gaussian</th>
            </tr>
          </thead>
          <tbody>
            {"".join(resource_row_html)}
          </tbody>
        </table>
      </div>
      <div class="footnote">
        Snapshot resource berasal dari satu profile run per metode. Nilainya dipakai sebagai konteks tambahan, bukan
        sebagai pengganti tabel runtime median di atas.
      </div>
    </section>

    <section class="panel">
      <h2>Sources</h2>
      <div class="source-list">
        <div>Unified summary CSV: <code>{summary_path}</code></div>
        <div>Generated website: <code>{output_path}</code></div>
      </div>
    </section>
  </main>
</body>
</html>
"""

    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    root = Path.cwd()
    output_root = root / "hasil_website_4_metode"
    output_root.mkdir(parents=True, exist_ok=True)
    pivot_tol = 1e-12

    cases: list[tuple[int, Path]] = []
    for entry in root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("ukuran "):
            continue
        size = int(entry.name.split()[1])
        cases.append((size, entry))

    rows: list[dict] = []
    for size, case_dir in sorted(cases, key=lambda item: item[0]):
        A, b, c = load_case(case_dir, size)
        size_dir = output_root / f"ukuran_{size}x{size}"
        size_dir.mkdir(parents=True, exist_ok=True)

        for method in METHODS:
            metrics = benchmark_method(method, A, b, c, size=size, pivot_tol=pivot_tol)
            rows.append(metrics)
            write_json(size_dir / f"{method.slug}.json", metrics)
            print(
                f"[OK] {size}x{size} | {method.display_name} | "
                f"median={metrics['elapsed_compute_median_seconds']:.6f}s"
            )

    summary_path = output_root / "ringkasan.csv"
    html_path = output_root / "website_perbandingan_4_metode.html"
    write_summary_csv(summary_path, rows)
    build_html(rows, html_path, summary_path)
    print(f"Summary written to: {summary_path}")
    print(f"Website written to: {html_path}")


if __name__ == "__main__":
    main()
