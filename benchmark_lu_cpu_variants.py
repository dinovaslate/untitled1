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

from generate_hasil_lu import (
    BackendInfo,
    PerformanceMonitor,
    block_lu_cpu,
    block_size_for,
    load_case,
    normalized_residual_inf,
    relative_error_inf,
    solve_via_lu,
)


@dataclass(frozen=True)
class Variant:
    slug: str
    display_name: str
    factorize: Callable[[np.ndarray, int, float], tuple[np.ndarray, np.ndarray, np.ndarray]]
    note: str


def factorize_doolittle(matrix: np.ndarray, block_size: int, pivot_tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del block_size
    block = np.array(matrix, copy=True, dtype=np.float64)
    n = block.shape[0]
    L = np.eye(n, dtype=block.dtype)
    U = np.zeros((n, n), dtype=block.dtype)

    for j in range(n):
        for i in range(j + 1):
            U[i, j] = block[i, j] - L[i, :i] @ U[:i, j]

        pivot = U[j, j]
        if abs(float(pivot)) < pivot_tol:
            raise np.linalg.LinAlgError(
                f"Near-zero pivot encountered at local index {j}: {pivot}."
            )

        for i in range(j + 1, n):
            L[i, j] = (block[i, j] - L[i, :j] @ U[:j, j]) / pivot

    combined = np.tril(L, k=-1) + U
    return combined, L, U


def factorize_gaussian_elimination(
    matrix: np.ndarray,
    block_size: int,
    pivot_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del block_size
    A = np.array(matrix, copy=True, dtype=np.float64)
    n = A.shape[0]

    for k in range(n):
        pivot = A[k, k]
        if abs(float(pivot)) < pivot_tol:
            raise np.linalg.LinAlgError(
                f"Near-zero pivot encountered at index {k}: {pivot}."
            )
        if k + 1 >= n:
            continue
        A[k + 1 :, k] /= pivot
        A[k + 1 :, k + 1 :] -= np.outer(A[k + 1 :, k], A[k, k + 1 :])

    L = np.tril(A, k=-1) + np.eye(n, dtype=A.dtype)
    U = np.triu(A)
    return A, L, U


VARIANTS: tuple[Variant, ...] = (
    Variant(
        slug="block_lu",
        display_name="Block LU",
        factorize=block_lu_cpu,
        note="Blocked factorization with block diagonal panel plus Schur update.",
    ),
    Variant(
        slug="doolittle",
        display_name="Doolittle",
        factorize=factorize_doolittle,
        note="Naive full-matrix Doolittle LU without pivoting.",
    ),
    Variant(
        slug="gaussian_elimination",
        display_name="Gaussian Elimination LU",
        factorize=factorize_gaussian_elimination,
        note="In-place elimination form of LU without pivoting.",
    ),
)


def choose_repetitions(size: int) -> int:
    if size <= 16:
        return 32
    if size <= 32:
        return 20
    if size <= 128:
        return 10
    if size <= 256:
        return 5
    return 3


def fmt_seconds(value: float) -> str:
    return f"{value:.6f} s"


def fmt_ratio(value: float) -> str:
    return f"{value:.2f}x"


def fmt_sci(value: float) -> str:
    return f"{value:.3e}"


def fmt_mb(value: float) -> str:
    return f"{value:.2f} MB"


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


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


def benchmark_variant(
    variant: Variant,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    size: int,
    pivot_tol: float,
) -> dict:
    repetitions = choose_repetitions(size)
    block_size = block_size_for(size, "cpu")

    warmup_combined, warmup_L, warmup_U = variant.factorize(A, block_size, pivot_tol)
    del warmup_combined

    backend_info = BackendInfo(backend="cpu", note="CPU benchmark for LU variant comparison.")
    monitor = PerformanceMonitor(backend_info)
    monitor.start()

    timings: list[float] = []
    combined = None
    L = warmup_L
    U = warmup_U
    for _ in range(repetitions):
        start = time.perf_counter()
        combined, L, U = variant.factorize(A, block_size, pivot_tol)
        timings.append(time.perf_counter() - start)

    runtime_metrics = monitor.stop()

    x_b = solve_via_lu(L, U, b)
    x_c = solve_via_lu(L, U, c)
    x_b_reference = np.linalg.solve(A, b)
    x_c_reference = np.linalg.solve(A, c)

    assert combined is not None
    return {
        "size": size,
        "size_label": f"ukuran_{size}x{size}",
        "variant": variant.slug,
        "variant_display_name": variant.display_name,
        "variant_note": variant.note,
        "block_size": block_size,
        "repetitions": repetitions,
        "elapsed_factorization_median_seconds": float(statistics.median(timings)),
        "elapsed_factorization_mean_seconds": float(statistics.mean(timings)),
        "elapsed_factorization_min_seconds": float(min(timings)),
        "elapsed_factorization_max_seconds": float(max(timings)),
        "lu_relative_error_inf": relative_error_inf(L @ U, A),
        "x_b_relative_error_inf_vs_numpy": relative_error_inf(x_b, x_b_reference),
        "x_c_relative_error_inf_vs_numpy": relative_error_inf(x_c, x_c_reference),
        "x_b_normalized_residual_inf": normalized_residual_inf(A, x_b, b),
        "x_c_normalized_residual_inf": normalized_residual_inf(A, x_c, c),
        "combined_lu_shape": list(combined.shape),
        **runtime_metrics,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "size",
        "size_label",
        "variant",
        "variant_display_name",
        "block_size",
        "repetitions",
        "elapsed_factorization_median_seconds",
        "elapsed_factorization_mean_seconds",
        "elapsed_factorization_min_seconds",
        "elapsed_factorization_max_seconds",
        "lu_relative_error_inf",
        "x_b_relative_error_inf_vs_numpy",
        "x_c_relative_error_inf_vs_numpy",
        "x_b_normalized_residual_inf",
        "x_c_normalized_residual_inf",
        "cpu_time_seconds",
        "cpu_usage_avg_percent_total_capacity",
        "cpu_usage_peak_percent_total_capacity",
        "memory_peak_mb",
        "cache_hit_percent_proxy_avg",
        "cache_miss_percent_proxy_avg",
        "variant_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: to_builtin(row.get(field)) for field in fieldnames})


def build_html(rows: list[dict], output_path: Path) -> None:
    by_size: dict[int, list[dict]] = {}
    for row in rows:
        by_size.setdefault(int(row["size"]), []).append(row)

    performance_rows: list[str] = []
    accuracy_rows: list[str] = []
    insights: list[str] = []

    for size in sorted(by_size):
        group = sorted(by_size[size], key=lambda item: item["elapsed_factorization_median_seconds"])
        best = group[0]
        block = next(row for row in group if row["variant"] == "block_lu")
        doolittle = next(row for row in group if row["variant"] == "doolittle")
        gaussian = next(row for row in group if row["variant"] == "gaussian_elimination")

        performance_rows.append(
            f"""
            <tr>
              <td><span class="size-pill">{size} x {size}</span></td>
              <td>{fmt_seconds(block["elapsed_factorization_median_seconds"])}</td>
              <td>{fmt_seconds(doolittle["elapsed_factorization_median_seconds"])}</td>
              <td>{fmt_seconds(gaussian["elapsed_factorization_median_seconds"])}</td>
              <td>{fmt_ratio(doolittle["elapsed_factorization_median_seconds"] / block["elapsed_factorization_median_seconds"])}</td>
              <td>{fmt_ratio(gaussian["elapsed_factorization_median_seconds"] / block["elapsed_factorization_median_seconds"])}</td>
              <td><span class="winner-badge">{html.escape(best["variant_display_name"])}</span></td>
            </tr>
            """
        )

        accuracy_rows.append(
            f"""
            <tr>
              <td><span class="size-pill">{size} x {size}</span></td>
              <td>{fmt_sci(block["lu_relative_error_inf"])}</td>
              <td>{fmt_sci(doolittle["lu_relative_error_inf"])}</td>
              <td>{fmt_sci(gaussian["lu_relative_error_inf"])}</td>
              <td>{fmt_sci(block["x_b_relative_error_inf_vs_numpy"])}</td>
              <td>{fmt_sci(doolittle["x_b_relative_error_inf_vs_numpy"])}</td>
              <td>{fmt_sci(gaussian["x_b_relative_error_inf_vs_numpy"])}</td>
              <td>{fmt_sci(block["x_c_relative_error_inf_vs_numpy"])}</td>
              <td>{fmt_sci(doolittle["x_c_relative_error_inf_vs_numpy"])}</td>
              <td>{fmt_sci(gaussian["x_c_relative_error_inf_vs_numpy"])}</td>
            </tr>
            """
        )

        insights.append(
            f"<li>Pada ukuran <strong>{size} x {size}</strong>, algoritma tercepat adalah "
            f"<strong>{html.escape(best['variant_display_name'])}</strong>.</li>"
        )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LU Variant Comparison</title>
  <style>
    :root {{
      --bg: #f6f1e7;
      --ink: #172033;
      --muted: #5a6474;
      --panel: rgba(255, 253, 249, 0.95);
      --line: rgba(23, 32, 51, 0.12);
      --accent: #9a3412;
      --accent-soft: rgba(217, 119, 6, 0.14);
      --good: #166534;
      --good-soft: rgba(34, 197, 94, 0.16);
      --shadow: 0 24px 54px rgba(23, 32, 51, 0.12);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217, 119, 6, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(16, 185, 129, 0.16), transparent 26%),
        linear-gradient(180deg, #fffdf7 0%, var(--bg) 100%);
    }}

    .page {{
      width: min(1220px, calc(100vw - 30px));
      margin: 30px auto 48px;
    }}

    .hero {{
      padding: 30px;
      border-radius: 30px;
      background: linear-gradient(135deg, rgba(23, 32, 51, 0.95), rgba(154, 52, 18, 0.92));
      color: white;
      box-shadow: var(--shadow);
    }}

    .hero h1 {{
      margin: 0 0 12px;
      font-size: clamp(30px, 5vw, 54px);
      line-height: 0.98;
    }}

    .hero p {{
      max-width: 840px;
      margin: 0;
      color: rgba(255, 255, 255, 0.84);
      line-height: 1.7;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin-top: 22px;
    }}

    .card {{
      padding: 20px;
      border-radius: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}

    .card h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}

    .card p, .card li {{
      color: var(--muted);
      line-height: 1.65;
    }}

    .panel {{
      margin-top: 24px;
      padding: 24px;
      border-radius: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}

    .panel h2 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}

    .panel p {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.65;
    }}

    .table-wrap {{
      overflow-x: auto;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: white;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
    }}

    th, td {{
      padding: 14px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }}

    thead th {{
      background: #fbfcfd;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    tbody tr:nth-child(even) {{
      background: #fcfdfd;
    }}

    .size-pill {{
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      background: #eff6ff;
      color: #1d4ed8;
      font-weight: 700;
      font-size: 13px;
    }}

    .winner-badge {{
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--good-soft);
      color: var(--good);
      font-weight: 700;
      font-size: 13px;
    }}

    code {{
      padding: 2px 6px;
      border-radius: 8px;
      background: rgba(23, 32, 51, 0.06);
      font-family: Consolas, monospace;
    }}

    .note {{
      margin-top: 18px;
      padding: 16px 18px;
      border-radius: 18px;
      background: var(--accent-soft);
      color: #7c2d12;
      line-height: 1.7;
    }}

    ul {{
      margin: 0;
      padding-left: 20px;
    }}

    @media (max-width: 960px) {{
      .page {{
        width: min(100vw - 18px, 100%);
        margin: 10px auto 24px;
      }}

      .hero, .panel, .card {{
        padding: 18px;
        border-radius: 20px;
      }}

      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>CPU LU Variants Comparison</h1>
      <p>
        Perbandingan ini menguji tiga implementasi CPU pada matriks yang sama:
        <code>Block LU</code>, <code>Doolittle</code>, dan <code>Gaussian Elimination LU</code>.
        Runtime factorization diukur sebagai median beberapa repetisi agar noise ukuran kecil tidak mendominasi hasil.
      </p>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Metode</h2>
        <ul>
          <li><strong>Block LU</strong>: panel diagonal + trailing Schur update.</li>
          <li><strong>Doolittle</strong>: formulasi LU penuh dengan loop level tinggi.</li>
          <li><strong>Gaussian Elimination LU</strong>: update in-place berbasis outer product.</li>
        </ul>
      </article>
      <article class="card">
        <h2>Timing Policy</h2>
        <ul>
          <li>Timing yang dibandingkan adalah <strong>factorization-only</strong>.</li>
          <li>Setiap ukuran diulang beberapa kali dan dipakai <strong>median</strong>.</li>
          <li>Solusi <code>x_b</code> dan <code>x_c</code> dihitung terpisah untuk verifikasi numerik.</li>
        </ul>
      </article>
      <article class="card">
        <h2>Interpretation</h2>
        <ul>
          {"".join(insights)}
        </ul>
      </article>
    </section>

    <section class="panel">
      <h2>Performance Table</h2>
      <p>
        Kolom rasio menunjukkan seberapa lambat algoritma biasa relatif terhadap <code>Block LU</code>.
        Nilai di atas <code>1.00x</code> berarti lebih lambat daripada Block LU.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Matrix Size</th>
              <th>Block LU</th>
              <th>Doolittle</th>
              <th>Gaussian Elimination LU</th>
              <th>Doolittle / Block</th>
              <th>Gaussian / Block</th>
              <th>Fastest</th>
            </tr>
          </thead>
          <tbody>
            {"".join(performance_rows)}
          </tbody>
        </table>
      </div>
      <div class="note">
        Bila hasil tidak sesuai ekspektasi teoretis, penyebab paling mungkin adalah detail implementasi.
        Di Python/NumPy, <code>Gaussian Elimination LU</code> memakai update slice dan <code>np.outer</code> yang
        berjalan di level C, sedangkan <code>Doolittle</code> dan bagian panel dari <code>Block LU</code> masih
        membawa overhead loop Python yang lebih tinggi.
      </div>
    </section>

    <section class="panel">
      <h2>Accuracy Table</h2>
      <p>
        Akurasi dinilai dengan infinity norm. Error solusi dibandingkan terhadap
        <code>numpy.linalg.solve</code> sebagai referensi.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Matrix Size</th>
              <th>LU Error Block</th>
              <th>LU Error Doolittle</th>
              <th>LU Error Gaussian</th>
              <th>x_b Error Block</th>
              <th>x_b Error Doolittle</th>
              <th>x_b Error Gaussian</th>
              <th>x_c Error Block</th>
              <th>x_c Error Doolittle</th>
              <th>x_c Error Gaussian</th>
            </tr>
          </thead>
          <tbody>
            {"".join(accuracy_rows)}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    root = Path.cwd()
    output_root = root / "hasil_varian_cpu"
    output_root.mkdir(parents=True, exist_ok=True)

    pivot_tol = 1e-12
    rows: list[dict] = []

    cases: list[tuple[int, Path]] = []
    for size_dir in root.iterdir():
        if not size_dir.is_dir() or not size_dir.name.startswith("ukuran "):
            continue
        size_text = size_dir.name.split()[1]
        cases.append((int(size_text), size_dir))

    for size, size_dir in sorted(cases, key=lambda item: item[0]):
        A, b, c = load_case(size_dir, size)

        for variant in VARIANTS:
            metrics = benchmark_variant(variant, A, b, c, size=size, pivot_tol=pivot_tol)
            rows.append(metrics)
            case_dir = output_root / f"ukuran_{size}x{size}"
            case_dir.mkdir(parents=True, exist_ok=True)
            write_json(case_dir / f"{variant.slug}.json", metrics)
            print(
                f"[OK] {size}x{size} | {variant.display_name} | "
                f"median={metrics['elapsed_factorization_median_seconds']:.6f}s"
            )

    summary_path = output_root / "ringkasan.csv"
    html_path = output_root / "perbandingan_varian_lu_cpu.html"
    write_summary_csv(summary_path, rows)
    build_html(rows, html_path)
    print(f"Summary written to: {summary_path}")
    print(f"HTML written to   : {html_path}")


if __name__ == "__main__":
    main()
