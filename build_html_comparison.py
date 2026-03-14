#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Row:
    size: int
    size_label: str
    block_size: int
    backend: str
    lu_error: float
    xb_error: float
    xc_error: float
    xb_residual: float
    xc_residual: float
    elapsed: float
    cpu_time: float
    cpu_usage_avg: float
    memory_peak_mb: float
    gpu_peak_percent: float
    gpu_memory_peak_mb: float
    cache_hit_percent: float
    cache_miss_percent: float
    backend_note: str


def read_summary(path: Path) -> dict[int, Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for raw in reader:
            row = Row(
                size=int(raw["size"]),
                size_label=raw["size_label"],
                block_size=int(raw["block_size"]),
                backend=raw["backend"],
                lu_error=float(raw["lu_relative_error_inf"]),
                xb_error=float(raw["x_b_relative_error_inf_vs_numpy"]),
                xc_error=float(raw["x_c_relative_error_inf_vs_numpy"]),
                xb_residual=float(raw["x_b_normalized_residual_inf"]),
                xc_residual=float(raw["x_c_normalized_residual_inf"]),
                elapsed=float(raw["elapsed_seconds_algorithm"]),
                cpu_time=float(raw["cpu_time_seconds"]),
                cpu_usage_avg=float(raw["cpu_usage_avg_percent_total_capacity"]),
                memory_peak_mb=float(raw["memory_peak_mb"]),
                gpu_peak_percent=float(raw["gpu_usage_peak_percent"]),
                gpu_memory_peak_mb=float(raw["gpu_memory_peak_mb"]),
                cache_hit_percent=float(raw["cache_hit_percent_proxy_avg"]),
                cache_miss_percent=float(raw["cache_miss_percent_proxy_avg"]),
                backend_note=raw["backend_note"],
            )
            rows[row.size] = row
        return rows


def fmt_seconds(value: float) -> str:
    return f"{value:.6f} s"


def fmt_ratio(value: float) -> str:
    return f"{value:.2f}x"


def fmt_percent(value: float) -> str:
    return f"{value:.2f}%"


def fmt_mb(value: float) -> str:
    return f"{value:.2f} MB"


def fmt_sci(value: float) -> str:
    return f"{value:.3e}"


def badge_class(winner: str) -> str:
    return {
        "CPU": "badge badge-cpu",
        "GPU": "badge badge-gpu",
        "Tie": "badge badge-tie",
    }[winner]


def render_summary_card(title: str, value: str, subtext: str) -> str:
    return f"""
    <article class="summary-card">
      <div class="summary-title">{html.escape(title)}</div>
      <div class="summary-value">{html.escape(value)}</div>
      <div class="summary-subtext">{html.escape(subtext)}</div>
    </article>
    """


def build_report(cpu_path: Path, gpu_path: Path, output_path: Path) -> None:
    cpu_rows = read_summary(cpu_path)
    gpu_rows = read_summary(gpu_path)
    sizes = sorted(set(cpu_rows) & set(gpu_rows))
    if not sizes:
        raise SystemExit("Tidak ada ukuran yang cocok antara ringkasan CPU dan GPU.")

    performance_rows: list[str] = []
    accuracy_rows: list[str] = []
    winner_counts = {"CPU": 0, "GPU": 0, "Tie": 0}
    ratio_values: list[tuple[int, float]] = []

    for size in sizes:
        cpu = cpu_rows[size]
        gpu = gpu_rows[size]
        ratio_gpu_over_cpu = gpu.elapsed / cpu.elapsed if cpu.elapsed else float("inf")
        winner = "GPU" if gpu.elapsed < cpu.elapsed else "CPU" if cpu.elapsed < gpu.elapsed else "Tie"
        winner_counts[winner] += 1
        ratio_values.append((size, ratio_gpu_over_cpu))

        performance_rows.append(
            f"""
            <tr>
              <td><span class="size-pill">{size} x {size}</span></td>
              <td>{cpu.block_size}</td>
              <td>{fmt_seconds(cpu.elapsed)}</td>
              <td>{fmt_seconds(gpu.elapsed)}</td>
              <td>
                <div class="ratio-cell">
                  <span>{fmt_ratio(ratio_gpu_over_cpu)}</span>
                  <div class="ratio-track">
                    <div class="ratio-fill" style="width:{min(ratio_gpu_over_cpu * 18, 100):.1f}%"></div>
                  </div>
                </div>
              </td>
              <td><span class="{badge_class(winner)}">{winner} faster</span></td>
              <td>{fmt_mb(cpu.memory_peak_mb)}</td>
              <td>{fmt_mb(gpu.memory_peak_mb)}</td>
              <td>{fmt_percent(gpu.gpu_peak_percent)}</td>
            </tr>
            """
        )

        accuracy_rows.append(
            f"""
            <tr>
              <td><span class="size-pill">{size} x {size}</span></td>
              <td>{fmt_sci(cpu.lu_error)}</td>
              <td>{fmt_sci(gpu.lu_error)}</td>
              <td>{fmt_sci(cpu.xb_error)}</td>
              <td>{fmt_sci(gpu.xb_error)}</td>
              <td>{fmt_sci(cpu.xc_error)}</td>
              <td>{fmt_sci(gpu.xc_error)}</td>
              <td>{fmt_sci(cpu.xb_residual)}</td>
              <td>{fmt_sci(gpu.xb_residual)}</td>
            </tr>
            """
        )

    closest_size, closest_ratio = min(ratio_values, key=lambda item: abs(item[1] - 1.0))
    widest_size, widest_ratio = max(ratio_values, key=lambda item: item[1])
    gpu_max_util = max(gpu_rows[size].gpu_peak_percent for size in sizes)
    gpu_max_memory = max(gpu_rows[size].gpu_memory_peak_mb for size in sizes)

    summary_cards = "\n".join(
        [
            render_summary_card(
                "Overall Winner",
                f"CPU wins {winner_counts['CPU']} / {len(sizes)}",
                "Pada seluruh ukuran yang diuji, CPU masih lebih cepat.",
            ),
            render_summary_card(
                "Closest Race",
                f"{closest_size} x {closest_size}",
                f"GPU/CPU time = {fmt_ratio(closest_ratio)}",
            ),
            render_summary_card(
                "Largest GPU Overhead",
                f"{widest_size} x {widest_size}",
                f"GPU/CPU time = {fmt_ratio(widest_ratio)}",
            ),
            render_summary_card(
                "GPU Activity",
                fmt_percent(gpu_max_util),
                f"Peak GPU memory mencapai {fmt_mb(gpu_max_memory)}",
            ),
        ]
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Block LU CPU vs GPU</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --card: rgba(255, 252, 246, 0.88);
      --card-strong: rgba(255, 250, 242, 0.96);
      --ink: #1f2937;
      --muted: #5b6472;
      --line: rgba(46, 56, 77, 0.12);
      --cpu: #b45309;
      --cpu-soft: rgba(245, 158, 11, 0.18);
      --gpu: #166534;
      --gpu-soft: rgba(34, 197, 94, 0.16);
      --tie: #475569;
      --tie-soft: rgba(100, 116, 139, 0.14);
      --accent: #0f766e;
      --accent-soft: rgba(15, 118, 110, 0.12);
      --shadow: 0 24px 60px rgba(31, 41, 55, 0.14);
      --radius: 24px;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(251, 191, 36, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(45, 212, 191, 0.18), transparent 25%),
        linear-gradient(180deg, #fffdf8 0%, var(--bg) 100%);
    }}

    .page {{
      width: min(1240px, calc(100vw - 32px));
      margin: 32px auto 56px;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      border-radius: 32px;
      padding: 32px;
      background:
        linear-gradient(140deg, rgba(17, 24, 39, 0.92), rgba(15, 118, 110, 0.92)),
        linear-gradient(120deg, #111827, #0f766e);
      color: #f8fafc;
      box-shadow: var(--shadow);
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, rgba(255, 255, 255, 0.06) 1px, transparent 1px),
        linear-gradient(rgba(255, 255, 255, 0.06) 1px, transparent 1px);
      background-size: 28px 28px;
      mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.75), transparent);
      pointer-events: none;
    }}

    .eyebrow {{
      position: relative;
      z-index: 1;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.14);
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    h1 {{
      position: relative;
      z-index: 1;
      margin: 18px 0 12px;
      max-width: 820px;
      font-size: clamp(32px, 5vw, 58px);
      line-height: 0.98;
    }}

    .hero p {{
      position: relative;
      z-index: 1;
      margin: 0;
      max-width: 760px;
      color: rgba(248, 250, 252, 0.84);
      font-size: 17px;
      line-height: 1.65;
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
      background: var(--card);
      border: 1px solid rgba(255, 255, 255, 0.36);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }}

    .summary-title {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .summary-value {{
      margin-top: 10px;
      font-size: 29px;
      font-weight: 700;
    }}

    .summary-subtext {{
      margin-top: 8px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }}

    .panel {{
      margin-top: 26px;
      border-radius: var(--radius);
      padding: 24px;
      background: var(--card-strong);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}

    .panel-header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }}

    .panel-title {{
      margin: 0;
      font-size: 28px;
    }}

    .panel-note {{
      margin: 6px 0 0;
      color: var(--muted);
      line-height: 1.6;
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

    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 16px 14px;
      background: #f8fafc;
      color: #0f172a;
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      text-align: left;
      border-bottom: 1px solid var(--line);
    }}

    tbody td {{
      padding: 15px 14px;
      border-bottom: 1px solid var(--line);
      vertical-align: middle;
    }}

    tbody tr:nth-child(even) {{
      background: #fcfdfd;
    }}

    tbody tr:hover {{
      background: #f7fbfb;
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

    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 13px;
      letter-spacing: 0.01em;
    }}

    .badge-cpu {{
      background: var(--cpu-soft);
      color: var(--cpu);
    }}

    .badge-gpu {{
      background: var(--gpu-soft);
      color: var(--gpu);
    }}

    .badge-tie {{
      background: var(--tie-soft);
      color: var(--tie);
    }}

    .ratio-cell {{
      display: grid;
      gap: 8px;
      min-width: 120px;
    }}

    .ratio-track {{
      height: 9px;
      border-radius: 999px;
      background: #e2e8f0;
      overflow: hidden;
    }}

    .ratio-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #f59e0b, #ef4444);
    }}

    .footnote {{
      margin-top: 18px;
      padding: 18px 20px;
      border-radius: 18px;
      background: var(--accent-soft);
      color: #134e4a;
      line-height: 1.7;
    }}

    .source-list {{
      display: grid;
      gap: 8px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }}

    code {{
      padding: 2px 6px;
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.06);
      font-family: Consolas, monospace;
      font-size: 0.92em;
    }}

    @media (max-width: 980px) {{
      .page {{
        width: min(100vw - 20px, 100%);
        margin: 12px auto 28px;
      }}

      .hero,
      .panel {{
        padding: 20px;
        border-radius: 24px;
      }}

      .summary-grid {{
        grid-template-columns: 1fr;
      }}

      .panel-header {{
        display: block;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="eyebrow">Benchmark Report • Block LU Factorization</div>
      <h1>Block LU CPU vs GPU</h1>
      <p>
        Perbandingan ini dirangkum dari dua eksekusi terpisah: satu memakai backend CPU dan satu memakai backend GPU
        CuPy + CUDA. Fokus utamanya adalah kecepatan algoritma, jejak memori, aktivitas GPU, dan kestabilan numerik.
      </p>
    </section>

    <section class="summary-grid">
      {summary_cards}
    </section>

    <section class="panel">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">Performance Table</h2>
          <p class="panel-note">
            Kolom <code>GPU / CPU time</code> menunjukkan berapa kali runtime GPU dibanding CPU.
            Nilai di atas <code>1.00x</code> berarti GPU lebih lambat pada ukuran tersebut.
          </p>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Matrix Size</th>
              <th>Block Size</th>
              <th>CPU Time</th>
              <th>GPU Time</th>
              <th>GPU / CPU Time</th>
              <th>Winner</th>
              <th>CPU Peak Memory</th>
              <th>GPU Run Peak Memory</th>
              <th>GPU Peak Util</th>
            </tr>
          </thead>
          <tbody>
            {"".join(performance_rows)}
          </tbody>
        </table>
      </div>
      <div class="footnote">
        GPU memang aktif, tetapi ukuran matriks yang diuji masih relatif kecil. Pada rentang ini, overhead upload data,
        sinkronisasi, dan kernel launch masih lebih dominan daripada keuntungan komputasi paralel.
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">Accuracy Table</h2>
          <p class="panel-note">
            Error relatif dihitung dengan infinity norm. Solusi <code>x_b</code> dan <code>x_c</code> dibandingkan
            terhadap referensi <code>numpy.linalg.solve</code>.
          </p>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Matrix Size</th>
              <th>LU Error CPU</th>
              <th>LU Error GPU</th>
              <th>x_b Error CPU</th>
              <th>x_b Error GPU</th>
              <th>x_c Error CPU</th>
              <th>x_c Error GPU</th>
              <th>x_b Residual CPU</th>
              <th>x_b Residual GPU</th>
            </tr>
          </thead>
          <tbody>
            {"".join(accuracy_rows)}
          </tbody>
        </table>
      </div>
      <div class="footnote">
        Secara numerik, CPU dan GPU memberikan hasil yang hampir identik. Perbedaan yang terlihat tetap berada dalam
        skala machine precision untuk data uji ini.
      </div>
    </section>

    <section class="panel">
      <h2 class="panel-title">Sources</h2>
      <div class="source-list">
        <div>CPU summary: <code>{cpu_path}</code></div>
        <div>GPU summary: <code>{gpu_path}</code></div>
        <div>Generated HTML: <code>{output_path}</code></div>
      </div>
    </section>
  </main>
</body>
</html>
"""

    output_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Build an HTML comparison for CPU vs GPU Block LU summaries."
    )
    parser.add_argument(
        "--cpu-summary",
        type=Path,
        default=root / "hasil" / "ringkasan.csv",
        help="Path to CPU summary CSV.",
    )
    parser.add_argument(
        "--gpu-summary",
        type=Path,
        default=root / "hasil_gpu" / "ringkasan.csv",
        help="Path to GPU summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "hasil" / "perbandingan_block_lu_cpu_vs_gpu.html",
        help="Output HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_path = args.cpu_summary.resolve()
    gpu_path = args.gpu_summary.resolve()
    output_path = args.output.resolve()

    if not cpu_path.exists():
        raise SystemExit(f"File tidak ditemukan: {cpu_path}")
    if not gpu_path.exists():
        raise SystemExit(f"File tidak ditemukan: {gpu_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_report(cpu_path=cpu_path, gpu_path=gpu_path, output_path=output_path)
    print(f"HTML report generated at: {output_path}")


if __name__ == "__main__":
    main()
