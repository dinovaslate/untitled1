# Banded Linear System Experiments

This repository contains implementations and experiments for solving banded linear systems, with two main viewpoints:

- `Block LU` as a general-purpose method for banded systems that do not have a simple specialized solver.
- `Thomas algorithm` as the specialized solver for tridiagonal systems.

The current codebase includes CPU and GPU variants of Block LU, benchmark scripts, generated tridiagonal datasets, and a report-oriented workflow used to compare numerical accuracy, runtime, and resource usage.

## Main Files

- `block_lu_cpu.py`: Block LU factorization on CPU.
- `block_lu_gpu.py`: Block LU factorization with GPU-assisted solves and Schur complement updates.
- `thomas_algorithm.py`: Thomas algorithm for tridiagonal systems.
- `benchmark_tridiagonal_html.py`: tridiagonal benchmark pipeline and HTML output generation.
- `benchmark_lu_cpu_variants.py`: CPU-side comparison between Block LU, Doolittle, and Gaussian elimination.
- `generate_tridiagonal_datasets.py`: generator for large tridiagonal test matrices.

## Project Direction

The experiments in this repository separate two cases that should not be mixed.

For a general banded system, Block LU is still a reasonable fallback because it does not rely on tridiagonal-specific structure. In this implementation, the blocked organization is used to improve locality and to make the expensive trailing update more efficient than a naive non-blocked LU path.

For a special banded system such as the tridiagonal matrices used in the benchmark, Thomas algorithm is the right tool. It works directly on compact 1D bands, which is why its runtime and memory footprint are much smaller than Block LU on the same structured input.

## GitHub-Friendly Tables

The report and HTML dashboard contain the full discussion, but the tables below are the shortest GitHub-visible summary for the tridiagonal benchmark up to `512 x 512`.

Important note: the memory numbers below use **peak algorithm memory**, not process RSS. In other words, they estimate the core arrays actually held by the method during factorization and solve, which is the right lens for comparing Thomas against dense Block LU.

### Runtime and Algorithm Memory

| Size | Block LU backend | Block LU time (ms) | Thomas time (ms) | Block LU algo peak mem (MB) | Thomas algo peak mem (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 x 8 | CPU | 0.2944 | 0.0756 | 0.0028 | 0.0007 |
| 16 x 16 | CPU | 0.6092 | 0.1500 | 0.0095 | 0.0014 |
| 32 x 32 | CPU | 1.3113 | 0.2959 | 0.0347 | 0.0029 |
| 128 x 128 | CPU | 4.8423 | 1.1687 | 0.5215 | 0.0117 |
| 256 x 256 | CPU | 11.2256 | 2.3781 | 2.0430 | 0.0234 |
| 512 x 512 | CPU + GPU | 21.5605 | 5.3477 | 14.2813 | 0.0468 |

This is the clearest reason Thomas dominates on the tridiagonal benchmark: its runtime is lower, and its memory stays genuinely linear instead of inheriting the dense working-set cost of Block LU.

### Accuracy Snapshot

| Size | Block LU LU err | Thomas LU err | Block LU `x_b` err | Thomas `x_b` err |
| --- | ---: | ---: | ---: | ---: |
| 8 x 8 | 1.850e-17 | 1.850e-17 | 0.000e+00 | 0.000e+00 |
| 16 x 16 | 1.850e-17 | 1.850e-17 | 1.110e-16 | 1.110e-16 |
| 32 x 32 | 1.850e-17 | 1.850e-17 | 1.110e-16 | 1.110e-16 |
| 128 x 128 | 1.850e-17 | 1.850e-17 | 1.110e-16 | 1.110e-16 |
| 256 x 256 | 1.850e-17 | 1.850e-17 | 1.110e-16 | 1.110e-16 |
| 512 x 512 | 1.850e-17 | 1.850e-17 | 2.220e-16 | 1.110e-16 |

The important point is that the speedup is not coming from a numerical compromise. Thomas remains at the same very small error scale while being substantially cheaper.

### Resource Snapshot

| Size | Block LU backend | Block LU CPU avg (%) | Thomas CPU avg (%) | Block LU GPU peak (%) | Thomas GPU peak (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 x 8 | CPU | 6.39 | 5.67 | 0 | 0 |
| 16 x 16 | CPU | 6.15 | 6.86 | 0 | 0 |
| 32 x 32 | CPU | 6.32 | 6.23 | 0 | 0 |
| 128 x 128 | CPU | 5.66 | 5.70 | 0 | 0 |
| 256 x 256 | CPU | 5.26 | 5.08 | 0 | 0 |
| 512 x 512 | CPU + GPU | 4.49 | 5.11 | 36 | 0 |

For the GPU-assisted Block LU path, device activity is visible at `512 x 512`, but the problem is still too small for that overhead to beat a compact tridiagonal solver.

### Full Artifacts

- HTML dashboard: [website_perbandingan_tridiagonal_dengan_thomas.html](hasil_tridiagonal_html_dengan_thomas/website_perbandingan_tridiagonal_dengan_thomas.html)
- Summary CSV: [ringkasan.csv](hasil_tridiagonal_html_dengan_thomas/ringkasan.csv)
- Latest report PDF: [laporan_algoritma_block_lu_thomas_professor_tone_final_v26.pdf](hasil_tridiagonal_html_dengan_thomas/laporan_algoritma_block_lu_thomas_professor_tone_final_v26.pdf)

## High-Level Findings

- For tridiagonal systems, Thomas algorithm is the preferred solver because it has `O(n)` time and `O(n)` memory complexity.
- Block LU remains relevant as a more general method for banded systems that cannot be reduced to Thomas or a close variant.
- In the current implementation, Block LU uses CPU only for `n < 512` and switches to CPU + GPU assistance for `n >= 512`.
- The practical performance story is dominated by structure and locality, not just by raw FLOP counts.

## Repository Notes

- The repository contains generated experiment artifacts and report drafts produced during the study.
- The tridiagonal benchmark tables in this README are mirrored by the generated HTML dashboard and summary CSV in `hasil_tridiagonal_html_dengan_thomas/`.
