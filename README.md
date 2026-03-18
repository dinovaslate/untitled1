# Banded Linear System Experiments

Repository ini berisi implementasi dan eksperimen untuk dua pendekatan pada sistem linear banded:

- `Block LU` sebagai metode umum untuk banded system yang belum mempunyai solver compact yang sederhana.
- `Thomas 4-Diagonal` sebagai solver khusus untuk matriks dengan lower bandwidth `p = 1` dan upper bandwidth `q = 2`.

Pada revisi terakhir, eksperimen utama untuk `Soal 1` tidak lagi memakai data tridiagonal sintetis. Seluruh angka utama sekarang berasal langsung dari matriks yang diberikan pada folder `ukuran 8 x 8` sampai `ukuran 512 x 512`.

## File Utama

- `block_lu_cpu.py`: Block LU pada CPU.
- `block_lu_gpu.py`: Block LU dengan bantuan GPU pada triangular solve dan Schur complement.
- `thomas_algorithm.py`: solver Thomas 4-diagonal untuk matriks dengan `p = 1`, `q = 2`.
- `algorithm_memory.py`: tracker peak core memory dari array-array algoritma yang benar-benar hidup.
- `benchmark_banded4_given_data.py`: benchmark utama pada data asli `8 x 8` sampai `512 x 512`.
- `generate_hasil_lu.py`: runner Block LU untuk menghasilkan artefak LU, solusi, dan metrik per ukuran.

## Ringkasan Hasil Utama

Eksperimen utama ada di folder [hasil_banded4_aktual](hasil_banded4_aktual). Ringkasan numerik dan solusi yang direferensikan laporan ada di:

- [ringkasan.csv](hasil_banded4_aktual/ringkasan.csv)
- [ringkasan.json](hasil_banded4_aktual/ringkasan.json)
- [website_perbandingan_banded4_aktual.html](hasil_banded4_aktual/website_perbandingan_banded4_aktual.html)

Jika ingin langsung melihat rekap solusi, buka [ringkasan.csv](hasil_banded4_aktual/ringkasan.csv). File itu adalah sumber utama yang dipakai untuk tabel eksperimen pada laporan.

### Runtime dan Peak Algorithm Memory

| Size | Block LU backend | Block LU time (ms) | Thomas 4-diagonal time (ms) | Block LU algo peak mem (MB) | Thomas 4-diagonal algo peak mem (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 x 8 | CPU | 0.7914 | 0.3704 | 0.0022 | 0.0007 |
| 16 x 16 | CPU | 1.3629 | 0.4600 | 0.0083 | 0.0014 |
| 32 x 32 | CPU | 2.5960 | 0.6577 | 0.0322 | 0.0029 |
| 128 x 128 | CPU | 7.9704 | 1.8462 | 0.5039 | 0.0117 |
| 256 x 256 | CPU | 17.9436 | 3.4204 | 2.0078 | 0.0234 |
| 512 x 512 | CPU + GPU | 36.6547 | 6.6125 | 8.0156 | 0.0468 |

Polanya jelas: pada matriks asli dengan `p = 1` dan `q = 2`, solver 4-diagonal selalu lebih cepat daripada Block LU, dan okupasi memorinya tetap linear.

### Accuracy Snapshot

| Size | Block LU LU err | Thomas LU err | Block LU `x_b` err | Thomas `x_b` err | Block LU `x_c` err | Thomas `x_c` err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 x 8 | 1.468e-16 | 1.468e-16 | 1.117e-15 | 9.930e-16 | 7.563e-16 | 3.781e-16 |
| 16 x 16 | 8.436e-17 | 8.436e-17 | 3.244e-16 | 2.595e-16 | 2.451e-16 | 1.838e-16 |
| 32 x 32 | 4.696e-16 | 4.696e-16 | 1.789e-15 | 1.715e-15 | 2.026e-15 | 1.075e-15 |
| 128 x 128 | 2.786e-15 | 2.786e-15 | 2.196e-15 | 3.675e-14 | 7.328e-15 | 8.200e-15 |
| 256 x 256 | 3.528e-22 | 3.528e-22 | 3.328e-14 | 3.790e-14 | 3.637e-14 | 7.092e-14 |
| 512 x 512 | 8.427e-15 | 8.427e-15 | 2.219e-15 | 2.358e-15 | 2.614e-15 | 2.365e-15 |

Error tetap kecil pada kedua metode. Pada ukuran `128` dan `256`, error solusi naik ke orde `1e-14`, yang konsisten dengan condition number data asli yang memang membesar tajam pada ukuran tersebut.

### Resource Snapshot

| Size | Block LU backend | Block LU CPU avg (%) | Thomas CPU avg (%) | Block LU GPU peak (%) | Thomas GPU peak (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 x 8 | CPU | 6.55 | 5.52 | 0 | 0 |
| 16 x 16 | CPU | 5.53 | 5.25 | 0 | 0 |
| 32 x 32 | CPU | 5.67 | 6.15 | 0 | 0 |
| 128 x 128 | CPU | 5.83 | 5.35 | 0 | 0 |
| 256 x 256 | CPU | 10.21 | 5.25 | 0 | 0 |
| 512 x 512 | CPU + GPU | 4.14 | 6.50 | 35 | 0 |

Untuk `512 x 512`, Block LU memang mulai memakai GPU, tetapi ukuran masalah masih terlalu kecil untuk menutup overhead jalur dense umum.

## Cara Menjalankan

Gunakan Python `3.10+`.

### CPU only

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas psutil
```

### GPU di Windows

1. Pastikan driver NVIDIA aktif:

```powershell
nvidia-smi
```

2. Install `CUDA Toolkit 12.x`.
3. Buka PowerShell baru lalu verifikasi:

```powershell
$env:CUDA_PATH
where.exe nvcc
Get-ChildItem "$env:CUDA_PATH\bin\nvrtc64_*.dll"
```

4. Install dependensi Python:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas psutil
pip install cupy-cuda12x
```

5. Verifikasi CuPy:

```powershell
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'].decode()); x=cp.arange(10); print(int(x.sum().get()))"
```

Sebagai fallback, CuPy juga bisa memasang komponen CUDA lewat:

```powershell
pip install "cupy-cuda12x[ctk]"
```

### Menjalankan benchmark utama pada data asli

```powershell
python benchmark_banded4_given_data.py --input-root . --output-root hasil_banded4_aktual
```

Script ini akan menghasilkan:

- `hasil_banded4_aktual/ringkasan.csv`
- `hasil_banded4_aktual/ringkasan.json`
- `hasil_banded4_aktual/ringkasan_detail.json`
- `hasil_banded4_aktual/website_perbandingan_banded4_aktual.html`
- `hasil_banded4_aktual/solusi/*.txt`

File solusi per ukuran disimpan eksplisit pada `hasil_banded4_aktual/solusi/`, misalnya:

- `block_lu_x_b_512x512.txt`
- `block_lu_x_c_512x512.txt`
- `thomas4_x_b_512x512.txt`
- `thomas4_x_c_512x512.txt`

### Menjalankan Block LU saja pada data yang diberikan

```powershell
python generate_hasil_lu.py --input-root . --output-root hasil_block_lu --backend auto --sizes 8 16 32 128 256 512
```

## Catatan

- `Block LU` di repo ini masih bekerja dengan matriks kerja dense, jadi kompleksitas waktunya tetap `O(n^3)` dan memorinya `O(n^2)`.
- `Thomas 4-diagonal` bekerja langsung pada empat pita 1D, jadi kompleksitas waktunya `O(n)` dan memorinya `O(n)`.
- Angka memory yang dilaporkan pada benchmark utama adalah **peak core memory**, yaitu peak dari buffer array algoritma yang hidup saat factorization dan solve, bukan sekadar RSS proses.

## Laporan

Laporan gabungan terakhir yang memuat pembaruan eksperimen ini ada di:

- [laporan_gabungan_technical_reports_v25.pdf](hasil_tridiagonal_html_dengan_thomas/laporan_gabungan_technical_reports_v25.pdf)

## Referensi Instalasi Resmi

- CuPy installation guide: `https://docs.cupy.dev/en/stable/install.html`
- NVIDIA CUDA Installation Guide for Microsoft Windows: `https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/`
