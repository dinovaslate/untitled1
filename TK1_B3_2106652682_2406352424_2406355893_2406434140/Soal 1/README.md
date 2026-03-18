# README Soal 1

## Isi Folder

- `block_lu_cpu.py`: implementasi Block LU pada CPU.
- `block_lu_gpu.py`: implementasi Block LU dengan bantuan GPU.
- `thomas_algorithm.py`: implementasi solver Thomas 4-diagonal untuk matriks dengan `p = 1` dan `q = 2`.
- `algorithm_memory.py`: tracker peak core memory inti algoritma.
- `benchmark_banded4_given_data.py`: benchmark utama pada data asli ukuran `8 x 8` sampai `512 x 512`.
- `generate_hasil_lu.py`: runner Block LU untuk menghasilkan faktor LU, solusi, dan metrik pada data yang diberikan.
- `Data/`: dataset yang diberikan, yaitu ukuran `8 x 8` sampai `512 x 512`.
- `Hasil/`: ringkasan hasil benchmark, HTML perbandingan, dan file solusi `x_b/x_c`.

## Instalasi

Gunakan Python `3.10+`.

### Opsi 1: CPU Only

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas psutil
```

### Opsi 2: GPU di Windows dengan CUDA Toolkit

Jika ingin memakai jalur GPU untuk Block LU ukuran `512 x 512`, install `CUDA Toolkit 12.x` lalu `CuPy`.

1. Pastikan driver NVIDIA aktif:

```powershell
nvidia-smi
```

2. Install `CUDA Toolkit 12.x` dari NVIDIA.
3. Tutup terminal lama, buka PowerShell baru, lalu verifikasi:

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

Sebagai fallback, CuPy juga menyediakan komponen CUDA lewat:

```powershell
pip install "cupy-cuda12x[ctk]"
```

## Menjalankan Script

### Benchmark utama pada data asli

```powershell
python benchmark_banded4_given_data.py --input-root Data --output-root hasil_banded4_aktual
```

Script ini membaca langsung folder `Data/ukuran 8 x 8` sampai `Data/ukuran 512 x 512`, lalu menghasilkan:

- `hasil_banded4_aktual/ringkasan.csv`
- `hasil_banded4_aktual/ringkasan.json`
- `hasil_banded4_aktual/website_perbandingan_banded4_aktual.html`
- `hasil_banded4_aktual/solusi/*.txt`

### Menjalankan Block LU saja

```powershell
python generate_hasil_lu.py --input-root Data --output-root hasil_block_lu --backend auto --sizes 8 16 32 128 256 512
```

Jika ingin memaksa CPU:

```powershell
python generate_hasil_lu.py --input-root Data --output-root hasil_block_lu --backend cpu --sizes 8 16 32 128 256 512
```

Jika ingin memaksa GPU:

```powershell
python generate_hasil_lu.py --input-root Data --output-root hasil_block_lu --backend gpu --sizes 8 16 32 128 256 512
```

## Lokasi Hasil

Folder `Hasil/` di paket ini sudah berisi artefak final yang dipakai laporan:

- `ringkasan.csv`: rekap utama runtime, memory, CPU usage, GPU usage, condition number, dan error untuk Block LU vs Thomas 4-diagonal.
- `ringkasan.json`: versi JSON dari ringkasan utama.
- `website_perbandingan_banded4_aktual.html`: HTML perbandingan.
- `block_lu_x_b_<size>.txt` dan `block_lu_x_c_<size>.txt`: solusi Block LU.
- `thomas4_x_b_<size>.txt` dan `thomas4_x_c_<size>.txt`: solusi solver 4-diagonal.

Jika ingin langsung melihat hasil solusi dan metrik yang dirujuk laporan, buka `Hasil/ringkasan.csv`.

## Catatan

- `Block LU` pada paket ini masih memakai matriks kerja dense, jadi kompleksitas waktunya tetap `O(n^3)` dan memorinya `O(n^2)`.
- `Thomas 4-diagonal` bekerja langsung pada empat pita 1D, jadi kompleksitas waktunya `O(n)` dan memorinya `O(n)`.
- Angka memory pada benchmark utama adalah `peak core memory`, yaitu peak dari buffer array algoritma yang hidup selama factorization dan solve.

## Referensi Instalasi Resmi

- CuPy installation guide: `https://docs.cupy.dev/en/stable/install.html`
- NVIDIA CUDA Installation Guide for Microsoft Windows: `https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/`
