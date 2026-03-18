# README Soal 2

## Isi Folder

- `pemodelan_lintasan_bintang.m`: script MATLAB utama untuk formulasi model, penyelesaian dengan `Normal Equation` dan `QR Householder`, analisis kompleksitas, serta ekstraksi parameter geometri elips.
- `Data/data.csv`: 300 titik observasi yang dipakai sebagai input.
- `Hasil/plot_ellipse_fit.png`: visualisasi hasil fitting elips.
- `Hasil/plot_comparison.png`: perbandingan hasil `Normal Equation` dan `QR Householder`.

## Menjalankan Script

Script ini ditulis untuk dijalankan dari folder `Soal 2`, karena path data dan output dibuat relatif.

Di MATLAB:

```matlab
cd('Soal 2')
pemodelan_lintasan_bintang
```

Script akan:

- membaca `Data/data.csv`
- menghitung solusi dengan `Normal Equation`
- menghitung solusi dengan `QR Householder`
- mencetak analisis residual, condition number, FLOPs, dan parameter geometri elips
- menyimpan plot ke folder `Hasil`

## Catatan

- `QR Householder` dipakai sebagai metode yang lebih stabil secara numerik.
- Plot yang disertakan di folder `Hasil` adalah artefak yang dipakai pada laporan final.
