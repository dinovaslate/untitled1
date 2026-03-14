#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
DIAG_VALUE = 4
OFF_DIAG_VALUE = -1


def row_bytes(n: int, i: int) -> bytes:
    if n == 1:
        return f"{DIAG_VALUE}\n".encode("ascii")

    if i == 0:
        prefix_zeros = 0
        middle = f"{DIAG_VALUE},{OFF_DIAG_VALUE}".encode("ascii")
        suffix_zeros = n - 2
    elif i == n - 1:
        prefix_zeros = n - 2
        middle = f"{OFF_DIAG_VALUE},{DIAG_VALUE}".encode("ascii")
        suffix_zeros = 0
    else:
        prefix_zeros = i - 1
        middle = f"{OFF_DIAG_VALUE},{DIAG_VALUE},{OFF_DIAG_VALUE}".encode("ascii")
        suffix_zeros = n - i - 2

    chunks: list[bytes] = []
    if prefix_zeros > 0:
        chunks.append(b"0," * prefix_zeros)
    chunks.append(middle)
    if suffix_zeros > 0:
        chunks.append(b",")
        if suffix_zeros > 1:
            chunks.append(b"0," * (suffix_zeros - 1))
        chunks.append(b"0")
    chunks.append(b"\n")
    return b"".join(chunks)


def write_matrix_csv(path: Path, n: int) -> None:
    with path.open("wb", buffering=1024 * 1024) as handle:
        for i in range(n):
            handle.write(row_bytes(n, i))


def b_value(n: int, i: int) -> int:
    if n == 1:
        return DIAG_VALUE
    if i == 0 or i == n - 1:
        return DIAG_VALUE + OFF_DIAG_VALUE
    return DIAG_VALUE + 2 * OFF_DIAG_VALUE


def c_value(n: int, i: int) -> int:
    if n == 1:
        return DIAG_VALUE
    if i == 0:
        return DIAG_VALUE * 1 + OFF_DIAG_VALUE * 2
    if i == n - 1:
        return OFF_DIAG_VALUE * (n - 1) + DIAG_VALUE * n
    x_i = i + 1
    return OFF_DIAG_VALUE * i + DIAG_VALUE * x_i + OFF_DIAG_VALUE * (i + 2)


def write_vector_csv(path: Path, n: int, value_fn) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        for i in range(n):
            handle.write(f"{value_fn(n, i)}\n")


def generate_dataset(root: Path, n: int, overwrite: bool) -> None:
    folder = root / f"Ukuran {n} x {n}"
    folder.mkdir(parents=True, exist_ok=True)

    targets = [
        folder / f"A_{n}.csv",
        folder / f"b_{n}.csv",
        folder / f"c_{n}.csv",
    ]

    if not overwrite and all(path.exists() for path in targets):
        print(f"[SKIP] {n}x{n} already exists")
        return

    print(f"[WRITE] {n}x{n} -> {folder}")
    write_matrix_csv(folder / f"A_{n}.csv", n)
    write_vector_csv(folder / f"b_{n}.csv", n, b_value)
    write_vector_csv(folder / f"c_{n}.csv", n, c_value)
    print(f"[DONE]  {n}x{n}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dense CSV tridiagonal datasets with the same A/b/c layout as the existing folders."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root folder where 'Ukuran N x N' directories will be created.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=DEFAULT_SIZES,
        help="Matrix sizes to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    for n in args.sizes:
        generate_dataset(root=root, n=n, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
