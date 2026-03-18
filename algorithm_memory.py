from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _is_array_like(obj: Any) -> bool:
    return (
        hasattr(obj, "shape")
        and hasattr(obj, "dtype")
        and hasattr(obj, "nbytes")
    )


def _iter_array_like(obj: Any):
    if obj is None:
        return
    if _is_array_like(obj):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_array_like(value)
        return
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _iter_array_like(value)


def _root_owner(array: Any) -> Any:
    root = array
    seen: set[int] = set()
    while True:
        base = getattr(root, "base", None)
        if base is None:
            return root
        base_id = id(base)
        if base_id in seen:
            return root
        seen.add(base_id)
        if _is_array_like(base):
            root = base
            continue
        return root


def _buffer_signature_and_size(array: Any) -> tuple[tuple[Any, ...], int]:
    root = _root_owner(array)
    size = int(getattr(root, "nbytes"))

    cuda_interface = getattr(root, "__cuda_array_interface__", None)
    if isinstance(cuda_interface, dict):
        data = cuda_interface.get("data")
        if isinstance(data, tuple) and data and data[0] is not None:
            return ("cuda", int(data[0]), size), size

    array_interface = getattr(root, "__array_interface__", None)
    if isinstance(array_interface, dict):
        data = array_interface.get("data")
        if isinstance(data, tuple) and data and data[0] is not None:
            return ("cpu", int(data[0]), size), size

    data_attr = getattr(root, "data", None)
    ptr = getattr(data_attr, "ptr", None)
    if ptr is not None:
        return ("ptr", int(ptr), size), size

    return ("obj", id(root), size), size


@dataclass
class AlgorithmMemoryTracker:
    peak_bytes: int = 0
    snapshots: list[tuple[str, int]] = field(default_factory=list)

    def snapshot(self, *objects: Any, label: str | None = None) -> int:
        seen: set[tuple[Any, ...]] = set()
        total = 0
        for obj in objects:
            for array in _iter_array_like(obj):
                signature, size = _buffer_signature_and_size(array)
                if signature in seen:
                    continue
                seen.add(signature)
                total += size
        self.peak_bytes = max(self.peak_bytes, total)
        if label is not None:
            self.snapshots.append((label, total))
        return total

    @property
    def peak_megabytes(self) -> float:
        return self.peak_bytes / (1024.0 * 1024.0)


def snapshot_memory(
    tracker: AlgorithmMemoryTracker | None,
    *objects: Any,
    label: str | None = None,
) -> None:
    if tracker is None:
        return
    tracker.snapshot(*objects, label=label)


__all__ = [
    "AlgorithmMemoryTracker",
    "snapshot_memory",
]
