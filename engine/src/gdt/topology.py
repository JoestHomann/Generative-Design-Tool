"""Connectivity of element sets: components and support (rigid-body) checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def components(solid: NDArray[np.bool_], nx: int, ny: int) -> NDArray[np.int64]:
    """Label edge-connected groups of solid elements (-1 for empty elements)."""
    label = np.full(nx * ny, -1, dtype=np.int64)
    current = 0
    for start in np.flatnonzero(solid):
        if label[start] >= 0:
            continue
        stack = [int(start)]
        label[start] = current
        while stack:
            e = stack.pop()
            i, j = e % nx, e // nx
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny:
                    n = b * nx + a
                    if solid[n] and label[n] < 0:
                        label[n] = current
                        stack.append(n)
        current += 1
    return label


def nodes_of(elements: NDArray[np.int64], nx: int) -> NDArray[np.int64]:
    nnx = nx + 1
    i, j = elements % nx, elements // nx
    n0 = j * nnx + i
    return np.unique(np.concatenate([n0, n0 + 1, n0 + nnx + 1, n0 + nnx]))


def support_rank(nodes: NDArray[np.int64], fixed: NDArray[np.bool_], nx: int, h: float) -> int:
    """Rank of the rigid-body constraints the fixed dofs of these nodes impose (3 = held)."""
    nnx = nx + 1
    rows = []
    for n in nodes:
        x, y = (n % nnx) * h, (n // nnx) * h
        if fixed[2 * n]:
            rows.append([1.0, 0.0, -y])
        if fixed[2 * n + 1]:
            rows.append([0.0, 1.0, x])
    return int(np.linalg.matrix_rank(np.array(rows))) if rows else 0


def loaded(nodes: NDArray[np.int64], forces: list[NDArray[np.float64]]) -> bool:
    return any(bool(np.any(f[2 * nodes]) or np.any(f[2 * nodes + 1])) for f in forces)
