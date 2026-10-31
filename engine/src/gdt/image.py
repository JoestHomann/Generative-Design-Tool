"""Renders the binary design as a PNG (no third-party imaging library needed)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .setup import Problem

SOLID = (24, 33, 43)
EMPTY = (255, 255, 255)
PRESERVED = (29, 106, 134)
KEEP_OUT = (246, 225, 225)
SUPPORT = (90, 101, 115)
LOAD = (210, 70, 27)


def write_png(path: Path, rgb: NDArray[np.uint8]) -> None:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def render(problem: Problem, design: NDArray[np.bool_]) -> NDArray[np.uint8]:
    nx, ny = problem.nx, problem.ny
    s = int(max(2, min(16, 1200 // max(nx, 1))))
    pad = 3 * s
    img = np.full((ny * s + 2 * pad, nx * s + 2 * pad, 3), 250, dtype=np.uint8)
    colour = np.empty((ny * nx, 3), dtype=np.uint8)
    colour[:] = EMPTY
    colour[problem.keep_out] = KEEP_OUT
    colour[design] = SOLID
    colour[design & problem.preserved] = PRESERVED
    grid = colour.reshape(ny, nx, 3)[::-1]  # top row first
    img[pad:pad + ny * s, pad:pad + nx * s] = np.repeat(np.repeat(grid, s, axis=0), s, axis=1)

    def node_px(node: int) -> tuple[int, int]:
        i, j = node % (nx + 1), node // (nx + 1)
        return pad + i * s, pad + (ny - j) * s

    r = max(1, s // 2)
    for node in np.flatnonzero(problem.fixed[0::2] | problem.fixed[1::2]):
        x, y = node_px(int(node))
        img[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1] = SUPPORT
    for lc in problem.load_cases:
        f = lc.force.reshape(-1, 2)
        for node in np.flatnonzero(np.any(f != 0, axis=1)):
            x, y = node_px(int(node))
            d = f[node] / np.linalg.norm(f[node])
            for t in np.linspace(0, 2.5 * s, 4 * s):
                px, py = int(round(x - d[0] * t)), int(round(y + d[1] * t))
                img[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2] = LOAD
            img[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1] = LOAD
    return img
