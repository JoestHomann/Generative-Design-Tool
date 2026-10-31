"""Problem setups used across the engine tests (plain setup-file dictionaries)."""

from __future__ import annotations

import copy
from typing import Any

Setup = dict[str, Any]


def tension_bar(nx: int = 20, ny: int = 4, length: float = 100.0, height: float = 20.0,
                thickness: float = 2.0, force: float = 1000.0) -> Setup:
    """Plate pulled by a uniform traction on its right edge.

    Left edge fixed in x, bottom-left node also fixed in y. Fully solid
    (budget 1.0), so the result is the plain elastic solution.
    """
    return {
        "schema_version": 1,
        "envelope": {"width_mm": length, "height_mm": height, "thickness_mm": thickness},
        "mesh": {"nx": nx, "ny": ny},
        "material": {"id": "al-7075-t6"},
        "material_budget": 1.0,
        "min_member_size_mm": 0.0,
        "preserved_regions": [],
        "keep_out_regions": [],
        "supports": [
            {"x_mm": 0.0, "y_mm": 0.0, "width_mm": 0.0, "height_mm": height, "fix": "x"},
            {"x_mm": 0.0, "y_mm": 0.0, "width_mm": 0.0, "height_mm": 0.0, "fix": "y"},
        ],
        "load_cases": [
            {"name": "tension", "weight": 1.0, "loads": [
                {"type": "edge", "edge": "right", "from_mm": 0.0, "to_mm": height,
                 "fx_N": force, "fy_N": 0.0},
            ]},
        ],
    }


def cantilever(nx: int = 40, ny: int = 20, width: float = 80.0, height: float = 40.0,
               budget: float = 0.4, force: float = 1000.0) -> Setup:
    """Left edge clamped, downward point load at the middle of the right edge."""
    return {
        "schema_version": 1,
        "envelope": {"width_mm": width, "height_mm": height, "thickness_mm": 5.0},
        "mesh": {"nx": nx, "ny": ny},
        "material": {"id": "ti-6al-4v"},
        "material_budget": budget,
        "min_member_size_mm": 2.0 * width / nx,
        "preserved_regions": [],
        "keep_out_regions": [],
        "supports": [
            {"x_mm": 0.0, "y_mm": 0.0, "width_mm": 0.0, "height_mm": height, "fix": "xy"},
        ],
        "load_cases": [
            {"name": "tip", "weight": 1.0, "loads": [
                {"type": "point", "x_mm": width, "y_mm": height / 2, "fx_N": 0.0, "fy_N": -force},
            ]},
        ],
    }


def mbb_beam(nx: int = 60, ny: int = 20, budget: float = 0.5) -> Setup:
    """Half MBB beam: symmetry line on the left, roller bottom right, load top left."""
    width, height = 3.0 * nx, 3.0 * ny
    return {
        "schema_version": 1,
        "envelope": {"width_mm": width, "height_mm": height, "thickness_mm": 3.0},
        "mesh": {"nx": nx, "ny": ny},
        "material": {"id": "al-2024-t3"},
        "material_budget": budget,
        "min_member_size_mm": 6.0,
        "preserved_regions": [],
        "keep_out_regions": [],
        "supports": [
            {"x_mm": 0.0, "y_mm": 0.0, "width_mm": 0.0, "height_mm": height, "fix": "x"},
            {"x_mm": width, "y_mm": 0.0, "width_mm": 0.0, "height_mm": 0.0, "fix": "y"},
        ],
        "load_cases": [
            {"name": "centre load", "weight": 1.0, "loads": [
                {"type": "point", "x_mm": 0.0, "y_mm": height, "fx_N": 0.0, "fy_N": -2000.0},
            ]},
        ],
    }


def bracket(nx: int = 60, ny: int = 30, budget: float = 0.35) -> Setup:
    """Wall bracket: two bolt pads on the left (preserved), bolt hole keep-out,
    load pad on the right, two load cases."""
    width, height = 2.0 * nx, 2.0 * ny
    pad = height / 6
    return {
        "schema_version": 1,
        "envelope": {"width_mm": width, "height_mm": height, "thickness_mm": 4.0},
        "mesh": {"nx": nx, "ny": ny},
        "material": {"id": "ti-6al-4v"},
        "material_budget": budget,
        "min_member_size_mm": 4.0,
        "preserved_regions": [
            {"shape": "rect", "x_mm": 0.0, "y_mm": 0.0, "width_mm": pad, "height_mm": pad},
            {"shape": "rect", "x_mm": 0.0, "y_mm": height - pad, "width_mm": pad, "height_mm": pad},
            {"shape": "rect", "x_mm": width - pad, "y_mm": height / 2 - pad / 2,
             "width_mm": pad, "height_mm": pad},
        ],
        "keep_out_regions": [
            {"shape": "circle", "cx_mm": width / 2, "cy_mm": height / 2, "r_mm": height / 8},
        ],
        "supports": [
            {"x_mm": 0.0, "y_mm": 0.0, "width_mm": 0.0, "height_mm": pad, "fix": "xy"},
            {"x_mm": 0.0, "y_mm": height - pad, "width_mm": 0.0, "height_mm": pad, "fix": "xy"},
        ],
        "load_cases": [
            {"name": "down", "weight": 1.0, "loads": [
                {"type": "point", "x_mm": width, "y_mm": height / 2, "fx_N": 0.0, "fy_N": -3000.0},
            ]},
            {"name": "outward", "weight": 0.5, "loads": [
                {"type": "point", "x_mm": width, "y_mm": height / 2, "fx_N": 2000.0, "fy_N": 0.0},
            ]},
        ],
    }


def with_changes(setup: Setup, **changes: Any) -> Setup:
    """Deep copy of a setup with top-level keys replaced."""
    new = copy.deepcopy(setup)
    new.update(changes)
    return new
