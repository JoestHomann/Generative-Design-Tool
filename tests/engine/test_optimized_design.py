"""The optimized result is a feasible binary design that respects every constraint."""

from __future__ import annotations

import math

import numpy as np
import pytest

import gdt
from setups import bracket, cantilever, mbb_beam

BENCHMARKS = {
    "cantilever": lambda: cantilever(nx=40, ny=20, budget=0.4),
    "mbb": lambda: mbb_beam(nx=60, ny=20, budget=0.5),
    "bracket": lambda: bracket(nx=48, ny=24, budget=0.35),
}


@pytest.fixture(scope="module", params=sorted(BENCHMARKS))
def solved(request: pytest.FixtureRequest) -> tuple[dict, gdt.Result]:
    setup = BENCHMARKS[request.param]()
    return setup, gdt.optimize(setup)


def element_centroids(setup: dict) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = setup["mesh"]["nx"], setup["mesh"]["ny"]
    h = setup["envelope"]["width_mm"] / nx
    x, y = np.meshgrid((np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h)
    return x.ravel(), y.ravel()


def test_design_is_binary_with_one_entry_per_element(solved: tuple[dict, gdt.Result]) -> None:
    setup, result = solved
    assert result.design.dtype == bool
    assert result.design.size == setup["mesh"]["nx"] * setup["mesh"]["ny"]


def test_design_uses_no_more_than_the_material_budget(solved: tuple[dict, gdt.Result]) -> None:
    setup, result = solved
    n = result.design.size
    assert result.design.sum() <= math.floor(setup["material_budget"] * n + 1e-9)
    # and it uses the material: a stiff design does not leave most of the budget unused
    assert result.design.sum() >= 0.9 * setup["material_budget"] * n


def test_preserved_regions_are_solid_and_keep_out_regions_empty(
        solved: tuple[dict, gdt.Result]) -> None:
    setup, result = solved
    x, y = element_centroids(setup)
    for r in setup.get("preserved_regions", []):
        inside = ((x >= r["x_mm"]) & (x <= r["x_mm"] + r["width_mm"])
                  & (y >= r["y_mm"]) & (y <= r["y_mm"] + r["height_mm"]))
        assert result.design[inside].all()
    for r in setup.get("keep_out_regions", []):
        inside = (x - r["cx_mm"]) ** 2 + (y - r["cy_mm"]) ** 2 <= r["r_mm"] ** 2
        assert not result.design[inside].any()


def test_every_load_is_carried(solved: tuple[dict, gdt.Result]) -> None:
    _, result = solved
    bounds = result.certificate["bounds"]
    assert bounds["status"] == "certified"
    assert math.isfinite(bounds["upper_bound_Nmm"]) and bounds["upper_bound_Nmm"] > 0


def test_optimized_design_is_stiffer_than_a_uniform_strip() -> None:
    """Sanity: at 40 % material the optimum beats a plain horizontal band through the load."""
    setup = cantilever(nx=40, ny=20, budget=0.4)
    band = dict(setup)
    band["material_budget"] = 1.0
    band["keep_out_regions"] = [
        {"shape": "rect", "x_mm": 0.0, "y_mm": 0.0, "width_mm": 80.0, "height_mm": 12.0},
        {"shape": "rect", "x_mm": 0.0, "y_mm": 28.0, "width_mm": 80.0, "height_mm": 12.0},
    ]
    optimized = gdt.optimize(setup).certificate["bounds"]["upper_bound_Nmm"]
    strip = gdt.optimize(band).certificate["bounds"]["upper_bound_Nmm"]
    assert optimized < 0.5 * strip


def test_same_setup_gives_identical_result() -> None:
    a = gdt.optimize(cantilever(nx=30, ny=15))
    b = gdt.optimize(cantilever(nx=30, ny=15))
    assert np.array_equal(a.design, b.design)
    assert a.certificate["bounds"] == b.certificate["bounds"]
