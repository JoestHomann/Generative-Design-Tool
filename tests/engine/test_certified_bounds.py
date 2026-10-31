"""The certificate's bounds bracket the true optimum, and the gap is honest."""

from __future__ import annotations

import itertools
import math

import pytest

import gdt
from setups import cantilever, mbb_beam, tension_bar


def tiny_cantilever(budget: float) -> dict:
    """6 x 2 elements of 10 mm: clamped left edge, tip load at the bottom-right corner."""
    return {
        "schema_version": 1,
        "envelope": {"width_mm": 60.0, "height_mm": 20.0, "thickness_mm": 1.0},
        "mesh": {"nx": 6, "ny": 2},
        "material": {"id": "al-7075-t6"},
        "material_budget": budget,
        "min_member_size_mm": 0.0,
        "supports": [{"x_mm": 0.0, "y_mm": 0.0, "width_mm": 0.0, "height_mm": 20.0, "fix": "xy"}],
        "load_cases": [{"name": "tip", "weight": 1.0, "loads": [
            {"type": "point", "x_mm": 60.0, "y_mm": 0.0, "fx_N": 0.0, "fy_N": -100.0}]}],
    }


def compliance_of(design_elements: set[int]) -> float:
    """Compliance of one binary design, evaluated through the engine: every element not in
    the design becomes a keep-out region and the budget is 100 %."""
    s = tiny_cantilever(1.0)
    s["keep_out_regions"] = [
        {"shape": "rect", "x_mm": 10.0 * (e % 6) + 4, "y_mm": 10.0 * (e // 6) + 4,
         "width_mm": 2.0, "height_mm": 2.0}
        for e in range(12) if e not in design_elements]
    try:
        cert = gdt.optimize(s).certificate
    except gdt.SetupError:
        return math.inf  # load not held: not a valid design
    b = cert["bounds"]
    return b["upper_bound_Nmm"] if b["status"] == "certified" else math.inf


@pytest.fixture(scope="module")
def true_optimum() -> float:
    # Adding material never makes a part more flexible, so the optimum uses the full budget.
    return min(compliance_of(set(c)) for c in itertools.combinations(range(12), 6))


def test_bounds_bracket_the_enumerated_optimum(true_optimum: float) -> None:
    b = gdt.optimize(tiny_cantilever(0.5)).certificate["bounds"]
    assert b["status"] == "certified"
    assert b["lower_bound_Nmm"] <= true_optimum <= b["upper_bound_Nmm"]


def test_reported_gap_covers_the_true_distance(true_optimum: float) -> None:
    b = gdt.optimize(tiny_cantilever(0.5)).certificate["bounds"]
    true_gap = (b["upper_bound_Nmm"] - true_optimum) / true_optimum
    assert b["gap"] >= true_gap


def test_full_budget_gives_zero_gap() -> None:
    """With 100 % budget the solid envelope is optimal, so the bounds coincide."""
    b = gdt.optimize(tension_bar()).certificate["bounds"]
    assert b["gap"] == pytest.approx(0.0, abs=1e-6)
    assert b["target"]["met"] and b["target"]["stretch_met"]


@pytest.mark.parametrize("setup", [cantilever(nx=40, ny=20), mbb_beam(nx=60, ny=20)],
                         ids=["cantilever", "mbb"])
def test_benchmark_bounds_are_ordered_and_gap_is_reported(setup: dict) -> None:
    b = gdt.optimize(setup).certificate["bounds"]
    assert b["status"] == "certified"
    assert 0 < b["lower_bound_Nmm"] <= b["upper_bound_Nmm"]
    assert b["lower_bound_adjusted_Nmm"] <= b["lower_bound_Nmm"]
    assert b["upper_bound_adjusted_Nmm"] >= b["upper_bound_Nmm"]
    expected = (b["upper_bound_adjusted_Nmm"] - b["lower_bound_adjusted_Nmm"]) \
        / b["lower_bound_adjusted_Nmm"]
    assert b["gap"] == pytest.approx(expected, rel=1e-12)
    assert b["target"]["met"] == (b["gap"] <= 0.05)
    assert b["target"]["stretch_met"] == (b["gap"] <= 0.02)
