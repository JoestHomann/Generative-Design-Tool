"""The stress check reports margins of safety against the material allowables."""

import pytest

import gdt
from setups import tension_bar, with_changes

KSI = 6.894757
# MIL-HDBK-5J Table 3.7.6.0(b), 7075-T651 plate 0.500-1.000 in, A-basis, lower of L / LT
FTY_7075 = 68 * KSI
FTU_7075 = 77 * KSI
STRESS = 1000.0 / (20.0 * 2.0)  # uniform tension bar: F / (H t) = 25 MPa


def stress_check(setup: dict) -> dict:
    check = gdt.optimize(setup).certificate["stress_check"]
    assert check["certified"] is False
    return check


def test_tension_bar_margins_match_hand_calculation() -> None:
    check = stress_check(tension_bar())
    case = check["load_cases"][0]
    assert case["max_von_mises_at_limit_load_MPa"] == pytest.approx(STRESS, rel=1e-6)
    assert check["min_ms_yield"] == pytest.approx(FTY_7075 / STRESS - 1, rel=1e-6)
    assert check["min_ms_ultimate"] == pytest.approx(FTU_7075 / (1.5 * STRESS) - 1, rel=1e-6)
    assert check["passed"] is True


def test_fitting_factor_applies_in_preserved_regions() -> None:
    s = tension_bar()
    s["preserved_regions"] = [{"shape": "rect", "x_mm": 0.0, "y_mm": 0.0,
                               "width_mm": 100.0, "height_mm": 20.0}]
    check = stress_check(s)
    assert check["min_ms_yield"] == pytest.approx(FTY_7075 / (1.15 * STRESS) - 1, rel=1e-6)


def test_overloaded_part_fails_the_check() -> None:
    s = tension_bar(force=1000.0 * 30)  # 750 MPa > 7075 allowables
    check = stress_check(s)
    assert check["passed"] is False and check["min_ms_ultimate"] < 0


def test_material_without_allowables_warns() -> None:
    check = stress_check(with_changes(tension_bar(), material={"id": "alsi10mg-am"}))
    assert any("no design allowables" in w for w in check["warnings"])
    assert check["allowables"]["is_design_allowable"] is False
