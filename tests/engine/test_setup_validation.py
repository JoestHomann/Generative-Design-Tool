"""The engine rejects setups that cannot be solved, naming each problem."""

import pytest

import gdt
from setups import cantilever, with_changes


def issues_of(setup: dict) -> list[str]:
    with pytest.raises(gdt.SetupError) as err:
        gdt.optimize(setup)
    return err.value.issues


def test_setup_without_supports_is_rejected() -> None:
    issues = issues_of(with_changes(cantilever(), supports=[]))
    assert any("support" in i.lower() for i in issues)


def test_load_inside_keep_out_region_is_rejected() -> None:
    s = cantilever()
    s["keep_out_regions"] = [{"shape": "rect", "x_mm": 70.0, "y_mm": 10.0,
                              "width_mm": 10.0, "height_mm": 20.0}]
    issues = issues_of(s)
    assert any("load_cases[0].loads[0]" in i and "keep-out" in i for i in issues)


def test_budget_smaller_than_preserved_regions_is_rejected() -> None:
    s = cantilever(budget=0.1)
    s["preserved_regions"] = [{"shape": "rect", "x_mm": 0.0, "y_mm": 0.0,
                               "width_mm": 40.0, "height_mm": 40.0}]
    issues = issues_of(s)
    assert any("material_budget" in i and "preserved" in i for i in issues)


def test_unknown_field_is_rejected_by_name() -> None:
    s = cantilever()
    s["envelope"]["width_m"] = 0.08
    assert any("envelope.width_m" in i and "unknown" in i for i in issues_of(s))


def test_negative_dimension_is_rejected_with_unit() -> None:
    s = cantilever()
    s["envelope"]["thickness_mm"] = -5.0
    assert any("envelope.thickness_mm" in i and "mm" in i for i in issues_of(s))


def test_unknown_material_is_rejected() -> None:
    s = with_changes(cantilever(), material={"id": "unobtainium"})
    assert any("unobtainium" in i for i in issues_of(s))


def test_supports_that_allow_rigid_body_rotation_are_rejected() -> None:
    # Only one node fixed in x and y: the part can still rotate about it.
    s = cantilever()
    s["supports"] = [{"x_mm": 0.0, "y_mm": 0.0, "width_mm": 0.0, "height_mm": 0.0, "fix": "xy"}]
    assert any("rigid-body" in i for i in issues_of(s))


def test_all_problems_are_reported_at_once() -> None:
    s = with_changes(cantilever(), supports=[], material_budget=1.5)
    issues = issues_of(s)
    assert any("material_budget" in i for i in issues)
    assert any(i.startswith("supports") for i in issues)


def test_unknown_material_field_is_rejected() -> None:
    s = with_changes(cantilever(), material={"id": "ti-6al-4v", "temper": "STA"})
    assert any("material.temper" in i and "unknown" in i for i in issues_of(s))


def test_support_entirely_inside_keep_out_region_is_rejected() -> None:
    s = cantilever()
    s["keep_out_regions"] = [{"shape": "rect", "x_mm": 0.0, "y_mm": 0.0,
                              "width_mm": 4.0, "height_mm": 40.0}]
    assert any("supports[0]" in i and "keep-out" in i for i in issues_of(s))
