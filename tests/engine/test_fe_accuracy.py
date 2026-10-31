"""With the material budget at 100 % every element is solid, so the certified upper bound
is the plain elastic compliance and can be compared with analytical solutions."""

import pytest

import gdt
from setups import tension_bar, with_changes

# MIL-HDBK-5J, Table 3.7.6.0(b): 7075-T6, E = 10.3e3 ksi; mu = 0.33
E_7075 = 10.3e3 * 6.894757  # MPa
NU_7075 = 0.33
# Ti-6Al-4V annealed, Table 5.4.1.0(b): E = 16.0e3 ksi, mu = 0.31
E_TI = 16.0e3 * 6.894757
NU_TI = 0.31


def upper_bound(setup: dict) -> float:
    return float(gdt.optimize(setup).certificate["bounds"]["upper_bound_Nmm"])


def test_uniform_tension_bar_matches_exact_compliance() -> None:
    F, L, H, t = 1000.0, 100.0, 20.0, 2.0
    exact = F ** 2 * L / (E_7075 * H * t)  # c = F * delta, delta = F L / (E A)
    assert upper_bound(tension_bar(force=F, length=L, height=H, thickness=t)) == \
        pytest.approx(exact, rel=1e-9)


def cantilever_full(n_through: int) -> dict:
    L, H, t, P = 160.0, 20.0, 1.0, 100.0
    return {
        "schema_version": 1,
        "envelope": {"width_mm": L, "height_mm": H, "thickness_mm": t},
        "mesh": {"nx": 8 * n_through, "ny": n_through},
        "material": {"id": "ti-6al-4v"},
        "material_budget": 1.0,
        "min_member_size_mm": 0.0,
        "supports": [{"x_mm": 0.0, "y_mm": 0.0, "width_mm": 0.0, "height_mm": H, "fix": "xy"}],
        "load_cases": [{"name": "tip shear", "weight": 1.0, "loads": [
            {"type": "edge", "edge": "right", "from_mm": 0.0, "to_mm": H,
             "fx_N": 0.0, "fy_N": -P}]}],
    }


def timoshenko_compliance() -> float:
    L, H, t, P = 160.0, 20.0, 1.0, 100.0
    inertia = t * H ** 3 / 12
    shear_modulus = E_TI / (2 * (1 + NU_TI))
    tip = P * L ** 3 / (3 * E_TI * inertia) + P * L / ((5 / 6) * shear_modulus * H * t)
    return P * tip


def test_cantilever_converges_to_beam_theory_as_mesh_is_refined() -> None:
    reference = timoshenko_compliance()
    errors = [abs(upper_bound(cantilever_full(n)) - reference) / reference for n in (2, 4, 8)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.03


def test_material_stiffness_scales_compliance() -> None:
    """Doubling Young's modulus halves the compliance (custom material)."""
    base = tension_bar()
    soft = with_changes(base, material={"custom": {
        "name": "soft", "youngs_modulus_GPa": 70.0, "poissons_ratio": 0.33,
        "density_kg_m3": 2700.0}})
    stiff = with_changes(base, material={"custom": {
        "name": "stiff", "youngs_modulus_GPa": 140.0, "poissons_ratio": 0.33,
        "density_kg_m3": 2700.0}})
    assert upper_bound(soft) == pytest.approx(2 * upper_bound(stiff), rel=1e-9)
