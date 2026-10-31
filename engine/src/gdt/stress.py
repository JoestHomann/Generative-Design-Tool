"""Stress check of the binary design (reported, not certified).

Margins follow common aerospace practice: yield is checked at limit load, ultimate at
ultimate load (limit x factor of safety). A fitting factor is applied in preserved regions,
which are the part's connections.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import fem
from .setup import Problem

NOTE = ("Not part of the certificate. Element-centroid von Mises stresses from the linear "
        "plane-stress model; stresses at point loads and sharp corners are mesh-dependent.")


def _centroid_b(h: float) -> NDArray[np.float64]:
    dn_dx = np.array([-1, 1, 1, -1]) / (2 * h)
    dn_dy = np.array([-1, -1, 1, 1]) / (2 * h)
    b = np.zeros((3, 8))
    b[0, 0::2] = dn_dx
    b[1, 1::2] = dn_dy
    b[2, 0::2] = dn_dy
    b[2, 1::2] = dn_dx
    return b


def von_mises(problem: Problem, model: fem.Model, u: NDArray[np.float64]) -> NDArray[np.float64]:
    nu, e0 = problem.material.poissons_ratio, problem.material.youngs_modulus_MPa
    d = e0 / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    s = (d @ _centroid_b(problem.h) @ u[model.dofs].T).T  # (n_elements, 3)
    vm: NDArray[np.float64] = np.sqrt(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2
                                      + 3 * s[:, 2] ** 2)
    return vm


def check(problem: Problem, model: fem.Model, design: NDArray[np.bool_],
          sol: fem.Solution) -> dict[str, Any]:
    st = problem.settings
    allow = problem.material.allowables
    warnings = list(problem.material.warnings)
    factor = np.where(problem.preserved, st.fitting_factor, 1.0)
    h = problem.h
    cases = []
    for lc, u in zip(problem.load_cases, sol.displacements):
        vm = np.where(design, von_mises(problem, model, u), 0.0)
        governing = vm * factor
        e = int(np.argmax(governing))
        entry: dict[str, Any] = {
            "name": lc.name,
            "max_von_mises_at_limit_load_MPa": float(vm[e]),
            "location_mm": [((e % problem.nx) + 0.5) * h, ((e // problem.nx) + 0.5) * h],
            "in_preserved_region": bool(problem.preserved[e]),
        }
        if allow is not None and governing[e] > 0:
            entry["ms_yield"] = float(allow.yield_MPa / governing[e] - 1)
            entry["ms_ultimate"] = float(
                allow.ultimate_MPa / (st.factor_of_safety * governing[e]) - 1)
        cases.append(entry)
    out: dict[str, Any] = {
        "certified": False, "note": NOTE,
        "factor_of_safety": st.factor_of_safety, "fitting_factor": st.fitting_factor,
        "allowables": None if allow is None else allow.to_dict(),
        "load_cases": cases, "warnings": warnings,
    }
    if allow is not None and all("ms_yield" in c for c in cases):
        out["min_ms_yield"] = min(c["ms_yield"] for c in cases)
        out["min_ms_ultimate"] = min(c["ms_ultimate"] for c in cases)
        out["passed"] = bool(out["min_ms_yield"] >= 0 and out["min_ms_ultimate"] >= 0)
    else:
        out["passed"] = None
    return out
