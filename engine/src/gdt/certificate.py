"""Builds the certificate: everything needed to re-check the claims from one file."""

from __future__ import annotations

import base64
import datetime as dt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._version import __version__ as _version
from . import bounds, fem
from .binarize import Binary
from .optimizer import OptimizerResult
from .setup import Problem

SCHEMA_VERSION = 1
TARGET_GAP = 0.05
STRETCH_GAP = 0.02
MIN_MARGIN = 1e-9  # relative floating-point margin applied to both bounds
RESIDUAL_FACTOR = 10.0

ASSUMPTIONS = [
    "2D plane stress, linear elastic, isotropic material, small deformations",
    "structured mesh of square bilinear (Q4) elements, 2x2 Gauss integration",
    "empty elements have zero stiffness; the bound holds for every binary design on this mesh "
    "with the stated supports, load cases and material budget",
    "loads are limit loads; the stress check (not certified) scales them to ultimate loads",
]


def encode_design(design: NDArray[np.bool_]) -> str:
    return base64.b64encode(np.packbits(design.astype(np.uint8)).tobytes()).decode("ascii")


def encode_floats(x: NDArray[np.float64]) -> str:
    return base64.b64encode(x.astype("<f8").tobytes()).decode("ascii")


def build(problem: Problem, model: fem.Model, binary: Binary,
          opt: OptimizerResult | None) -> dict[str, Any]:
    from . import stress  # local import keeps module dependencies one-directional

    design = binary.design
    n_solid = int(design.sum())
    h = problem.h
    mass = n_solid * h * h * problem.thickness * 1e-9 * problem.material.density_kg_m3
    cert: dict[str, Any] = {
        "schema": "gdt-certificate",
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "gdt-engine", "version": _version},
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "setup": problem.raw,
        "material": _material(problem),
        "model": {
            "assumptions": ASSUMPTIONS,
            "nx": problem.nx, "ny": problem.ny, "element_size_mm": h,
            "element_order": "index = ey * nx + ex, ey = 0 is the bottom row",
            "node_order": "index = j * (nx + 1) + i; dofs 2k (x), 2k + 1 (y)",
        },
        "design": {
            "encoding": "numpy.packbits of the element array (big-endian bit order), base64",
            "data": encode_design(design),
            "solid_elements": n_solid,
            "max_solid_elements": problem.max_solid,
            "volume_fraction": n_solid / problem.n_elements,
            "mass_kg": mass,
            "removed_islands": binary.removed_islands,
        },
        "optimizer": None if opt is None else {
            "method": "SIMP (p = 3), density filter, Heaviside projection, Optimality Criteria",
            "iterations": opt.iterations, "converged": opt.converged,
            "interrupted": opt.interrupted, "kkt_residual": opt.kkt_residual,
            "final_continuous_compliance_Nmm": opt.compliance,
        },
        "numerics": {
            "precision": "IEEE 754 binary64 (double precision)",
            "linear_solver": "sparse direct LU (SciPy SuperLU)",
        },
    }
    if not binary.feasible:
        cert["bounds"] = {"status": "infeasible", "issues": binary.issues}
        cert["stress_check"] = None
        return cert

    sol = model.solve(design.astype(float) * problem.material.youngs_modulus_MPa)
    ub = model.compliance(sol)
    lb = bounds.lower_bound(problem, model)
    worst_residual = max(sol.max_relative_residual, lb.max_relative_residual)
    margin = max(MIN_MARGIN, RESIDUAL_FACTOR * worst_residual)
    cert["numerics"].update({
        "upper_bound_relative_residual": sol.max_relative_residual,
        "lower_bound_relative_residual": lb.max_relative_residual,
        "relative_margin": margin,
        "margin_rule": f"max({MIN_MARGIN:g}, {RESIDUAL_FACTOR:g} x worst relative residual), "
                       "added to the upper bound and subtracted from the lower bound",
    })
    cert["relaxed_design"] = {
        "encoding": "float64 little-endian, base64",
        "data": encode_floats(lb.relaxed_design),
        "relaxed_compliance_Nmm": lb.relaxed_compliance,
        "iterations": lb.iterations,
    }
    if worst_residual > fem.MAX_RELATIVE_RESIDUAL:
        cert["bounds"] = {"status": "no bound established", "upper_bound_Nmm": ub,
                          "reason": f"a linear solve reached only a relative residual of "
                                    f"{worst_residual:.1e}; at most "
                                    f"{fem.MAX_RELATIVE_RESIDUAL:.0e} is required"}
    elif not np.isfinite(lb.value) or lb.value <= 0:
        cert["bounds"] = {"status": "no bound established", "upper_bound_Nmm": ub,
                          "reason": "the relaxed problem gave no positive lower bound"}
    else:
        ub_adj = ub * (1 + margin)
        lb_adj = lb.value * (1 - margin)
        gap = (ub_adj - lb_adj) / lb_adj
        cert["bounds"] = {
            "status": "certified",
            "upper_bound_Nmm": ub, "lower_bound_Nmm": lb.value,
            "upper_bound_adjusted_Nmm": ub_adj, "lower_bound_adjusted_Nmm": lb_adj,
            "gap": gap, "gap_percent": 100 * gap,
            "target": {"gap_max": TARGET_GAP, "met": bool(gap <= TARGET_GAP),
                       "stretch_gap_max": STRETCH_GAP, "stretch_met": bool(gap <= STRETCH_GAP)},
            "statement": (f"No binary design on this {problem.nx} x {problem.ny} mesh with these "
                          f"supports, load cases and material budget has a compliance below "
                          f"{lb_adj:.6g} N mm; this design's compliance is {ub:.6g} N mm, "
                          f"at most {100 * gap:.2f} % above the best possible."),
        }
    cert["stress_check"] = stress.check(problem, model, design, sol)
    return cert


def _material(problem: Problem) -> dict[str, Any]:
    m = problem.material
    a = m.allowables
    return {
        "id": m.id, "name": m.name,
        "youngs_modulus_MPa": m.youngs_modulus_MPa, "poissons_ratio": m.poissons_ratio,
        "density_kg_m3": m.density_kg_m3,
        "allowables": None if a is None else a.to_dict(),
        "sources": m.sources, "warnings": list(m.warnings),
    }
