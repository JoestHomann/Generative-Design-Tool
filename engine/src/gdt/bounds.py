"""Proven lower bound on the compliance of every binary design on the mesh.

Why it holds. For stiffness linear in density, the weighted compliance is

    c(y) = sum_k w_k max_u ( 2 f_k^T u - u^T K(y) u ),

a maximum of functions that are affine in y. So for ANY displacement fields u_k (zero on
supported dofs) and every design y,

    c(y) >= sum_k w_k ( 2 f_k^T u_k - sum_e y_e E0 u_k,e^T k0 u_k,e ).

Minimizing the right-hand side over the relaxed feasible set S (densities in [0, 1],
preserved = 1, keep-out = 0, at most max_solid elements' worth of material) gives a bound
that every binary design in S satisfies. Because S is a box with one budget constraint and
the coefficients are >= 0, the minimum is found exactly by filling the budget with the
elements of largest energy.

The bound is valid for any u_k; good u_k come from solving the relaxed problem, which this
module does with Optimality Criteria on the convex problem (stiffness linear in density).
When u_k solves K(x)u_k = f_k for a relaxed design x, the bound equals the Frank-Wolfe
bound at x and tends to the relaxed optimum as x converges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import fem, topology
from .setup import Problem

FloatArray = NDArray[np.float64]

X_MIN = 1e-6
MOVE = 0.5
MAX_ITERATIONS = 400
REL_TOL = 1e-3


@dataclass
class LowerBound:
    value: float
    relaxed_design: FloatArray  # the x whose displacement fields give the bound
    relaxed_compliance: float
    iterations: int
    max_relative_residual: float


def bound_from_energy(problem: Problem, work: float, energy: FloatArray) -> float:
    """sum_k w_k 2 f_k^T u_k  minus  the largest budget-feasible sum of E0 * energy."""
    a = problem.material.youngs_modulus_MPa * energy
    top = np.sort(a[problem.free])[::-1][:max(problem.room, 0)]
    return float(work - a[problem.preserved].sum() - top.sum())


def lower_bound(problem: Problem, model: fem.Model) -> LowerBound:
    e0 = problem.material.youngs_modulus_MPa
    n = problem.n_elements
    usable = ~problem.keep_out
    # Usable material that cannot reach any load adds nothing; keep it empty so the relaxed
    # stiffness matrix stays non-singular.
    active = np.zeros(n, dtype=bool)
    label = topology.components(usable, problem.nx, problem.ny)
    forces = [lc.force for lc in problem.load_cases]
    for comp in range(int(np.max(label, initial=-1)) + 1):
        elems = np.flatnonzero(label == comp)
        if topology.loaded(topology.nodes_of(elems, problem.nx), forces):
            active[elems] = True
    fixed_one = problem.preserved & active
    free = active & ~problem.preserved
    room = problem.max_solid - int(fixed_one.sum())

    x = np.zeros(n)
    x[fixed_one] = 1.0
    if free.any():
        x[free] = np.clip(room / free.sum(), X_MIN, 1.0)

    best = LowerBound(-np.inf, x.copy(), np.inf, 0, 0.0)
    for it in range(1, MAX_ITERATIONS + 1):
        sol = model.solve(x * e0)
        compliance = model.compliance(sol)
        energy = model.element_energy(sol)
        work = 2 * float(sum(w * f @ u for w, f, u in zip(model.weights, model.forces,
                                                           sol.displacements)))
        lb = bound_from_energy(problem, work, energy)
        if lb > best.value:
            best = LowerBound(lb, x.copy(), compliance, it, sol.max_relative_residual)
        if not free.any() or (compliance - best.value) <= REL_TOL * abs(best.value):
            break
        # Optimality Criteria step for the convex problem (dc/dx_e = -E0 * energy_e)
        dc = np.maximum(e0 * energy[free], 1e-300)
        lo, hi = 0.0, float(dc.max()) * 1e6
        xf = x[free]
        new = xf
        while (hi - lo) / (hi + lo + 1e-300) > 1e-9:
            lam = 0.5 * (lo + hi)
            cand = np.clip(xf * np.sqrt(dc / lam), np.maximum(X_MIN, xf - MOVE),
                           np.minimum(1.0, xf + MOVE))
            if cand.sum() > room:
                lo = lam
            else:
                hi, new = lam, cand
        x = x.copy()
        x[free] = new
    best.iterations = it
    return best
