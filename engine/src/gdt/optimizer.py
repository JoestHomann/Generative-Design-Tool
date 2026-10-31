"""SIMP topology optimization with a density filter and Heaviside projection.

Minimizes the weighted compliance subject to the material budget. Preserved elements are
fixed solid and keep-out elements fixed empty. The update rule is Optimality Criteria.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from . import fem
from .setup import Problem

FloatArray = NDArray[np.float64]
Progress = Callable[[int, float, float, float], None]

PENALTY = 3.0
E_MIN_RATIO = 1e-9
ETA = 0.5
BETA_MAX = 32.0
MOVE = 0.2
BETA_STEP_ITERATIONS = 40
STALL_WINDOW = 10
MIN_ITERATIONS_AT_BETA_MAX = 50
STALL_TOL = 1e-3


@dataclass
class OptimizerResult:
    density: FloatArray  # physical (filtered, projected) densities in [0, 1]
    iterations: int
    converged: bool
    interrupted: bool
    kkt_residual: float
    compliance: float


def density_filter(problem: Problem) -> tuple[sp.csr_matrix, FloatArray]:
    """Linear density filter; radius = minimum member size, at least 1.5 elements."""
    nx, ny = problem.nx, problem.ny
    r = max(problem.min_member_size / problem.h, 1.5)
    reach = int(np.ceil(r)) - 1
    rows, cols, vals = [], [], []
    for j in range(ny):
        for i in range(nx):
            e = j * nx + i
            for b in range(max(j - reach, 0), min(j + reach, ny - 1) + 1):
                for a in range(max(i - reach, 0), min(i + reach, nx - 1) + 1):
                    w = r - np.hypot(i - a, j - b)
                    if w > 0:
                        rows.append(e)
                        cols.append(b * nx + a)
                        vals.append(w)
    h = sp.csr_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))
    return h, np.asarray(h.sum(axis=1)).ravel()


def _project(xt: FloatArray, beta: float) -> FloatArray:
    d = np.tanh(beta * ETA) + np.tanh(beta * (1 - ETA))
    out: FloatArray = (np.tanh(beta * ETA) + np.tanh(beta * (xt - ETA))) / d
    return out


def _project_grad(xt: FloatArray, beta: float) -> FloatArray:
    d = np.tanh(beta * ETA) + np.tanh(beta * (1 - ETA))
    out: FloatArray = beta * (1 - np.tanh(beta * (xt - ETA)) ** 2) / d
    return out


def run(problem: Problem, model: fem.Model, progress: Progress | None = None) -> OptimizerResult:
    n = problem.n_elements
    free = problem.free
    e0 = problem.material.youngs_modulus_MPa
    e_min = E_MIN_RATIO * e0
    filt, filt_sum = density_filter(problem)
    target = problem.max_solid / n  # volume fraction of physical densities

    x = np.zeros(n)
    x[problem.preserved] = 1.0
    n_free = int(free.sum())
    x[free] = (problem.max_solid - problem.preserved.sum()) / max(n_free, 1)

    def physical(xd: FloatArray, beta: float) -> tuple[FloatArray, FloatArray]:
        xt = (filt @ xd) / filt_sum
        xp = _project(xt, beta)
        xp[problem.preserved] = 1.0
        xp[problem.keep_out] = 0.0
        return xt, xp

    beta, since_beta = 1.0, 0
    lam = 0.0
    dc_x = np.zeros(n)
    dv_x = np.zeros(n)
    compliance = float("nan")
    converged = interrupted = False
    it = 0
    last_complete_x = x.copy()
    history: list[float] = []
    try:
        while it < problem.settings.max_iterations:
            it += 1
            since_beta += 1
            xt, xp = physical(x, beta)
            sol = model.solve(e_min + xp ** PENALTY * (e0 - e_min))
            compliance = model.compliance(sol)
            energy = model.element_energy(sol)
            dc_p = -PENALTY * xp ** (PENALTY - 1) * (e0 - e_min) * energy
            dp = _project_grad(xt, beta)
            dc_t = dc_p * dp
            dv_t = np.ones(n) * dp
            dc_t[problem.preserved | problem.keep_out] = 0.0
            dv_t[problem.preserved | problem.keep_out] = 0.0
            dc_x = filt.T @ (dc_t / filt_sum)
            dv_x = filt.T @ (dv_t / filt_sum)
            dc_x = np.minimum(dc_x, 0.0)
            dv_x = np.maximum(dv_x, 1e-12)

            lo, hi = 0.0, 1e9 * max(1.0, float(-dc_x[free].min()) if n_free else 1.0)
            xnew = x
            while (hi - lo) / (hi + lo + 1e-300) > 1e-6:
                lam = 0.5 * (lo + hi)
                cand = x.copy()
                step = x[free] * np.sqrt(-dc_x[free] / (dv_x[free] * lam))
                cand[free] = np.clip(step, np.maximum(0.0, x[free] - MOVE),
                                     np.minimum(1.0, x[free] + MOVE))
                if physical(cand, beta)[1].mean() > target:
                    lo = lam
                else:
                    hi = lam
                    xnew = cand
            change = float(np.abs(xnew - x).max()) if n_free else 0.0
            x = xnew
            last_complete_x = x.copy()
            if progress:
                progress(it, compliance, float(physical(x, beta)[1].mean()), change)
            history.append(compliance)
            if beta < BETA_MAX and (change < problem.settings.tolerance
                                    or since_beta >= BETA_STEP_ITERATIONS):
                beta, since_beta = beta * 2, 0
            elif beta >= BETA_MAX and (change < problem.settings.tolerance or (
                    since_beta >= MIN_ITERATIONS_AT_BETA_MAX and _stalled(history))):
                converged = True
                break
    except KeyboardInterrupt:
        interrupted = True
        x = last_complete_x  # an interrupt mid-iteration must not leave a half-updated x

    _, xp = physical(x, beta)
    # KKT residual of the final continuous design (bound-projected gradient of the Lagrangian)
    g = dc_x + lam * dv_x
    r = np.where(x <= 1e-3, np.minimum(g, 0.0), np.where(x >= 1 - 1e-3, np.maximum(g, 0.0), g))
    scale = float(np.abs(dc_x[free]).max()) if n_free else 1.0
    kkt = float(np.abs(r[free]).max() / scale) if n_free and scale > 0 else 0.0
    return OptimizerResult(xp, it, converged, interrupted, kkt, compliance)



def _stalled(history: list[float]) -> bool:
    """Compliance changed by less than STALL_TOL (relative) over the last STALL_WINDOW steps."""
    if len(history) <= STALL_WINDOW:
        return False
    recent = history[-STALL_WINDOW - 1:]
    return (max(recent) - min(recent)) <= STALL_TOL * abs(recent[-1])
