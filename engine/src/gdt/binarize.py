"""Turns continuous densities into a feasible binary design."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from . import fem, topology
from .setup import Problem


@dataclass
class Binary:
    design: NDArray[np.bool_]
    feasible: bool
    issues: list[str] = field(default_factory=list)
    removed_islands: int = 0


def binarize(problem: Problem, model: fem.Model, density: NDArray[np.float64]) -> Binary:
    """Build candidate binary designs from the densities and keep the stiffest feasible one.

    Candidates: (1) the densest free elements up to the budget (a plain threshold);
    (2) growth from the supports, the loaded nodes and the preserved regions, always adding
    the densest free element that touches the part. Growth keeps the design connected
    where the densities are grey; for a near-binary density field both agree.
    """
    best: Binary | None = None
    best_c = np.inf
    for candidate in (_threshold(problem, density), _grow(problem, density)):
        # Members that meet only at a corner node still carry load through that node, so
        # try keeping corner-connected material first; edge-only cleaning is the fallback
        # that can never leave a hinge mechanism behind.
        for corners in (True, False):
            b = clean(problem, candidate, corners)
            if not b.feasible:
                if best is None:
                    best = b
                continue
            c = _compliance_if_solvable(problem, model, b.design)
            if c < best_c:
                best, best_c = b, c
    assert best is not None
    return best


def _compliance_if_solvable(problem: Problem, model: fem.Model,
                            design: NDArray[np.bool_]) -> float:
    """Compliance of a binary design, or infinity if its stiffness matrix is singular."""
    try:
        with np.errstate(all="ignore"):
            sol = model.solve(design.astype(float) * problem.material.youngs_modulus_MPa)
    except RuntimeError:  # SuperLU: factor is exactly singular
        return float("inf")
    c = model.compliance(sol)
    if not np.isfinite(c) or not sol.max_relative_residual <= fem.MAX_RELATIVE_RESIDUAL:
        return float("inf")
    return c


def _threshold(problem: Problem, density: NDArray[np.float64]) -> NDArray[np.bool_]:
    free = problem.free
    k = min(problem.room, int(free.sum()))
    idx = np.flatnonzero(free)
    design = problem.preserved.copy()
    design[idx[np.argsort(-density[idx], kind="stable")][:k]] = True
    return design


def _grow(problem: Problem, density: NDArray[np.float64]) -> NDArray[np.bool_]:
    nx, ny = problem.nx, problem.ny
    free = problem.free
    budget = min(problem.room, int(free.sum()))
    design = problem.preserved.copy()
    supported = problem.fixed[0::2] | problem.fixed[1::2]
    loaded = np.zeros(problem.n_dofs // 2, dtype=bool)
    for lc in problem.load_cases:
        loaded |= (lc.force[0::2] != 0) | (lc.force[1::2] != 0)
    heap: list[tuple[float, int]] = []
    queued = np.zeros(problem.n_elements, dtype=bool)

    def push(e: int) -> None:
        if free[e] and not queued[e] and not design[e]:
            queued[e] = True
            heapq.heappush(heap, (-float(density[e]), e))

    def push_neighbours(e: int) -> None:
        for n in topology.neighbours(e, nx, ny):
            push(n)

    added = 0
    # first connect every loaded node to the supports along the densest route
    usable = free | design
    support_elems = np.zeros(problem.n_elements, dtype=bool)
    for u in np.flatnonzero(usable):
        support_elems[u] = bool(supported[topology.nodes_of(np.array([u]), nx)].any())
    for node in np.flatnonzero(loaded):
        starts = [e for e in _elements_at(int(node), nx, ny) if usable[e]]
        new: list[int] = []
        for weight in (density, np.zeros_like(density)):  # densest route, else shortest
            path = _densest_path(starts, support_elems, usable, design, weight, nx, ny)
            new = [e for e in path if not design[e]]
            if added + len(new) <= budget:
                break
        if not new or added + len(new) > budget:
            continue
        design[new] = True
        added += len(new)
    for k in range(problem.n_elements):
        nodes = topology.nodes_of(np.array([k]), nx)
        if design[k]:
            push_neighbours(k)
        elif free[k] and supported[nodes].any():
            push(k)
    while heap and added < budget:
        _, picked = heapq.heappop(heap)
        design[picked] = True
        added += 1
        push_neighbours(picked)
    return design


def _densest_path(starts: list[int], targets: NDArray[np.bool_], usable: NDArray[np.bool_],
                  design: NDArray[np.bool_], density: NDArray[np.float64], nx: int,
                  ny: int) -> list[int]:
    """Edge-connected element path from a start element to a target element that minimizes
    the material it adds, weighting each new element by (1 - density). Empty if none."""
    cost = np.where(design, 0.0, 1.0 - np.clip(density, 0.0, 1.0) + 1e-6)
    dist = np.full(nx * ny, np.inf)
    prev = np.full(nx * ny, -1, dtype=np.int64)
    heap: list[tuple[float, int]] = []
    for e in starts:
        dist[e] = cost[e]
        heapq.heappush(heap, (float(dist[e]), e))
    while heap:
        d, e = heapq.heappop(heap)
        if d > dist[e]:
            continue
        if targets[e]:
            path = [e]
            while prev[path[-1]] >= 0:
                path.append(int(prev[path[-1]]))
            return path
        for n in topology.neighbours(e, nx, ny):
            if usable[n] and d + cost[n] < dist[n]:
                dist[n] = d + cost[n]
                prev[n] = e
                heapq.heappush(heap, (float(dist[n]), n))
    return []


def _elements_at(node: int, nx: int, ny: int) -> list[int]:
    i, j = node % (nx + 1), node // (nx + 1)
    out = []
    for a, b in ((i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j)):
        if 0 <= a < nx and 0 <= b < ny:
            out.append(b * nx + a)
    return out


def clean(problem: Problem, design: NDArray[np.bool_], corners: bool = False) -> Binary:
    """Remove solid groups that carry no load and check that every load is held.
    Groups connect through shared edges, and also through corner nodes if ``corners``."""
    design = design.copy()
    forces = [lc.force for lc in problem.load_cases]
    label = topology.components(design, problem.nx, problem.ny, corners)
    issues: list[str] = []
    removed = 0
    for comp in range(int(np.max(label, initial=-1)) + 1):
        elems = np.flatnonzero(label == comp)
        nodes = topology.nodes_of(elems, problem.nx)
        is_loaded = topology.loaded(nodes, forces)
        held = topology.support_rank(nodes, problem.fixed, problem.nx, problem.h) == 3
        if is_loaded and not held:
            issues.append(f"a loaded part of the design ({elems.size} elements) is not held by "
                          f"the supports")
        elif not is_loaded:
            if problem.preserved[elems].any():
                if not held:
                    issues.append("a preserved region is neither connected to the loads nor "
                                  "held by the supports")
            else:
                design[elems] = False
                removed += 1
    # every load must sit on the remaining solid material
    touched = np.zeros(problem.n_dofs, dtype=bool)
    if design.any():
        nodes = topology.nodes_of(np.flatnonzero(design), problem.nx)
        touched[2 * nodes] = touched[2 * nodes + 1] = True
    for lc in problem.load_cases:
        if np.any(lc.force[~touched]):
            issues.append(f"load case '{lc.name}': a load is not attached to the design")
    return Binary(design, not issues, issues, removed)
