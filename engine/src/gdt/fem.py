"""Plane-stress finite-element model on the structured mesh of square Q4 elements.

Element stiffness is computed with 2x2 Gauss integration. For square elements in 2D the
stiffness matrix does not depend on the element size, only on E, nu and the thickness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from .setup import Problem

FloatArray = NDArray[np.float64]

MAX_RELATIVE_RESIDUAL = 1e-10  # required accuracy of every linear solve (spec)


def element_stiffness(nu: float, thickness: float) -> FloatArray:
    """8x8 stiffness of a square element for E = 1 (multiply by the element's E).

    Node order: bottom-left, bottom-right, top-right, top-left; dofs (x, y) per node.
    """
    d = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu ** 2)
    ke = np.zeros((8, 8))
    g = 1 / np.sqrt(3)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    for xi in (-g, g):
        for eta in (-g, g):
            # derivatives of shape functions w.r.t. natural coords; element side 2 in
            # natural coords maps to side h, and B scales with 1/h while dA scales with h^2,
            # so we use a unit square directly (h cancels).
            dn_dxi = corners[:, 0] * (1 + corners[:, 1] * eta) / 4
            dn_deta = corners[:, 1] * (1 + corners[:, 0] * xi) / 4
            dn_dx, dn_dy = 2 * dn_dxi, 2 * dn_deta  # unit element: x = (xi + 1) / 2
            b = np.zeros((3, 8))
            b[0, 0::2] = dn_dx
            b[1, 1::2] = dn_dy
            b[2, 0::2] = dn_dy
            b[2, 1::2] = dn_dx
            ke += b.T @ d @ b * 0.25  # det J = 1/4 for the unit square
    result: FloatArray = ke * thickness
    return result


def element_dofs(nx: int, ny: int) -> NDArray[np.int64]:
    """(n_elements, 8) global dof indices, element index ey * nx + ex."""
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    ex, ey = gx.ravel(), gy.ravel()
    nnx = nx + 1
    n0 = ey * nnx + ex
    nodes = np.stack([n0, n0 + 1, n0 + nnx + 1, n0 + nnx], axis=1)
    dofs = np.empty((nx * ny, 8), dtype=np.int64)
    dofs[:, 0::2] = 2 * nodes
    dofs[:, 1::2] = 2 * nodes + 1
    return dofs


@dataclass
class Solution:
    displacements: list[FloatArray]  # one full-length vector per load case
    max_relative_residual: float


class Model:
    """Assembles and solves K(E_e) u = f for a problem's mesh, supports and load cases."""

    def __init__(self, problem: Problem):
        self.problem = problem
        self.ke = element_stiffness(problem.material.poissons_ratio, problem.thickness)
        self.dofs = element_dofs(problem.nx, problem.ny)
        self.rows = np.repeat(self.dofs, 8, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 8)).ravel()
        self.forces = [lc.force for lc in problem.load_cases]
        self.weights = np.array([lc.weight for lc in problem.load_cases])

    def stiffness(self, element_modulus: FloatArray) -> sp.csc_matrix:
        vals = (self.ke.ravel()[None, :] * element_modulus[:, None]).ravel()
        n = self.problem.n_dofs
        return sp.csc_matrix((vals, (self.rows, self.cols)), shape=(n, n))

    def solve(self, element_modulus: FloatArray) -> Solution:
        k = self.stiffness(element_modulus)
        touched = np.zeros(self.problem.n_dofs, dtype=bool)
        touched[self.dofs[element_modulus > 0].ravel()] = True
        free = np.flatnonzero(touched & ~self.problem.fixed)
        kff = k[free][:, free].tocsc()
        lu = spla.splu(kff, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                       options={"SymmetricMode": True})
        out, worst = [], 0.0
        for f in self.forces:
            u = np.zeros(self.problem.n_dofs)
            uf = lu.solve(f[free])
            u[free] = uf
            norm_f = np.linalg.norm(f[free])
            res = np.linalg.norm(kff @ uf - f[free]) / norm_f if norm_f > 0 else 0.0
            worst = max(worst, float(res))
            out.append(u)
        return Solution(out, worst)

    def compliance(self, sol: Solution) -> float:
        return float(sum(w * f @ u for w, f, u in zip(self.weights, self.forces,
                                                       sol.displacements)))

    def element_energy(self, sol: Solution) -> FloatArray:
        """Weighted sum over load cases of u_e^T k0 u_e per element (E = 1)."""
        s = np.zeros(self.problem.n_elements)
        for w, u in zip(self.weights, sol.displacements):
            ue = u[self.dofs]
            s += w * np.einsum("ij,jk,ik->i", ue, self.ke, ue)
        return s
