"""Independent re-verification of a Generative Design Tool certificate.

This module deliberately shares no code with the engine. It re-derives the mesh, regions,
supports and loads from the setup embedded in the certificate, uses the closed-form
stiffness matrix of the square bilinear plane-stress element, and recomputes every claim.

Conventions (from the certificate's ``model`` section and the setup file format):
origin bottom-left, x right, y up, mm; element index ey * nx + ex; node index
j * (nx + 1) + i with dofs 2k (x) and 2k + 1 (y); element node order bottom-left,
bottom-right, top-right, top-left; a region contains an element if it contains the
element centroid; a support covers the nodes inside its rectangle.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

REL_TOL = 1e-6        # agreement required between certificate and recomputed values
MIN_MARGIN = 1e-9     # smallest floating-point margin a certificate may claim
GAP_ABS_TOL = 1e-6    # the gap is a fraction; compare it absolutely
TARGET_GAP = 0.05
STRETCH_GAP = 0.02

FArr = NDArray[np.float64]
BArr = NDArray[np.bool_]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> bool:
        self.checks.append(Check(name, bool(passed), detail))
        return bool(passed)

    def summary(self) -> str:
        lines = [f"{'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}" for c in self.checks]
        lines.append(f"Verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


class _Invalid(Exception):
    pass


REQUIRED = {
    "schema": None, "schema_version": None, "setup": None, "material": None, "model": None,
    "design": None, "numerics": None, "bounds": None,
}
REQUIRED_BOUNDS = ("upper_bound_Nmm", "lower_bound_Nmm", "upper_bound_adjusted_Nmm",
                   "lower_bound_adjusted_Nmm", "gap", "target")


def check(cert: dict[str, Any]) -> Report:
    report = Report()
    try:
        _check(cert, report)
    except _Invalid as exc:
        report.add("schema", False, str(exc))
    return report


def _check(cert: dict[str, Any], report: Report) -> None:
    missing = [k for k in REQUIRED if k not in cert]
    if missing or cert.get("schema") != "gdt-certificate" or cert.get("schema_version") != 1:
        raise _Invalid(f"not a version-1 gdt certificate (missing: {', '.join(missing) or '-'})")
    bounds = cert["bounds"]
    if bounds.get("status") != "certified":
        report.add("status", False, f"certificate makes no certified claim "
                                    f"(status: {bounds.get('status')})")
        return
    missing = [k for k in REQUIRED_BOUNDS if k not in bounds]
    if missing or "relaxed_design" not in cert or "relative_margin" not in cert["numerics"]:
        raise _Invalid(f"certified bounds incomplete (missing: "
                       f"{', '.join(missing + ([] if 'relaxed_design' in cert else ['relaxed_design']))})")
    report.add("schema", True, "version-1 certificate with all required fields")

    mesh = _Mesh(cert["setup"])
    if (mesh.nx, mesh.ny) != (cert["model"]["nx"], cert["model"]["ny"]):
        report.add("setup", False, "mesh in setup and model section disagree")
        return
    report.add("setup", True, f"{mesh.nx} x {mesh.ny} square elements of {mesh.h:g} mm")

    design = _decode_bits(cert["design"]["data"], mesh.n_el)
    budget = cert["setup"]["material_budget"]
    max_solid = math.floor(budget * mesh.n_el + 1e-9)
    report.add("material budget", int(design.sum()) <= max_solid,
               f"{int(design.sum())} solid elements, budget allows {max_solid}")
    report.add("preserved regions", bool(design[mesh.preserved].all()),
               f"{int(mesh.preserved.sum())} preserved elements, "
               f"{int((mesh.preserved & ~design).sum())} empty")
    report.add("keep-out regions", not bool(design[mesh.keep_out].any()),
               f"{int((mesh.keep_out & design).sum())} solid elements inside keep-out regions")

    mat = cert["material"]
    fe = _FE(mesh, float(mat["youngs_modulus_MPa"]), float(mat["poissons_ratio"]))
    try:
        ub, ub_res = fe.compliance(design.astype(float))
    except _Singular as exc:
        report.add("load path", False, str(exc))
        return
    report.add("load path", True, "every load is carried by the design")
    ub_claim = float(bounds["upper_bound_Nmm"])
    report.add("upper bound", _close(ub, ub_claim),
               f"recomputed {ub:.10g} N mm, certificate {ub_claim:.10g} N mm")

    x_hat = _decode_floats(cert["relaxed_design"]["data"], mesh.n_el)
    try:
        lb, lb_res = fe.lower_bound(x_hat, design_budget=max_solid)
    except _Singular as exc:
        report.add("lower bound", False, f"relaxed design gives no solvable system ({exc})")
        return
    lb_claim = float(bounds["lower_bound_Nmm"])
    report.add("lower bound", _close(lb, lb_claim) and lb <= ub * (1 + REL_TOL),
               f"recomputed {lb:.10g} N mm, certificate {lb_claim:.10g} N mm")

    margin = float(cert["numerics"]["relative_margin"])
    own = max(MIN_MARGIN, 10 * max(ub_res, lb_res))
    gap = (ub * (1 + margin) - lb * (1 - margin)) / (lb * (1 - margin))
    ok = (margin >= own and abs(gap - float(bounds["gap"])) <= GAP_ABS_TOL
          and _close(ub * (1 + margin), float(bounds["upper_bound_adjusted_Nmm"]))
          and _close(lb * (1 - margin), float(bounds["lower_bound_adjusted_Nmm"])))
    report.add("gap", ok, f"recomputed {100 * gap:.4f} %, certificate "
                          f"{100 * float(bounds['gap']):.4f} %, margin {margin:.1e} "
                          f"(needs >= {own:.1e})")
    t = bounds["target"]
    report.add("target", t.get("met") == (gap <= TARGET_GAP)
               and t.get("stretch_met") == (gap <= STRETCH_GAP),
               f"gap {100 * gap:.2f} %: target {'met' if gap <= TARGET_GAP else 'not met'}, "
               f"stretch {'met' if gap <= STRETCH_GAP else 'not met'}")


def _close(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def _decode_bits(data: str, n: int) -> BArr:
    raw = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    bits = np.unpackbits(raw)
    if bits.size < n or bits[n:].any():
        raise _Invalid("design data has the wrong length")
    out: BArr = bits[:n].astype(bool)
    return out


def _decode_floats(data: str, n: int) -> FArr:
    x = np.frombuffer(base64.b64decode(data), dtype="<f8").astype(np.float64)
    if x.size != n or not np.all(np.isfinite(x)):
        raise _Invalid("relaxed design data has the wrong length or invalid values")
    return x


class _Mesh:
    """Mesh, regions, supports and loads re-derived from the setup."""

    def __init__(self, s: dict[str, Any]):
        try:
            env, msh = s["envelope"], s["mesh"]
            self.nx, self.ny = int(msh["nx"]), int(msh["ny"])
            self.h = float(env["width_mm"]) / self.nx
            if abs(float(env["height_mm"]) / self.ny - self.h) > 1e-6 * self.h:
                raise _Invalid("setup: elements are not square")
            self.t = float(env["thickness_mm"])
            self.n_el = self.nx * self.ny
            self.n_nodes = (self.nx + 1) * (self.ny + 1)
            cx: FArr = ((np.arange(self.n_el) % self.nx + 0.5) * self.h).astype(np.float64)
            cy: FArr = ((np.arange(self.n_el) // self.nx + 0.5) * self.h).astype(np.float64)
            self.preserved = np.zeros(self.n_el, dtype=bool)
            for r in s.get("preserved_regions", []):
                self.preserved |= self._inside(r, cx, cy)
            self.keep_out = np.zeros(self.n_el, dtype=bool)
            for r in s.get("keep_out_regions", []):
                self.keep_out |= self._inside(r, cx, cy)
            nxp = self.nx + 1
            node_x = (np.arange(self.n_nodes) % nxp) * self.h
            node_y = (np.arange(self.n_nodes) // nxp) * self.h
            self.fixed = np.zeros(2 * self.n_nodes, dtype=bool)
            for sup in s["supports"]:
                tol = 1e-6 * self.h
                on = ((node_x >= sup["x_mm"] - tol) & (node_x <= sup["x_mm"] + sup["width_mm"] + tol)
                      & (node_y >= sup["y_mm"] - tol)
                      & (node_y <= sup["y_mm"] + sup["height_mm"] + tol))
                # only nodes that touch an element outside the keep-out regions are supported
                on &= self._node_touches(~self.keep_out)
                if "x" in sup["fix"]:
                    self.fixed[0::2] |= on
                if "y" in sup["fix"]:
                    self.fixed[1::2] |= on
            self.weights = np.array([float(lc["weight"]) for lc in s["load_cases"]])
            self.forces = [self._force(lc["loads"]) for lc in s["load_cases"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise _Invalid(f"setup could not be read: {exc!r}") from None

    def _inside(self, r: dict[str, Any], cx: FArr, cy: FArr) -> BArr:
        eps = 1e-9
        if r["shape"] == "rect":
            m: BArr = ((cx >= r["x_mm"] - eps) & (cx <= r["x_mm"] + r["width_mm"] + eps)
                       & (cy >= r["y_mm"] - eps) & (cy <= r["y_mm"] + r["height_mm"] + eps))
            return m
        if r["shape"] == "circle":
            c: BArr = (cx - r["cx_mm"]) ** 2 + (cy - r["cy_mm"]) ** 2 <= r["r_mm"] ** 2 + eps
            return c
        raise _Invalid(f"unknown region shape {r['shape']!r}")

    def corners(self) -> NDArray[np.int64]:
        e = np.arange(self.n_el)
        i, j = e % self.nx, e // self.nx
        bl = j * (self.nx + 1) + i
        return np.stack([bl, bl + 1, bl + self.nx + 2, bl + self.nx + 1], axis=1)

    def _node_touches(self, elements: BArr) -> BArr:
        out = np.zeros(self.n_nodes, dtype=bool)
        out[self.corners()[elements].ravel()] = True
        return out

    def _force(self, loads: list[dict[str, Any]]) -> FArr:
        f = np.zeros(2 * self.n_nodes)
        nxp = self.nx + 1
        for ld in loads:
            if ld["type"] == "point":
                i = int(round(ld["x_mm"] / self.h))
                j = int(round(ld["y_mm"] / self.h))
                k = j * nxp + i
                f[2 * k] += ld["fx_N"]
                f[2 * k + 1] += ld["fy_N"]
            elif ld["type"] == "edge":
                vertical = ld["edge"] in ("left", "right")
                count = self.ny if vertical else self.nx
                # consistent nodal loads of a uniform traction on [from, to]:
                # integrate the linear hat function of each edge node over the range
                pos = np.arange(count + 1) * self.h
                w = np.array([_hat_integral(p, self.h, ld["from_mm"], ld["to_mm"]) for p in pos])
                w = w / w.sum()
                for s_, wi in enumerate(w):
                    if wi == 0:
                        continue
                    i, j = {"left": (0, s_), "right": (self.nx, s_), "bottom": (s_, 0),
                            "top": (s_, self.ny)}[ld["edge"]]
                    k = j * nxp + i
                    f[2 * k] += wi * ld["fx_N"]
                    f[2 * k + 1] += wi * ld["fy_N"]
            else:
                raise _Invalid(f"unknown load type {ld['type']!r}")
        return f


def _hat_integral(p: float, h: float, a: float, b: float) -> float:
    """Integral over [a, b] of the hat function centred at p with half-width h."""
    total = 0.0
    # rising flank on [p - h, p]: antiderivative (x - (p - h))^2 / (2h)
    lo, hi = max(p - h, a), min(p, b)
    if hi > lo:
        total += ((hi - (p - h)) ** 2 - (lo - (p - h)) ** 2) / (2 * h)
    # falling flank on [p, p + h]: antiderivative -((p + h) - x)^2 / (2h)
    lo, hi = max(p, a), min(p + h, b)
    if hi > lo:
        total += (((p + h) - lo) ** 2 - ((p + h) - hi) ** 2) / (2 * h)
    return total


class _Singular(Exception):
    pass


class _FE:
    """Closed-form Q4 plane-stress element (as in Sigmund's 99-line code)."""

    def __init__(self, mesh: _Mesh, e0: float, nu: float):
        self.mesh, self.e0 = mesh, e0
        k = np.array([1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
                      -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8])
        pattern = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2],
                            [2, 7, 0, 5, 6, 3, 4, 1], [3, 6, 5, 0, 7, 2, 1, 4],
                            [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6],
                            [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0]])
        self.ke = mesh.t / (1 - nu ** 2) * k[pattern]
        c = mesh.corners()
        self.edof = np.empty((mesh.n_el, 8), dtype=np.int64)
        self.edof[:, 0::2], self.edof[:, 1::2] = 2 * c, 2 * c + 1

    def _solve(self, density: FArr) -> tuple[list[FArr], float]:
        m = self.mesh
        n = 2 * m.n_nodes
        rows = np.repeat(self.edof, 8, axis=1).ravel()
        cols = np.tile(self.edof, (1, 8)).ravel()
        vals = (self.ke.ravel()[None, :] * (self.e0 * density)[:, None]).ravel()
        k = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        active = np.zeros(n, dtype=bool)
        active[self.edof[density > 0].ravel()] = True
        free = np.flatnonzero(active & ~m.fixed)
        for f in m.forces:
            if np.any(f[~active] != 0):
                raise _Singular("a load acts on a node that no material touches")
        kff = k[free][:, free].tocsc()
        us, worst = [], 0.0
        for f in m.forces:
            with np.errstate(all="raise"):
                try:
                    uf = spla.spsolve(kff, f[free])
                except (RuntimeError, FloatingPointError, spla.MatrixRankWarning) as exc:
                    raise _Singular(f"stiffness matrix is singular ({exc})") from None
            if not np.all(np.isfinite(uf)):
                raise _Singular("stiffness matrix is singular (non-finite displacements)")
            res = float(np.linalg.norm(kff @ uf - f[free]) / max(np.linalg.norm(f[free]), 1e-300))
            if res > 1e-6:
                raise _Singular(f"solve did not converge (relative residual {res:.1e}); "
                                "the part can move without resistance")
            worst = max(worst, res)
            u = np.zeros(n)
            u[free] = uf
            us.append(u)
        return us, worst

    def compliance(self, density: FArr) -> tuple[float, float]:
        us, res = self._solve(density)
        return float(sum(w * f @ u for w, f, u in zip(self.mesh.weights, self.mesh.forces, us))), res

    def lower_bound(self, x_hat: FArr, design_budget: int) -> tuple[float, float]:
        """min over the relaxed feasible set of sum_k w_k (2 f_k.u_k - sum_e y_e E0 u_e.k0.u_e)."""
        m = self.mesh
        us, res = self._solve(np.clip(x_hat, 0.0, 1.0))
        work = 0.0
        energy = np.zeros(m.n_el)
        for w, f, u in zip(m.weights, m.forces, us):
            work += w * 2 * float(f @ u)
            ue = u[self.edof]
            energy += w * self.e0 * np.einsum("ei,ij,ej->e", ue, self.ke, ue)
        free = ~m.preserved & ~m.keep_out
        room = max(design_budget - int(m.preserved.sum()), 0)
        best_free = np.sort(energy[free])[::-1][:room].sum()
        return float(work - energy[m.preserved].sum() - best_free), res
