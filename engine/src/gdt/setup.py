"""Problem setup file: parsing, validation and resolution onto the mesh.

Conventions (shared with the certificate):
- Origin at the bottom-left corner of the design envelope, x to the right, y up, mm.
- Square elements. Element (ex, ey) has index ey * nx + ex, ey = 0 is the bottom row.
- Node (i, j) has index j * (nx + 1) + i. Node k has degrees of freedom 2k (x), 2k + 1 (y).
- An element belongs to a region if its centroid lies inside the region (boundary included).
- A support covers every node inside its rectangle (boundary included).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import materials as mat
from . import topology

SCHEMA_VERSION = 1
GEOM_TOL = 1e-9  # mm, relative to element size where it matters


class SetupError(ValueError):
    """The setup cannot be solved. ``issues`` lists every problem found."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Setup is not solvable:\n" + "\n".join(f"- {i}" for i in issues))


@dataclass(frozen=True)
class Settings:
    max_iterations: int = 300
    tolerance: float = 0.01
    factor_of_safety: float = 1.5
    fitting_factor: float = 1.15


@dataclass
class LoadCase:
    name: str
    weight: float
    force: NDArray[np.float64]  # nodal force vector, N (limit load)


@dataclass
class Problem:
    width: float
    height: float
    thickness: float
    nx: int
    ny: int
    material: mat.Material
    budget: float
    min_member_size: float
    preserved: NDArray[np.bool_]
    keep_out: NDArray[np.bool_]
    fixed: NDArray[np.bool_]
    load_cases: list[LoadCase]
    settings: Settings
    raw: dict[str, Any] = field(repr=False)

    @property
    def h(self) -> float:
        return self.width / self.nx

    @property
    def n_elements(self) -> int:
        return self.nx * self.ny

    @property
    def n_dofs(self) -> int:
        return 2 * (self.nx + 1) * (self.ny + 1)

    @property
    def max_solid(self) -> int:
        """Largest number of solid elements the material budget allows."""
        return max_solid_elements(self.budget, self.n_elements)

    @property
    def free(self) -> NDArray[np.bool_]:
        """Elements the optimizer may fill or empty (neither preserved nor keep-out)."""
        out: NDArray[np.bool_] = ~self.preserved & ~self.keep_out
        return out

    @property
    def room(self) -> int:
        """Solid elements the budget leaves after the preserved regions."""
        return self.max_solid - int(self.preserved.sum())


def max_solid_elements(budget: float, n_elements: int) -> int:
    return math.floor(budget * n_elements + 1e-9)


_TOP_KEYS = {
    "schema_version": True, "envelope": True, "mesh": True, "material": True,
    "material_budget": True, "min_member_size_mm": True, "preserved_regions": False,
    "keep_out_regions": False, "supports": True, "load_cases": True, "settings": False,
}
_SETTINGS_KEYS = {"max_iterations", "tolerance", "factor_of_safety", "fitting_factor"}


def load(source: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    with open(source, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def parse(raw: dict[str, Any], library: dict[str, Any]) -> Problem:
    """Validate a setup and resolve it onto the mesh. Raises SetupError listing all issues."""
    issues: list[str] = []
    _check_keys(raw, _TOP_KEYS, "setup", issues)
    if issues:
        raise SetupError(issues)
    if raw["schema_version"] != SCHEMA_VERSION:
        raise SetupError([f"schema_version: expected {SCHEMA_VERSION}, got {raw['schema_version']}"])

    env = raw["envelope"]
    _check_keys(env, {"width_mm": True, "height_mm": True, "thickness_mm": True}, "envelope", issues)
    msh = raw["mesh"]
    _check_keys(msh, {"nx": True, "ny": True}, "mesh", issues)
    settings_raw = raw.get("settings", {})
    for k in settings_raw:
        if k not in _SETTINGS_KEYS:
            issues.append(f"settings.{k}: unknown field")
    if issues:
        raise SetupError(issues)

    width = _positive(env, "width_mm", "envelope", issues)
    height = _positive(env, "height_mm", "envelope", issues)
    thickness = _positive(env, "thickness_mm", "envelope", issues)
    nx, ny = msh["nx"], msh["ny"]
    if not (isinstance(nx, int) and isinstance(ny, int) and nx >= 1 and ny >= 1):
        issues.append("mesh: nx and ny must be positive integers")
    budget = raw["material_budget"]
    if not isinstance(budget, (int, float)) or not 0 < budget <= 1:
        issues.append("material_budget: must be a volume fraction in (0, 1]")
        budget = 1.0
    min_member = raw["min_member_size_mm"]
    if not isinstance(min_member, (int, float)) or min_member < 0:
        issues.append("min_member_size_mm: must be a length in mm, >= 0")
        min_member = 0.0
    if any(i.startswith(("envelope", "mesh")) for i in issues):
        raise SetupError(issues)
    if abs(width / nx - height / ny) > 1e-6 * (width / nx):
        raise SetupError([f"mesh: elements must be square, but width/nx = {width / nx:g} mm "
                          f"and height/ny = {height / ny:g} mm"])

    mspec = raw["material"]
    material_issues: list[str] = []
    if isinstance(mspec, dict) and "custom" in mspec:
        _check_keys(mspec, {"custom": True}, "material", material_issues)
        _check_keys(mspec["custom"], {"name": True, "youngs_modulus_GPa": True,
                                      "poissons_ratio": True, "density_kg_m3": True,
                                      "yield_MPa": False, "ultimate_MPa": False},
                    "material.custom", material_issues)
    else:
        _check_keys(mspec, {"id": True, "allowable": False}, "material", material_issues)
    if not material_issues:
        try:
            material = mat.resolve(raw["material"], library)
        except mat.MaterialError as exc:
            material_issues.append(str(exc))
    if material_issues:
        raise SetupError(issues + material_issues)

    h = width / nx
    cx = (np.arange(nx) + 0.5) * h
    cy = (np.arange(ny) + 0.5) * h
    ex, ey = np.meshgrid(cx, cy)  # shape (ny, nx), row ey
    ecx, ecy = ex.ravel(), ey.ravel()

    preserved = np.zeros(nx * ny, dtype=bool)
    for i, reg in enumerate(raw.get("preserved_regions", [])):
        m = _region_mask(reg, f"preserved_regions[{i}]", ecx, ecy, issues, allow_circle=False)
        if m is not None:
            if not m.any():
                issues.append(f"preserved_regions[{i}]: contains no element centroid")
            preserved |= m
    keep_out = np.zeros(nx * ny, dtype=bool)
    for i, reg in enumerate(raw.get("keep_out_regions", [])):
        m = _region_mask(reg, f"keep_out_regions[{i}]", ecx, ecy, issues, allow_circle=True)
        if m is not None:
            keep_out |= m
    if (preserved & keep_out).any():
        issues.append("preserved_regions and keep_out_regions overlap")

    # Nodes that touch at least one element that may hold material.
    node_ok = np.zeros((ny + 1) * (nx + 1), dtype=bool)
    usable = ~keep_out
    for e in np.flatnonzero(usable):
        i, j = e % nx, e // nx
        for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
            node_ok[(j + dj) * (nx + 1) + i + di] = True

    nnx, nny = nx + 1, ny + 1
    node_x = np.tile(np.arange(nnx) * h, nny)
    node_y = np.repeat(np.arange(nny) * h, nnx)
    fixed = np.zeros(2 * nnx * nny, dtype=bool)
    supports = raw["supports"]
    if not supports:
        issues.append("supports: at least one support is required, otherwise the part can "
                      "move freely (rigid-body motion)")
    for i, sup in enumerate(supports):
        name = f"supports[{i}]"
        if not _check_keys(sup, {"x_mm": True, "y_mm": True, "width_mm": True,
                                 "height_mm": True, "fix": True}, name, issues):
            continue
        if sup["fix"] not in ("x", "y", "xy"):
            issues.append(f"{name}.fix: must be 'x', 'y' or 'xy'")
            continue
        tol = 1e-6 * h
        inside = ((node_x >= sup["x_mm"] - tol) & (node_x <= sup["x_mm"] + sup["width_mm"] + tol)
                  & (node_y >= sup["y_mm"] - tol)
                  & (node_y <= sup["y_mm"] + sup["height_mm"] + tol))
        if not inside.any():
            issues.append(f"{name}: covers no node of the mesh")
            continue
        if not node_ok[inside].any():
            # Nodes over empty keep-out space carry no material, so a support covering
            # some of them only acts on the remaining nodes. Entirely inside is an error.
            issues.append(f"{name}: lies entirely inside keep-out regions")
            continue
        nodes = np.flatnonzero(inside & node_ok)
        if "x" in sup["fix"]:
            fixed[2 * nodes] = True
        if "y" in sup["fix"]:
            fixed[2 * nodes + 1] = True

    load_cases: list[LoadCase] = []
    if not raw["load_cases"]:
        issues.append("load_cases: at least one load case is required")
    for c, lc in enumerate(raw["load_cases"]):
        cname = f"load_cases[{c}]"
        if not _check_keys(lc, {"name": True, "weight": True, "loads": True}, cname, issues):
            continue
        if not isinstance(lc["weight"], (int, float)) or lc["weight"] <= 0:
            issues.append(f"{cname}.weight: must be > 0")
        f = np.zeros(2 * nnx * nny)
        for k, ld in enumerate(lc["loads"]):
            _apply_load(ld, f"{cname}.loads[{k}]", f, nx, ny, h, node_ok, issues)
        if not np.any(f):
            issues.append(f"{cname} ('{lc['name']}'): applies no force")
        load_cases.append(LoadCase(str(lc["name"]), float(lc["weight"]), f))

    if not issues:
        _check_rigid_body(usable, fixed, load_cases, nx, ny, h, issues)
    if preserved.sum() > max_solid_elements(budget, nx * ny):
        issues.append(f"material_budget: preserved regions alone need "
                      f"{preserved.sum() / (nx * ny):.3f} of the envelope, more than the budget "
                      f"{budget:g}")

    settings = Settings(**{**Settings().__dict__, **settings_raw})
    if issues:
        raise SetupError(issues)
    return Problem(width, height, thickness, nx, ny, material, float(budget), float(min_member),
                   preserved, keep_out, fixed, load_cases, settings, raw)


def _check_keys(obj: Any, spec: dict[str, bool], name: str, issues: list[str]) -> bool:
    if not isinstance(obj, dict):
        issues.append(f"{name}: must be an object")
        return False
    ok = True
    for k in obj:
        if k not in spec:
            issues.append(f"{name}.{k}: unknown field")
            ok = False
    for k, required in spec.items():
        if required and k not in obj:
            issues.append(f"{name}.{k}: missing")
            ok = False
    return ok


def _positive(obj: dict[str, Any], key: str, name: str, issues: list[str]) -> float:
    v = obj[key]
    if not isinstance(v, (int, float)) or v <= 0:
        issues.append(f"{name}.{key}: must be a positive number (mm)")
        return 1.0
    return float(v)


def _region_mask(reg: Any, name: str, ecx: NDArray[np.float64], ecy: NDArray[np.float64],
                 issues: list[str], allow_circle: bool) -> NDArray[np.bool_] | None:
    if not isinstance(reg, dict) or reg.get("shape") not in (("rect", "circle") if allow_circle
                                                            else ("rect",)):
        issues.append(f"{name}.shape: must be " + ("'rect' or 'circle'" if allow_circle
                                                    else "'rect'"))
        return None
    if reg["shape"] == "rect":
        if not _check_keys(reg, {"shape": True, "x_mm": True, "y_mm": True, "width_mm": True,
                                 "height_mm": True}, name, issues):
            return None
        rect: NDArray[np.bool_] = (
            (ecx >= reg["x_mm"] - GEOM_TOL) & (ecx <= reg["x_mm"] + reg["width_mm"] + GEOM_TOL)
            & (ecy >= reg["y_mm"] - GEOM_TOL) & (ecy <= reg["y_mm"] + reg["height_mm"] + GEOM_TOL))
        return rect
    if not _check_keys(reg, {"shape": True, "cx_mm": True, "cy_mm": True, "r_mm": True},
                       name, issues):
        return None
    mask: NDArray[np.bool_] = ((ecx - reg["cx_mm"]) ** 2 + (ecy - reg["cy_mm"]) ** 2
                               <= reg["r_mm"] ** 2 + GEOM_TOL)
    return mask


def _apply_load(ld: Any, name: str, f: NDArray[np.float64], nx: int, ny: int, h: float,
                node_ok: NDArray[np.bool_], issues: list[str]) -> None:
    nnx = nx + 1
    if not isinstance(ld, dict) or ld.get("type") not in ("point", "edge"):
        issues.append(f"{name}.type: must be 'point' or 'edge'")
        return
    if ld["type"] == "point":
        if not _check_keys(ld, {"type": True, "x_mm": True, "y_mm": True, "fx_N": True,
                                "fy_N": True}, name, issues):
            return
        i, j = round(ld["x_mm"] / h), round(ld["y_mm"] / h)
        if not (0 <= i <= nx and 0 <= j <= ny):
            issues.append(f"{name}: point ({ld['x_mm']}, {ld['y_mm']}) mm is outside the envelope")
            return
        node = j * nnx + i
        if not node_ok[node]:
            issues.append(f"{name}: load at ({ld['x_mm']}, {ld['y_mm']}) mm lies inside a "
                          f"keep-out region")
            return
        f[2 * node] += ld["fx_N"]
        f[2 * node + 1] += ld["fy_N"]
        return
    if not _check_keys(ld, {"type": True, "edge": True, "from_mm": True, "to_mm": True,
                            "fx_N": True, "fy_N": True}, name, issues):
        return
    edge = ld["edge"]
    if edge not in ("left", "right", "bottom", "top"):
        issues.append(f"{name}.edge: must be left, right, bottom or top")
        return
    a, b = float(ld["from_mm"]), float(ld["to_mm"])
    if not b > a:
        issues.append(f"{name}: to_mm must be greater than from_mm")
        return
    count = ny if edge in ("left", "right") else nx
    # Consistent nodal forces for a uniform traction over [a, b] (linear shape functions).
    weights = np.zeros(count + 1)
    for s in range(count):
        lo, hi = max(a, s * h), min(b, (s + 1) * h)
        if hi <= lo:
            continue
        # integrate N_left = 1 - t, N_right = t over the covered part of segment s
        t0, t1 = (lo - s * h) / h, (hi - s * h) / h
        weights[s] += h * ((t1 - t1 ** 2 / 2) - (t0 - t0 ** 2 / 2))
        weights[s + 1] += h * (t1 ** 2 / 2 - t0 ** 2 / 2)
    if weights.sum() <= 0:
        issues.append(f"{name}: range [{a}, {b}] mm does not overlap the {edge} edge")
        return
    weights /= weights.sum()
    for s, w in enumerate(weights):
        if w == 0:
            continue
        i, j = {"left": (0, s), "right": (nx, s), "bottom": (s, 0), "top": (s, ny)}[edge]
        node = j * nnx + i
        if not node_ok[node]:
            issues.append(f"{name}: part of the edge load lies inside a keep-out region")
            return
        f[2 * node] += w * ld["fx_N"]
        f[2 * node + 1] += w * ld["fy_N"]


def _check_rigid_body(usable: NDArray[np.bool_], fixed: NDArray[np.bool_],
                      load_cases: list[LoadCase], nx: int, ny: int, h: float,
                      issues: list[str]) -> None:
    """Every piece of usable material that carries load must be held against the three
    in-plane rigid-body motions (two translations, one rotation) by its supports."""
    label = topology.components(usable, nx, ny)
    for comp in range(int(np.max(label, initial=-1)) + 1):
        nodes = topology.nodes_of(np.flatnonzero(label == comp), nx)
        names = [lc.name for lc in load_cases if topology.loaded(nodes, [lc.force])]
        if not names:
            continue
        rank = topology.support_rank(nodes, fixed, nx, h)
        if rank < 3:
            free = {0: "translate and rotate", 1: "translate or rotate", 2: "rotate"}.get(
                rank, "move")
            issues.append(f"supports: the material carrying load case(s) "
                          f"{', '.join(repr(n) for n in names)} can still {free} freely "
                          f"(rigid-body motion); add supports that hold it in x, y and rotation")
