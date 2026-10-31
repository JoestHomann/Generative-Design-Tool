"""Material library access: resolves a setup's material entry to engine values."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_LIBRARY = Path(__file__).resolve().parents[3] / "data" / "materials.json"


@dataclass(frozen=True)
class Allowables:
    """Strength values (MPa) the stress check compares against."""

    yield_MPa: float
    ultimate_MPa: float
    basis: str
    description: str
    is_design_allowable: bool

    def to_dict(self) -> dict[str, Any]:
        return {"yield_MPa": self.yield_MPa, "ultimate_MPa": self.ultimate_MPa,
                "basis": self.basis, "description": self.description,
                "is_design_allowable": self.is_design_allowable}


@dataclass(frozen=True)
class Material:
    id: str
    name: str
    youngs_modulus_MPa: float
    poissons_ratio: float
    density_kg_m3: float
    allowables: Allowables | None
    sources: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class MaterialError(ValueError):
    pass


def load_library(path: Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_LIBRARY, encoding="utf-8") as fh:
        library: dict[str, Any] = json.load(fh)
    return library


def resolve(spec: dict[str, Any], library: dict[str, Any]) -> Material:
    """Turn a setup's ``material`` entry into a Material.

    ``spec`` is either ``{"id": ..., "allowable": {...}?}`` or ``{"custom": {...}}``.
    """
    if "custom" in spec:
        return _custom(spec["custom"])
    mid = spec.get("id")
    entries = {m["id"]: m for m in library["materials"]}
    if mid not in entries:
        raise MaterialError(f"material: unknown id '{mid}'. Known ids: {', '.join(sorted(entries))}")
    m = entries[mid]
    allowables, warnings = _pick_allowables(m, spec.get("allowable"))
    used_sources = {m[k]["source"] for k in ("density", "youngs_modulus", "poissons_ratio")}
    sources = {s: library["sources"][s] for s in sorted(used_sources) if s in library["sources"]}
    return Material(
        id=m["id"], name=m["name"],
        youngs_modulus_MPa=float(m["youngs_modulus"]["value"]) * 1000.0,
        poissons_ratio=float(m["poissons_ratio"]["value"]),
        density_kg_m3=float(m["density"]["value"]),
        allowables=allowables, sources=sources, warnings=tuple(warnings),
    )


def _custom(c: dict[str, Any]) -> Material:
    required = ("name", "youngs_modulus_GPa", "poissons_ratio", "density_kg_m3")
    missing = [k for k in required if k not in c]
    if missing:
        raise MaterialError(f"material.custom: missing {', '.join(missing)}")
    allowables = None
    warnings = []
    if "yield_MPa" in c and "ultimate_MPa" in c:
        allowables = Allowables(float(c["yield_MPa"]), float(c["ultimate_MPa"]), "user",
                                "custom material, user-supplied strengths", False)
        warnings.append("Custom material: strengths are user-supplied, not design allowables.")
    else:
        warnings.append("Custom material has no strengths; stress check margins are not available.")
    return Material(
        id="custom", name=str(c["name"]),
        youngs_modulus_MPa=float(c["youngs_modulus_GPa"]) * 1000.0,
        poissons_ratio=float(c["poissons_ratio"]), density_kg_m3=float(c["density_kg_m3"]),
        allowables=allowables, sources={"custom": {"citation": "user-defined in setup file"}},
        warnings=tuple(warnings),
    )


def _pick_allowables(m: dict[str, Any], override: dict[str, Any] | None
                     ) -> tuple[Allowables | None, list[str]]:
    entries = m.get("design_allowables")
    if entries:
        if override:
            matches = [e for e in entries
                       if all(str(e.get(k)) == str(v) for k, v in override.items())]
            if not matches:
                raise MaterialError(
                    f"material.allowable: no entry of '{m['id']}' matches {override}")
            e = matches[0]
        else:
            e = entries[m.get("default_allowable", 0)]
        desc = (f"{e['source']} {e['table']}: {e['product_form']}, {e['condition']}, "
                f"{e['thickness_in']} in, {e['basis']}-basis, min of L/LT")
        warn = ["Allowables from MIL-HDBK-5J (superseded by MMPDS; not valid for certification)."]
        return Allowables(float(e["Fty_min_MPa"]), float(e["Ftu_min_MPa"]), e["basis"], desc,
                          True), warn
    typical = m.get("strength_typical") or []
    if typical:
        y = min(float(t["Rp02_MPa"]) for t in typical)
        u = min(float(t["Rm_MPa"]) for t in typical)
        return Allowables(y, u, "typical", "lowest typical value over listed conditions and "
                          "orientations", False), [
            f"{m['name']} has no design allowables; margins use typical values."]
    return None, [f"{m['name']} has no strength data; margins are not available."]
