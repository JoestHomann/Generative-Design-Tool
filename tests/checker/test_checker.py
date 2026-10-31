"""The standalone checker re-verifies a certificate from the file alone."""

from __future__ import annotations

import ast
import base64
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import gdt
import gdt_check

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from setups import bracket, cantilever, tension_bar  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def certificates() -> dict[str, dict[str, Any]]:
    out = {}
    for name, setup in {"cantilever": cantilever(nx=24, ny=12),
                        "bracket": bracket(nx=36, ny=18),
                        "full budget": tension_bar()}.items():
        # round-trip through JSON: the checker sees exactly what is written to disk
        out[name] = json.loads(json.dumps(gdt.optimize(setup).certificate))
    return out


@pytest.mark.parametrize("name", ["cantilever", "bracket", "full budget"])
def test_engine_certificates_pass(certificates: dict[str, dict[str, Any]], name: str) -> None:
    report = gdt_check.check(certificates[name])
    assert report.passed, report.summary()
    assert {c.name for c in report.checks} >= {
        "material budget", "preserved regions", "keep-out regions", "load path",
        "upper bound", "lower bound", "gap"}


def flip_first_solid_element(c: dict[str, Any]) -> None:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(c["design"]["data"]), dtype=np.uint8))
    bits[int(np.flatnonzero(bits)[0])] ^= 1
    c["design"]["data"] = base64.b64encode(np.packbits(bits).tobytes()).decode()


def perturb_relaxed_design(c: dict[str, Any]) -> None:
    x = np.frombuffer(base64.b64decode(c["relaxed_design"]["data"]), dtype="<f8").copy()
    x[np.argmax(x)] *= 0.5
    c["relaxed_design"]["data"] = base64.b64encode(x.astype("<f8").tobytes()).decode()


def scale(path: list[str], factor: float) -> Callable[[dict[str, Any]], None]:
    def apply(c: dict[str, Any]) -> None:
        obj = c
        for key in path[:-1]:
            obj = obj[key]
        obj[path[-1]] *= factor
    return apply


def change_load(c: dict[str, Any]) -> None:
    c["setup"]["load_cases"][0]["loads"][0]["fy_N"] *= 1.5


def shrink_budget(c: dict[str, Any]) -> None:
    c["setup"]["material_budget"] *= 0.9


def swap_material_id(c: dict[str, Any]) -> None:
    c["setup"]["material"]["id"] = "al-7075-t6"


def halve_mass(c: dict[str, Any]) -> None:
    c["design"]["mass_kg"] *= 0.5


def drop_lower_bound(c: dict[str, Any]) -> None:
    del c["bounds"]["lower_bound_Nmm"]


TAMPERING = {
    "flipped design element": (flip_first_solid_element, {"upper bound", "material budget",
                                                           "preserved regions", "load path"}),
    "upper bound lowered": (scale(["bounds", "upper_bound_Nmm"], 0.99), {"upper bound"}),
    "lower bound raised": (scale(["bounds", "lower_bound_Nmm"], 1.01), {"lower bound"}),
    "gap shrunk": (scale(["bounds", "gap"], 0.5), {"gap"}),
    "relaxed design perturbed": (perturb_relaxed_design, {"lower bound"}),
    "load changed": (change_load, {"upper bound"}),
    "budget shrunk": (shrink_budget, {"material budget", "lower bound"}),
    "field removed": (drop_lower_bound, {"schema"}),
    "material id swapped": (swap_material_id, {"material"}),
    "mass halved": (halve_mass, {"design summary"}),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERING))
def test_tampered_certificate_fails_with_the_matching_check(
        certificates: dict[str, dict[str, Any]], tamper: str) -> None:
    apply, expected = TAMPERING[tamper]
    cert = copy.deepcopy(certificates["bracket"])
    apply(cert)
    report = gdt_check.check(cert)
    assert not report.passed
    failed = {c.name for c in report.checks if not c.passed}
    assert failed & expected, f"{tamper}: failed checks {failed}, expected one of {expected}"


def test_checker_never_imports_the_engine() -> None:
    for source in (REPO / "checker" / "src").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            assert not any(n == "gdt" or n.startswith("gdt.") for n in names), source


def run_checker_cli(path: Path) -> subprocess.CompletedProcess[str]:
    # only the checker on the path: proves it runs without the engine installed
    env = {"PYTHONPATH": str(REPO / "checker" / "src")}
    return subprocess.run([sys.executable, "-m", "gdt_check", str(path)], capture_output=True,
                          text=True, env=env)


def test_command_line_verdict_and_exit_code(certificates: dict[str, dict[str, Any]],
                                            tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(certificates["cantilever"]))
    done = run_checker_cli(good)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASS" in done.stdout

    bad_cert = copy.deepcopy(certificates["cantilever"])
    bad_cert["bounds"]["upper_bound_Nmm"] *= 0.9
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bad_cert))
    done = run_checker_cli(bad)
    assert done.returncode == 1
    assert "FAIL" in done.stdout and "upper bound" in done.stdout


def test_material_values_are_checked_against_a_library(
        certificates: dict[str, dict[str, Any]]) -> None:
    """A forger who doubles E and halves every bound keeps the certificate self-consistent;
    only the material library can expose it."""
    cert = copy.deepcopy(certificates["bracket"])
    cert["material"]["youngs_modulus_MPa"] *= 2
    for key in ("upper_bound_Nmm", "lower_bound_Nmm", "upper_bound_adjusted_Nmm",
                "lower_bound_adjusted_Nmm"):
        cert["bounds"][key] /= 2
    cert["relaxed_design"]["relaxed_compliance_Nmm"] /= 2
    library = REPO / "data" / "materials.json"
    report = gdt_check.check(cert, material_library=library)
    assert not report.passed
    assert "material" in {c.name for c in report.checks if not c.passed}
    assert gdt_check.check(certificates["bracket"], material_library=library).passed


def test_report_says_when_material_values_were_not_verified(
        certificates: dict[str, dict[str, Any]]) -> None:
    report = gdt_check.check(certificates["bracket"])
    material = next(c for c in report.checks if c.name == "material")
    assert "not verified" in material.detail
