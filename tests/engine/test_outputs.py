"""A run writes the certificate, a readable report, an image and the element grid."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import gdt
from setups import cantilever, with_changes

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, gdt.Result]:
    out = tmp_path_factory.mktemp("run")
    return out, gdt.run(cantilever(nx=30, ny=15), out)


def test_certificate_file_matches_result(run_dir: tuple[Path, gdt.Result]) -> None:
    out, result = run_dir
    cert = json.loads((out / "certificate.json").read_text(encoding="utf-8"))
    assert cert == json.loads(json.dumps(result.certificate))
    assert cert["setup"] == cantilever(nx=30, ny=15)


def test_report_states_gap_target_and_uncertified_stress_check(
        run_dir: tuple[Path, gdt.Result]) -> None:
    out, result = run_dir
    report = (out / "report.md").read_text(encoding="utf-8")
    gap = result.certificate["bounds"]["gap_percent"]
    assert f"{gap:.2f} %" in report
    assert "target" in report.lower()
    assert "not part of the certificate" in report.lower()
    assert "MIL-HDBK-5J" in report
    assert "design.png" in report


def test_image_is_a_png(run_dir: tuple[Path, gdt.Result]) -> None:
    out, _ = run_dir
    data = (out / "design.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_element_grid_has_one_row_per_element_row(run_dir: tuple[Path, gdt.Result]) -> None:
    out, result = run_dir
    rows = (out / "design_grid.txt").read_text(encoding="utf-8").split()
    assert len(rows) == 15 and all(len(r) == 30 for r in rows)
    # top line is the top row of the part
    assert rows[0] == "".join("1" if v else "0" for v in result.design.reshape(15, 30)[-1])


def test_interrupted_run_still_returns_a_certified_design() -> None:
    def stop_early(iteration: int, compliance: float, volume: float, change: float) -> None:
        if iteration == 5:
            raise KeyboardInterrupt

    result = gdt.optimize(cantilever(nx=30, ny=15), progress=stop_early)
    assert result.certificate["optimizer"]["interrupted"] is True
    assert result.certificate["bounds"]["status"] == "certified"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {"PYTHONPATH": str(REPO / "engine" / "src")}
    return subprocess.run([sys.executable, "-m", "gdt", *args], capture_output=True, text=True,
                          env=env, cwd=REPO)


def test_command_line_runs_a_setup_file_and_shows_progress(tmp_path: Path) -> None:
    setup_file = tmp_path / "setup.json"
    setup_file.write_text(json.dumps(cantilever(nx=20, ny=10)))
    done = run_cli(str(setup_file), "--out", str(tmp_path / "out"))
    assert done.returncode == 0, done.stderr
    assert "iteration" in done.stdout.lower()
    assert "gap" in done.stdout.lower()
    assert (tmp_path / "out" / "certificate.json").exists()


def test_command_line_reports_setup_problems(tmp_path: Path) -> None:
    setup_file = tmp_path / "setup.json"
    setup_file.write_text(json.dumps(with_changes(cantilever(), supports=[])))
    done = run_cli(str(setup_file), "--out", str(tmp_path / "out"))
    assert done.returncode == 2
    assert "supports" in done.stderr
