"""Public entry points of the engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import binarize, certificate, fem, image, materials, optimizer, report, setup


@dataclass
class Result:
    certificate: dict[str, Any]
    design: NDArray[np.bool_]  # binary design, element index ey * nx + ex
    problem: setup.Problem


def optimize(source: dict[str, Any] | str | Path, *,
             material_library: str | Path | None = None,
             progress: optimizer.Progress | None = None) -> Result:
    """Validate the setup, optimize, binarize, verify and build the certificate.

    Raises SetupError (with every issue found) if the setup cannot be solved. ``progress``
    is called after each iteration with (iteration, compliance, volume fraction, change);
    raising KeyboardInterrupt from it, or pressing Ctrl+C, stops the optimization early and
    certifies the best design so far.
    """
    raw = setup.load(source)
    library = materials.load_library(Path(material_library) if material_library else None)
    problem = setup.parse(raw, library)
    model = fem.Model(problem)
    usable = ~problem.keep_out
    if problem.max_solid >= int(usable.sum()):
        opt = None
        binary = binarize.clean(problem, usable)
    else:
        opt = optimizer.run(problem, model, progress)
        binary = binarize.binarize(problem, model, opt.density)
    cert = certificate.build(problem, model, binary, opt)
    return Result(cert, binary.design, problem)


def run(source: dict[str, Any] | str | Path, out_dir: str | Path, *,
        material_library: str | Path | None = None,
        progress: optimizer.Progress | None = None) -> Result:
    """optimize() and write certificate.json, report.md, design.png and design_grid.txt."""
    result = optimize(source, material_library=material_library, progress=progress)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "certificate.json").write_text(json.dumps(result.certificate, indent=2,
                                                     ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(report.markdown(result.certificate), encoding="utf-8")
    image.write_png(out / "design.png", image.render(result.problem, result.design))
    p = result.problem
    grid = result.design.reshape(p.ny, p.nx)[::-1]
    (out / "design_grid.txt").write_text(
        "\n".join("".join("1" if v else "0" for v in row) for row in grid) + "\n",
        encoding="utf-8")
    return result
