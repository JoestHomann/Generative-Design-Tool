"""Standalone checker for Generative Design Tool certificates.

Shares no code with the engine: it re-derives the problem from the setup embedded in the
certificate and recomputes feasibility, the upper bound, the lower bound and the gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .verify import Check, Report
from .verify import check as _check

__all__ = ["Check", "Report", "check"]
__version__ = "0.1.0"


def check(certificate: dict[str, Any] | str | Path, *,
          material_library: str | Path | None = None) -> Report:
    """Re-verify a certificate (a dict, or a path to the certificate JSON file).

    Pass the material library to also verify the material values the certificate used;
    without it the report says that the material values were not verified."""
    library = None
    if material_library is not None:
        with open(material_library, encoding="utf-8") as fh:
            library = json.load(fh)
    if isinstance(certificate, dict):
        return _check(certificate, library)
    with open(certificate, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return _check(data, library)
