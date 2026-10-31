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


def check(certificate: dict[str, Any] | str | Path) -> Report:
    """Re-verify a certificate (a dict, or a path to the certificate JSON file)."""
    if isinstance(certificate, dict):
        return _check(certificate)
    with open(certificate, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return _check(data)
