"""Generative Design Tool engine: 2D topology optimization with a certified optimality gap."""

from ._version import __version__
from .api import Result, optimize, run
from .setup import SetupError

__all__ = ["__version__", "Result", "SetupError", "optimize", "run"]
