"""NFR-05: a 200 x 100 problem is certified in under 60 s on the reference machine
(AMD Ryzen 7 9800X3D). Slow: run with  pytest -m slow"""

import time

import pytest

import gdt
from setups import bracket


@pytest.mark.slow
def test_200_by_100_bracket_is_certified_within_a_minute() -> None:
    start = time.perf_counter()
    result = gdt.optimize(bracket(nx=200, ny=100))
    elapsed = time.perf_counter() - start
    assert result.certificate["bounds"]["status"] == "certified"
    assert elapsed < 60.0, f"took {elapsed:.1f} s"
