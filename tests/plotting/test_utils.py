"""Tests for pyglotaran_extras.plotting.utils"""
from __future__ import annotations

from typing import Hashable
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import pytest
import xarray as xr
from cycler import Cycler
from cycler import cycle

from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import abs_max
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none

matplotlib.use("Agg")
DEFAULT_CYCLER = plt.rcParams["axes.prop_cycle"]


@pytest.mark.parametrize(
    "cycler,expected_cycler",
    ((None, DEFAULT_CYCLER()), (PlotStyle().cycler, PlotStyle().cycler())),
)
def test_add_cycler_if_not_none(cycler: Cycler | None, expected_cycler: cycle):
    """Default cycler inf None and cycler otherwise"""
    ax = plt.subplot()
    add_cycler_if_not_none(ax, cycler)

    for _ in range(10):
        assert next(ax._get_lines.prop_cycler) == next(expected_cycler)


@pytest.mark.parametrize(
    "result_dims, expected",
    (
        ((), xr.DataArray(40)),
        ("dim1", xr.DataArray([20, 40], coords={"dim1": [1, 2]})),
        ("dim2", xr.DataArray([30, 40], coords={"dim2": [3, 4]})),
        (("dim1",), xr.DataArray([20, 40], coords={"dim1": [1, 2]})),
        (
            ("dim1", "dim2"),
            xr.DataArray([[10, 20], [30, 40]], coords={"dim1": [1, 2], "dim2": [3, 4]}),
        ),
    ),
)
def test_abs_max(result_dims: Hashable | Iterable[Hashable], expected: xr.DataArray):
    """Result values are positive and dimensions are preserved if result_dims is not empty."""
    data = xr.DataArray([[-10, 20], [-30, 40]], coords={"dim1": [1, 2], "dim2": [3, 4]})
    assert abs_max(data, result_dims=result_dims).equals(expected)
