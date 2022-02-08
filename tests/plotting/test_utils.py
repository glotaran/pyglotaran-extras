"""Tests for pyglotaran_extras.plotting.utils"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
from cycler import Cycler
from cycler import cycle

from pyglotaran_extras.plotting.style import PlotStyle
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
