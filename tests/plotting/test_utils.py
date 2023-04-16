"""Tests for pyglotaran_extras.plotting.utils"""
from __future__ import annotations

from typing import Hashable
from typing import Iterable
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from cycler import Cycler
from cycler import cycle

from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import abs_max
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import add_subplot_labels
from pyglotaran_extras.plotting.utils import calculate_ticks_in_units_of_pi
from pyglotaran_extras.plotting.utils import ensure_axes_array
from pyglotaran_extras.plotting.utils import format_sub_plot_number_upper_case_letter
from pyglotaran_extras.plotting.utils import not_single_element_dims
from pyglotaran_extras.types import SubPlotLabelCoord

matplotlib.use("Agg")
DEFAULT_CYCLER = plt.rcParams["axes.prop_cycle"]


@pytest.mark.parametrize(
    "cycler,expected_cycler",
    ((None, DEFAULT_CYCLER()), (PlotStyle().cycler, PlotStyle().cycler())),
)
def test_add_cycler_if_not_none_single_axis(cycler: Cycler | None, expected_cycler: cycle):
    """Default cycler if None and cycler otherwise on a single axis"""
    ax = plt.subplot()
    add_cycler_if_not_none(ax, cycler)

    for _ in range(10):
        expected = next(expected_cycler)
        assert next(ax._get_lines.prop_cycler) == expected


@pytest.mark.parametrize(
    "cycler,expected_cycler",
    ((None, DEFAULT_CYCLER()), (PlotStyle().cycler, PlotStyle().cycler())),
)
def test_add_cycler_if_not_none_multiple_axes(cycler: Cycler | None, expected_cycler: cycle):
    """Default cycler if None and cycler otherwise on all axes"""
    _, axes = plt.subplots(1, 2)
    add_cycler_if_not_none(axes, cycler)

    for _ in range(10):
        expected = next(expected_cycler)
        assert next(axes[0]._get_lines.prop_cycler) == expected
        assert next(axes[1]._get_lines.prop_cycler) == expected


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


@pytest.mark.parametrize(
    "step_size, expected_tick_values,expected_tick_labels",
    (
        (0.5, np.linspace(-np.pi, 2 * np.pi, num=7), ["-1", "-0.5", "0", "0.5", "1", "1.5", "2"]),
        (1, np.linspace(-np.pi, 2 * np.pi, num=4), ["-1", "0", "1", "2"]),
    ),
)
def test_calculate_ticks_in_units_of_pi(
    step_size: float, expected_tick_values: list[float], expected_tick_labels: list[str]
):
    """Different values depending on ``step_size``."""
    values = np.linspace(-np.pi, 2 * np.pi)
    tick_values, tick_labels = calculate_ticks_in_units_of_pi(values, step_size=step_size)

    assert np.allclose(list(tick_values), expected_tick_values)
    assert list(tick_labels) == expected_tick_labels


@pytest.mark.parametrize(
    "data_array, expected",
    (
        (xr.DataArray([1]), []),
        (xr.DataArray([1], coords={"dim1": [1]}), []),
        (xr.DataArray([[1], [1]], coords={"dim1": [1, 2], "dim2": [1]}), ["dim1"]),
        (
            xr.DataArray(
                [[[1, 1]], [[1, 1]]], coords={"dim1": [1, 2], "dim2": [1], "dim3": [1, 2]}
            ),
            ["dim1", "dim3"],
        ),
    ),
)
def test_not_single_element_dims(data_array: xr.DataArray, expected: list[Hashable]):
    """Only get dim with more than one element."""
    assert not_single_element_dims(data_array) == expected


@pytest.mark.parametrize(
    ("value", "size", "expected"),
    (
        (1, None, "A"),
        (2, None, "B"),
        (26, None, "Z"),
        (27, None, "AA"),
        (26**2 + 26, None, "ZZ"),
        (1, 26**2, "AA"),
        (2, 26**2, "AB"),
        (26, 26**2, "AZ"),
        (26**2, 26**2, "ZZ"),
        (1, 26**3, "AAA"),
        (26**3, 26**3, "ZZZ"),
    ),
)
def test_format_sub_plot_number_upper_case_letter(value: int, size: int | None, expected: str):
    """Expected string format."""
    assert format_sub_plot_number_upper_case_letter(value, size) == expected


def test_ensure_axes_array():
    """Hasa flatten method."""
    _, ax = plt.subplots(1, 1)
    assert hasattr(ax, "flatten") is False
    assert hasattr(ensure_axes_array(ax), "flatten") is True

    _, axes = plt.subplots(1, 2)
    assert hasattr(axes, "flatten") is True
    assert hasattr(ensure_axes_array(axes), "flatten") is True


def test_add_subplot_labels_defaults():
    """Sanity check that default arguments got passed on to mpl annotate method."""
    _, axes = plt.subplots(2, 2)

    add_subplot_labels(axes)

    assert [ax.texts[0].get_text() for ax in axes.flatten()] == ["1", "2", "3", "4"]
    assert [ax.texts[0].get_position() for ax in axes.flatten()] == pytest.approx(
        [(-0.05, 1.05)] * 4
    )
    assert [ax.texts[0].get_anncoords() for ax in axes.flatten()] == ["axes fraction"] * 4
    assert [ax.texts[0].get_fontsize() for ax in axes.flatten()] == [16] * 4


@pytest.mark.parametrize(
    "direction, expected", (("row", ["1", "2", "3", "4"]), ("column", ["1", "3", "2", "4"]))
)
@pytest.mark.parametrize("label_position", ((0.01, 0.95), (-0.1, 1.0)))
@pytest.mark.parametrize("label_coords", ("data", ("axes fraction", "data")))
@pytest.mark.parametrize("fontsize", (12, 26))
def test_add_subplot_labels_assignment(
    direction: Literal["row", "column"],
    label_position: tuple[float, float],
    label_coords: SubPlotLabelCoord,
    fontsize: int,
    expected: list[str],
):
    """Test basic label text assignment."""
    _, axes = plt.subplots(2, 2)

    add_subplot_labels(
        axes,
        label_position=label_position,
        label_coords=label_coords,
        direction=direction,
        fontsize=fontsize,
    )

    assert [ax.texts[0].get_text() for ax in axes.flatten()] == expected
    assert [ax.texts[0].get_position() for ax in axes.flatten()] == pytest.approx(
        [label_position] * 4
    )
    assert [ax.texts[0].get_anncoords() for ax in axes.flatten()] == [label_coords] * 4
    assert [ax.texts[0].get_fontsize() for ax in axes.flatten()] == [fontsize] * 4

    plt.close()


@pytest.mark.parametrize("label_format_template, expected", (("{})", "1)"), ("({})", "(1)")))
def test_add_subplot_labels_label_format_template(label_format_template: str, expected: str):
    """Template is used."""
    _, ax = plt.subplots(1, 1)
    add_subplot_labels(ax, label_format_template=label_format_template)

    assert ax.texts[0].get_text() == expected
