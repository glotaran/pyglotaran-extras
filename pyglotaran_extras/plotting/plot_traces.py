"""Module containing functionality to plot fitted traces."""
from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import matplotlib.pyplot as plt

from pyglotaran_extras.io.utils import result_dataset_mapping
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import MinorSymLogLocator
from pyglotaran_extras.plotting.utils import PlotDuplicationWarning
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import add_unique_figure_legend
from pyglotaran_extras.plotting.utils import extract_dataset_scale
from pyglotaran_extras.plotting.utils import extract_irf_location
from pyglotaran_extras.plotting.utils import select_plot_wavelengths

__all__ = ["select_plot_wavelengths", "plot_fitted_traces"]

if TYPE_CHECKING:
    from typing import Iterable

    from cycler import Cycler
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import ResultLike


def plot_data_and_fits(
    result: ResultLike,
    wavelength: float,
    axis: Axis,
    center_λ: float | None = None,
    main_irf_nr: int = 0,
    linlog: bool = False,
    linthresh: float = 1,
    divide_by_scale: bool = True,
    per_axis_legend: bool = False,
    y_label: str = "a.u.",
    cycler: Cycler | None = PlotStyle().data_cycler_solid,
    show_zero_line: bool = True,
) -> None:
    """Plot data and fits for a given ``wavelength`` on a given ``axis``.

    If the wavelength isn't part of a dataset, that dataset will be skipped.

    Parameters
    ----------
    result : ResultLike
        Data structure which can be converted to a mapping.
    wavelength : float
        Wavelength to plot data and fits for.
    axis: Axis
        Axis to plot the data and fits on.
    center_λ: float | None
        Center wavelength (λ in nm)
    main_irf_nr : int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks. Defaults to 0.
    linlog : bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    divide_by_scale : bool
        Whether or not to divide the data by the dataset scale used for optimization.
        Defaults to True.
    per_axis_legend: bool
        Whether to use a legend per plot or for the whole figure. Defaults to False.
    y_label: str
        Label used for the y-axis of each subplot.
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().data_cycler_solid.
    show_zero_line: bool
        Whether or not to add a horizontal line at zero. Defaults to True.

    See Also
    --------
    plot_fit_overview
    """
    result_map = result_dataset_mapping(result)
    add_cycler_if_not_none(axis, cycler)
    for dataset_name in result_map.keys():
        if result_map[dataset_name].coords["time"].values.size == 1:
            continue
        spectral_coords = result_map[dataset_name].coords["spectral"].values
        if spectral_coords.min() <= wavelength <= spectral_coords.max():
            result_data = result_map[dataset_name].sel(spectral=[wavelength], method="nearest")
            scale = extract_dataset_scale(result_data, divide_by_scale)
            irf_loc = extract_irf_location(result_data, center_λ, main_irf_nr)
            result_data = result_data.assign_coords(time=result_data.coords["time"] - irf_loc)
            (result_data.data / scale).plot(x="time", ax=axis, label=f"{dataset_name}_data")
            (result_data.fitted_data / scale).plot(x="time", ax=axis, label=f"{dataset_name}_fit")
        else:
            [next(axis._get_lines.prop_cycler) for _ in range(2)]
    if linlog:
        axis.set_xscale("symlog", linthresh=linthresh)
        axis.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))
    if show_zero_line is True:
        axis.axhline(0, color="k", linewidth=1)
    axis.set_ylabel(y_label)
    if per_axis_legend is True:
        axis.legend()


def plot_fitted_traces(
    result: ResultLike,
    wavelengths: Iterable[float],
    axes_shape: tuple[int, int] = (4, 4),
    center_λ: float | None = None,
    main_irf_nr: int = 0,
    linlog: bool = False,
    linthresh: float = 1,
    divide_by_scale: bool = True,
    per_axis_legend: bool = False,
    figsize: tuple[float, float] = (30, 15),
    title: str = "Fit overview",
    y_label: str = "a.u.",
    cycler: Cycler | None = PlotStyle().data_cycler_solid,
    show_zero_line: bool = True,
) -> tuple[Figure, Axes]:
    """Plot data and their fit in per wavelength plot grid.

    Parameters
    ----------
    result : ResultLike
        Data structure which can be converted to a mapping of datasets.
    axes_shape : tuple[int, int]
        Shape of the plot grid (N, M). Defaults to (4, 4).
    wavelengths: Iterable[float]
        Wavelength which should be used for each subplot, should to be of length N*M
        with ``axes_shape`` being of shape (N, M), else it will result in missing plots.
    center_λ: float | None
        Center wavelength of the IRF (λ in nm).
    main_irf_nr : int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks. Defaults to 0.
    linlog : bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    divide_by_scale : bool
        Whether or not to divide the data by the dataset scale used for optimization.
        Defaults to True.
    per_axis_legend : bool
        Whether to use a legend per plot or for the whole figure. Defaults to False.
    figsize : tuple[int, int]
        Size of the figure (N, M) in inches. Defaults to (30, 15).
    title : str
        Title to add to the figure. Defaults to "Fit overview".
    y_label: str
        Label used for the y-axis of each subplot.
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().data_cycler_solid.
    show_zero_line: bool
        Whether or not to add a horizontal line at zero. Defaults to True.


    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes which can then be refined by the user.

    See Also
    --------
    maximum_coordinate_range
    add_unique_figure_legend
    plot_data_and_fits
    calculate_wavelengths
    """
    result_map = result_dataset_mapping(result)
    fig, axes = plt.subplots(*axes_shape, figsize=figsize)
    nr_of_plots = len(axes.flatten())
    max_spectral_values = max(
        len(result_map[dataset_name].coords["spectral"]) for dataset_name in result_map.keys()
    )
    if nr_of_plots > max_spectral_values:
        warn(
            PlotDuplicationWarning(
                f"The number of plots ({nr_of_plots}) exceeds the maximum number of "
                f"spectral data points ({max_spectral_values}), "
                "which will lead in duplicated plots."
            ),
            stacklevel=2,
        )
    for wavelength, axis in zip(wavelengths, axes.flatten()):
        plot_data_and_fits(
            result=result_map,
            wavelength=wavelength,
            axis=axis,
            center_λ=center_λ,
            main_irf_nr=main_irf_nr,
            linlog=linlog,
            linthresh=linthresh,
            divide_by_scale=divide_by_scale,
            per_axis_legend=per_axis_legend,
            y_label=y_label,
            cycler=cycler,
            show_zero_line=show_zero_line,
        )
    if per_axis_legend is False:
        add_unique_figure_legend(fig, axes)
    fig.suptitle(title, fontsize=28)
    fig.tight_layout()
    return fig, axes
