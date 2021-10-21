from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import matplotlib.pyplot as plt

from pyglotaran_extras.io.utils import result_dataset_mapping
from pyglotaran_extras.plotting.plot_svd import plot_lsv_data
from pyglotaran_extras.plotting.plot_svd import plot_rsv_data
from pyglotaran_extras.plotting.plot_svd import plot_sv_data
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import PlotDuplicationWarning
from pyglotaran_extras.plotting.utils import add_unique_figure_legend
from pyglotaran_extras.plotting.utils import extract_irf_location

if TYPE_CHECKING:
    from typing import Iterable

    from cycler import Cycler
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes
    from xarray import Dataset

    from pyglotaran_extras.types import ResultLike


def plot_data_overview(
    dataset: Dataset,
    title="Data overview",
    linlog: bool = False,
    linthresh: float = 1,
    figsize: tuple[int, int] = (30, 15),
) -> tuple[Figure, Axes]:
    """Plot data as filled contour plot and SVD components.

    Parameters
    ----------
    dataset : Dataset
        Dataset containing data and SVD of the data.
    title : str, optional
        Title to add to the figure., by default "Data overview"
    linlog : bool, optional
        Whether to use 'symlog' scale or not, by default False
    linthresh : float, optional
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero., by default 1

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes which can then be refined by the user.
    """
    fig = plt.figure(figsize=figsize)
    data_ax = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3, fig=fig)
    lsv_ax = plt.subplot2grid((4, 3), (3, 0), fig=fig)
    sv_ax = plt.subplot2grid((4, 3), (3, 1), fig=fig)
    rsv_ax = plt.subplot2grid((4, 3), (3, 2), fig=fig)

    if len(dataset.data.time) > 1:
        dataset.data.plot(x="time", ax=data_ax, center=False)
    else:
        dataset.data.plot(ax=data_ax)
    plot_lsv_data(dataset, lsv_ax)
    plot_sv_data(dataset, sv_ax)
    plot_rsv_data(dataset, rsv_ax)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    if linlog:
        data_ax.set_xscale("symlog", linthresh=linthresh)
    return fig, (data_ax, lsv_ax, sv_ax, rsv_ax)


def plot_data_and_fits(
    result: ResultLike,
    wavelength: float,
    axis: Axes,
    center_λ: float | None = None,
    main_irf_nr: int = 0,
    linlog: bool = False,
    linthresh: float = 1,
    per_axis_legend: bool = False,
    cycler: Cycler = PlotStyle().data_cycler_solid,
) -> None:
    """Plot data and fits for a given ``wavelength`` on a given ``axis``.

    If the wavelength isn't part of a dataset, that dataset will be skipped.

    Parameters
    ----------
    result : ResultLike
        Data structure which can be converted to a mapping.
    wavelength : float
        Wavelength to plot data and fits for.
    axis : Axes
        Axis to plot the data and fits on.
    center_λ: float | None
        Center wavelength (λ in nm)
    main_irf_nr : int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks , by default 0
    linlog : bool
        Whether to use 'symlog' scale or not, by default False
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero., by default 1
    per_axis_legend: bool
        Whether to use a legend per plot or for the whole figure., by default False
    cycler : Cycler
        Plot style cycler to use., by default PlotStyle().data_cycler_solid

    See Also
    --------
    plot_fit_overview
    """
    result_map = result_dataset_mapping(result)
    axis.set_prop_cycle(cycler)
    for dataset_name in result_map.keys():
        spectral_coords = result_map[dataset_name].coords["spectral"].values
        if spectral_coords.min() <= wavelength <= spectral_coords.max():
            result_data = result_map[dataset_name].sel(spectral=[wavelength], method="nearest")
            irf_loc = extract_irf_location(result_data, center_λ, main_irf_nr)
            result_data = result_data.assign_coords(time=result_data.coords["time"] - irf_loc)
            result_data.data.plot(x="time", ax=axis, label=f"{dataset_name}_data")
            result_data.fitted_data.plot(x="time", ax=axis, label=f"{dataset_name}_fit")
        else:
            [next(axis._get_lines.prop_cycler) for _ in range(2)]
    if linlog:
        axis.set_xscale("symlog", linthresh=linthresh)
    axis.set_ylabel("Intensity")
    if per_axis_legend is True:
        axis.legend()


def plot_fit_overview(
    result: ResultLike,
    wavelengths: Iterable[float],
    axes_shape: tuple[int, int] = (4, 4),
    center_λ: float | None = None,
    main_irf_nr: int = 0,
    linlog: bool = False,
    linthresh: float = 1,
    per_axis_legend: bool = False,
    figsize: tuple[int, int] = (30, 15),
    title: str = "Fit overview",
    cycler: Cycler = PlotStyle().data_cycler_solid,
) -> tuple[Figure, Axes]:
    """Plot data and their fit in per wavelength plot grid.

    Parameters
    ----------
    result : ResultLike
        Data structure which can be converted to a mapping of datasets.
    axes_shape : tuple[int, int]
        Shape of the plot grid (N, M), by default (4, 4)
    wavelengths: Iterable[float]
        Wavelength which should be used for each subplot, should to be of length N*M
        with ``axes_shape`` being of shape (N, M), else it will result in missing plots.
    center_λ: float | None
        Center wavelength of the IRF (λ in nm).
    main_irf_nr : int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks , by default 0
    linlog : bool
        Whether to use 'symlog' scale or not, by default False
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero., by default 1
    per_axis_legend : bool
        Whether to use a legend per plot or for the whole figure., by default False
    figsize : tuple[int, int]
        Size of the figure (N, M) in inches., by default (30, 15)
    title : str
        Title to add to the figure., by default "Fit overview"
    cycler : Cycler
        Plot style cycler to use., by default PlotStyle().data_cycler_solid

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
            per_axis_legend=per_axis_legend,
            cycler=cycler,
        )
    if per_axis_legend is False:
        add_unique_figure_legend(fig, axes)
    fig.suptitle(title, fontsize=28)
    fig.tight_layout()
    return fig, axes
