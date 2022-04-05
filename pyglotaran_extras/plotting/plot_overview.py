"""Module containing overview plotting functionality."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import matplotlib.pyplot as plt

from pyglotaran_extras.deprecation.deprecation_utils import FIG_ONLY_WARNING
from pyglotaran_extras.deprecation.deprecation_utils import PyglotaranExtrasApiDeprecationWarning
from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.plotting.plot_concentrations import plot_concentrations
from pyglotaran_extras.plotting.plot_residual import plot_residual
from pyglotaran_extras.plotting.plot_spectra import plot_spectra
from pyglotaran_extras.plotting.plot_svd import plot_svd
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none

if TYPE_CHECKING:
    from cycler import Cycler
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import DatasetConvertible


def plot_overview(
    result: DatasetConvertible,
    center_λ: float | None = None,
    linlog: bool = True,
    linthresh: float = 1,
    linscale: float = 1,
    show_data: bool = False,
    main_irf_nr: int = 0,
    figsize: tuple[int, int] = (18, 16),
    cycler: Cycler | None = PlotStyle().cycler,
    figure_only: bool = True,
    nr_of_data_svd_vectors: int = 4,
    nr_of_residual_svd_vectors: int = 2,
    show_data_svd_legend: bool = True,
    show_residual_svd_legend: bool = True,
) -> Figure | tuple[Figure, Axes]:
    """Plot overview of the optimization result.

    Parameters
    ----------
    result: DatasetConvertible
        Result from a pyglotaran optimization as dataset, Path or Result object.
    center_λ: float | None
        Center wavelength (λ in nm)
    linlog: bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh: float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    linscale: float
        This allows the linear range (-linthresh to linthresh) to be stretched
        relative to the logarithmic range.
        Its value is the number of decades to use for each half of the linear range.
        For example, when linscale == 1.0 (the default), the space used for the
        positive and negative halves of the linear range will be equal to one
        decade in the logarithmic range. Defaults to 1.
    show_data: bool
        Whether to show the input data or residual. Defaults to False.
    main_irf_nr: int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks. Defaults to 0.
    figsize : tuple[int, int]
        Size of the figure (N, M) in inches. Defaults to (18, 16).
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    figure_only: bool
        Whether or not to only return the figure.
        This is a deprecation helper argument to transition to a consistent return value
        consisting of the :class:`Figure` and the :class:`Axes`. Defaults to True.
    nr_of_data_svd_vectors: int
        Number of data SVD vector to plot. Defaults to 4.
    nr_of_residual_svd_vectors: int
        Number of residual SVD vector to plot. Defaults to 2.
    show_data_svd_legend: bool
        Whether or not to show the data SVD legend. Defaults to True.
    show_residual_svd_legend: bool
        Whether or not to show the residual SVD legend. Defaults to True.

    Returns
    -------
    Figure|tuple[Figure, Axes]
        If ``figure_only`` is True, Figure object which contains the plots (deprecated).
        If ``figure_only`` is False, Figure object which contains the plots and the Axes.
    """
    res = load_data(result)

    # Plot dimensions
    M = 4
    N = 3
    fig, axes = plt.subplots(M, N, figsize=figsize, constrained_layout=True)

    if center_λ is None:  # center wavelength (λ in nm)
        center_λ = min(res.dims["spectral"], round(res.dims["spectral"] / 2))

    # First and second row: concentrations - SAS/EAS - DAS
    plot_concentrations(
        res,
        axes[0, 0],
        center_λ,
        linlog=linlog,
        linthresh=linthresh,
        linscale=linscale,
        main_irf_nr=main_irf_nr,
        cycler=cycler,
    )
    plot_spectra(res, axes[0:2, 1:3], cycler=cycler)
    plot_svd(
        res,
        axes[2:4, 0:3],
        linlog=linlog,
        linthresh=linthresh,
        cycler=cycler,
        nr_of_data_svd_vectors=nr_of_data_svd_vectors,
        nr_of_residual_svd_vectors=nr_of_residual_svd_vectors,
        show_data_svd_legend=show_data_svd_legend,
        show_residual_svd_legend=show_residual_svd_legend,
    )
    plot_residual(
        res, axes[1, 0], linlog=linlog, linthresh=linthresh, show_data=show_data, cycler=cycler
    )
    # plt.tight_layout(pad=3, w_pad=4.0, h_pad=4.0)
    if figure_only is True:
        warn(PyglotaranExtrasApiDeprecationWarning(FIG_ONLY_WARNING), stacklevel=2)
        return fig
    else:
        return fig, axes


def plot_simple_overview(
    result: DatasetConvertible,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    cycler: Cycler | None = PlotStyle().cycler,
    figure_only: bool = True,
) -> Figure | tuple[Figure, Axes]:
    """Plot simple overview.

    Parameters
    ----------
    result: DatasetConvertible
        Result from a pyglotaran optimization as dataset, Path or Result object.
    title: str | None
        Title of the figure. Defaults to None.
    figsize : tuple[int, int]
        Size of the figure (N, M) in inches. Defaults to (18, 16).
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    figure_only: bool
        Whether or not to only return the figure.
        This is a deprecation helper argument to transition to a consistent return value
        consisting of the :class:`Figure` and the :class:`Axes`. Defaults to True.

    Returns
    -------
    Figure|tuple[Figure, Axes]
        If ``figure_only`` is True, Figure object which contains the plots (deprecated).
        If ``figure_only`` is False, Figure object which contains the plots and the Axes.
    """
    res = load_data(result)

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    for ax in axes.flatten():
        add_cycler_if_not_none(ax, cycler)
    if title:
        fig.suptitle(title, fontsize=16)
    sas = res.species_associated_spectra
    traces = res.species_concentration
    if "spectral" in traces.coords:
        traces.sel(spectral=res.spectral.values[0], method="nearest").plot.line(
            x="time", ax=axes[0, 0]
        )
    else:
        traces.plot.line(x="time", ax=axes[0, 0])
    sas.plot.line(x="spectral", ax=axes[0, 1])
    rLSV = res.residual_left_singular_vectors
    rLSV.isel(left_singular_value_index=range(min(2, len(rLSV)))).plot.line(
        x="time", ax=axes[1, 0]
    )

    axes[1, 0].set_title("res. LSV")
    rRSV = res.residual_right_singular_vectors
    rRSV.isel(right_singular_value_index=range(min(2, len(rRSV)))).plot.line(
        x="spectral", ax=axes[1, 1]
    )

    axes[1, 1].set_title("res. RSV")
    res.data.plot(x="time", ax=axes[0, 2])
    axes[0, 2].set_title("data")
    res.residual.plot(x="time", ax=axes[1, 2])
    axes[1, 2].set_title("residual")
    if figure_only is not True:
        return fig, axes
    warn(PyglotaranExtrasApiDeprecationWarning(FIG_ONLY_WARNING), stacklevel=2)
    return fig


if __name__ == "__main__":
    import sys

    result_path = Path(sys.argv[1])
    res = load_data(result_path)
    print(res)

    fig, plt.axes = plot_overview(res, figure_only=False)
    if len(sys.argv) > 2:
        fig.savefig(sys.argv[2], bbox_inches="tight")
        print(f"Saved figure to: {sys.argv[2]}")
    else:
        plt.show(block=False)
        input("press <ENTER> to continue")
