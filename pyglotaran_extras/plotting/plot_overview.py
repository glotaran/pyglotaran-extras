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
from pyglotaran_extras.plotting.plot_guidance import plot_guidance
from pyglotaran_extras.plotting.plot_residual import plot_residual
from pyglotaran_extras.plotting.plot_spectra import plot_sas
from pyglotaran_extras.plotting.plot_spectra import plot_spectra
from pyglotaran_extras.plotting.plot_svd import plot_lsv_residual
from pyglotaran_extras.plotting.plot_svd import plot_rsv_residual
from pyglotaran_extras.plotting.plot_svd import plot_svd
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import extract_irf_location

if TYPE_CHECKING:
    from cycler import Cycler
    from glotaran.project.result import Result
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import DatasetConvertible


def plot_overview(
    result: DatasetConvertible | Result,
    center_λ: float | None = None,
    linlog: bool = True,
    linthresh: float = 1,
    linscale: float = 1,
    show_data: bool | None = False,
    main_irf_nr: int = 0,
    figsize: tuple[float, float] = (18, 16),
    cycler: Cycler | None = PlotStyle().cycler,
    figure_only: bool | None = None,
    nr_of_data_svd_vectors: int = 4,
    nr_of_residual_svd_vectors: int = 2,
    show_data_svd_legend: bool = True,
    show_residual_svd_legend: bool = True,
    show_irf_dispersion_center: bool = True,
    show_zero_line: bool = True,
) -> tuple[Figure, Axes]:
    """Plot overview of the optimization result.

    Parameters
    ----------
    result: DatasetConvertible | Result
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
    show_data: bool | None
        Whether to show the input data or residual. If set to ``None`` the plot is skipped
        which improves plotting performance for big datasets. Defaults to False.
    main_irf_nr: int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks. Defaults to 0.
    figsize : tuple[int, int]
        Size of the figure (N, M) in inches. Defaults to (18, 16).
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    figure_only: bool | None
        Deprecated please remove this argument for you function calls. Defaults to None.
    nr_of_data_svd_vectors: int
        Number of data SVD vector to plot. Defaults to 4.
    nr_of_residual_svd_vectors: int
        Number of residual SVD vector to plot. Defaults to 2.
    show_data_svd_legend: bool
        Whether or not to show the data SVD legend. Defaults to True.
    show_residual_svd_legend: bool
        Whether or not to show the residual SVD legend. Defaults to True.
    show_irf_dispersion_center: bool
        Whether to show the the IRF dispersion center as overlay on the residual/data plot.
        Defaults to True.
    show_zero_line: bool
        Whether or not to add a horizontal line at zero to the plots of the spectra.
        Defaults to True.

    Returns
    -------
    tuple[Figure, Axes]
    """
    res = load_data(result, _stacklevel=3)

    if res.coords["time"].values.size == 1:
        fig, axes = plot_guidance(res)
        if figure_only is not None:
            warn(PyglotaranExtrasApiDeprecationWarning(FIG_ONLY_WARNING), stacklevel=2)
        return fig, axes
    # Plot dimensions
    M = 4
    N = 3
    fig, axes = plt.subplots(M, N, figsize=figsize, constrained_layout=True)

    irf_location = extract_irf_location(res, center_λ, main_irf_nr)

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
    plot_spectra(res, axes[0:2, 1:3], cycler=cycler, show_zero_line=show_zero_line)
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
        irf_location=irf_location,
    )
    plot_residual(
        res,
        axes[1, 0],
        linlog=linlog,
        linthresh=linthresh,
        show_data=show_data,
        cycler=cycler,
        show_irf_dispersion_center=show_irf_dispersion_center,
        irf_location=irf_location,
    )
    if figure_only is not None:
        warn(PyglotaranExtrasApiDeprecationWarning(FIG_ONLY_WARNING), stacklevel=2)
    return fig, axes


def plot_simple_overview(
    result: DatasetConvertible | Result,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    cycler: Cycler | None = PlotStyle().cycler,
    figure_only: bool | None = None,
    show_irf_dispersion_center: bool = True,
    show_data: bool | None = False,
) -> Figure | tuple[Figure, Axes]:
    """Plot simple overview.

    Parameters
    ----------
    result: DatasetConvertible | Result
        Result from a pyglotaran optimization as dataset, Path or Result object.
    title: str | None
        Title of the figure. Defaults to None.
    figsize : tuple[int, int]
        Size of the figure (N, M) in inches. Defaults to (18, 16).
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    figure_only: bool | None
        Deprecated please remove this argument for you function calls. Defaults to None.
    show_irf_dispersion_center: bool
        Whether to show the the IRF dispersion center as overlay on the residual/data plot.
        Defaults to True.
    show_data: bool | None
        Whether to show the input data or residual. If set to ``None`` the plot is skipped
        which improves plotting performance for big datasets. Defaults to False.

    Returns
    -------
    tuple[Figure, Axes]
    """
    res = load_data(result, _stacklevel=3)

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    for ax in axes.flatten():
        add_cycler_if_not_none(ax, cycler)
    if title:
        fig.suptitle(title, fontsize=16)

    plot_concentrations(res, ax=axes[0, 0], center_λ=res.coords["spectral"].values[0])
    plot_sas(res, ax=axes[0, 1])

    irf_location = extract_irf_location(res, center_λ=res.coords["spectral"].values[0])

    plot_lsv_residual(res, ax=axes[1, 0], irf_location=irf_location)
    plot_rsv_residual(res, ax=axes[1, 1])

    plot_residual(
        res,
        axes[0, 2],
        show_data=show_data,
        show_irf_dispersion_center=show_irf_dispersion_center,
        irf_location=irf_location,
    )
    plot_residual(
        res,
        axes[1, 2],
        show_data=False,
        show_irf_dispersion_center=show_irf_dispersion_center,
        irf_location=irf_location,
    )

    if figure_only is not None:
        warn(PyglotaranExtrasApiDeprecationWarning(FIG_ONLY_WARNING), stacklevel=2)
    return fig, axes


if __name__ == "__main__":
    import sys

    result_path = Path(sys.argv[1])
    res = load_data(result_path)
    print(res)

    fig, axes = plot_overview(res)
    if len(sys.argv) > 2:
        fig.savefig(sys.argv[2], bbox_inches="tight")
        print(f"Saved figure to: {sys.argv[2]}")
    else:
        plt.show(block=False)
        input("press <ENTER> to continue")
