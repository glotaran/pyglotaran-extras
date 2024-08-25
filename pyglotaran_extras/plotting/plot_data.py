"""Module containing data plotting functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import matplotlib.pyplot as plt
from glotaran.io.prepare_dataset import add_svd_to_dataset
from matplotlib.axis import Axis

from pyglotaran_extras.config.plot_config import use_plot_config
from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.plotting.plot_svd import plot_lsv_data
from pyglotaran_extras.plotting.plot_svd import plot_rsv_data
from pyglotaran_extras.plotting.plot_svd import plot_sv_data
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import MinorSymLogLocator
from pyglotaran_extras.plotting.utils import not_single_element_dims
from pyglotaran_extras.plotting.utils import shift_time_axis_by_irf_location

__all__ = ["plot_data_overview"]

if TYPE_CHECKING:
    from collections.abc import Hashable

    import xarray as xr
    from cycler import Cycler
    from glotaran.project.result import Result
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import DatasetConvertible


@use_plot_config(exclude_from_config=("svd_cycler",))
def plot_data_overview(
    dataset: DatasetConvertible | Result,
    title: str = "Data overview",
    linlog: bool = False,
    linthresh: float = 1,
    figsize: tuple[float, float] = (15, 10),
    nr_of_data_svd_vectors: int = 4,
    show_data_svd_legend: bool = True,
    irf_location: float | None = None,
    cmap: str = "PuRd",
    vmin: float | None = None,
    vmax: float | None = None,
    svd_cycler: Cycler | None = PlotStyle().cycler,
    use_svd_number: bool = False,
) -> tuple[Figure, Axes] | tuple[Figure, Axis]:
    """Plot data as filled contour plot and SVD components.

    Parameters
    ----------
    dataset : DatasetConvertible | Result
        Dataset containing data and SVD of the data.
    title : str
        Title to add to the figure. Defaults to "Data overview".
    linlog : bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    figsize : tuple[float, float]
        Size of the figure (N, M) in inches. Defaults to (15, 10).
    nr_of_data_svd_vectors : int
        Number of data SVD vector to plot. Defaults to 4.
    show_data_svd_legend : bool
        Whether or not to show the data SVD legend. Defaults to True.
    irf_location : float | None
        Location of the ``irf`` by which the time axis will get shifted. If it is None the time
        axis will not be shifted. Defaults to None.
    cmap : str
        Colormap to use for the filled contour plot. Defaults to "PuRd" which is most suitable
        for emission data (previous default was "viridis").
    vmin : float | None
        Lower value to anchor the colormap. Defaults to None meaning it inferred from the data.
    vmax : float | None
        Lower value to anchor the colormap. Defaults to None meaning it inferred from the data.
    svd_cycler : Cycler | None
        Plot style cycler to use for SVD plots. Defaults to ``PlotStyle().cycler``.
    use_svd_number : bool
        Whether to use singular value number (starts at 1) instead of singular value index
        (starts at 0) for labeling in plot. Defaults to False.

    Returns
    -------
    tuple[Figure, Axes] | tuple[Figure, Axis]
        Figure and axes which can then be refined by the user.
    """
    dataset = load_data(dataset, _stacklevel=3)
    data = shift_time_axis_by_irf_location(dataset.data, irf_location, _internal_call=True)

    if len(not_single_element_dims(data)) == 1:
        return _plot_single_trace(
            data,
            not_single_element_dims(data)[0],
            title="Single trace data",
            linlog=linlog,
            linthresh=linthresh,
            figsize=figsize,
        )

    fig = plt.figure(figsize=figsize)
    data_ax = cast(Axis, plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3, fig=fig))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    lsv_ax = cast(Axis, plt.subplot2grid((4, 3), (3, 0), fig=fig))
    sv_ax = cast(Axis, plt.subplot2grid((4, 3), (3, 1), fig=fig))
    rsv_ax = cast(Axis, plt.subplot2grid((4, 3), (3, 2), fig=fig))

    if len(data.time) > 1:
        data.plot(x="time", ax=data_ax, center=False, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        data.plot(ax=data_ax)

    add_svd_to_dataset(dataset=dataset, name="data")
    plot_lsv_data(
        dataset,
        lsv_ax,
        indices=range(nr_of_data_svd_vectors),
        show_legend=False,
        linlog=linlog,
        linthresh=linthresh,
        irf_location=irf_location,
        cycler=svd_cycler,
        use_svd_number=use_svd_number,
    )
    plot_sv_data(dataset, sv_ax, use_svd_number=use_svd_number)
    plot_rsv_data(
        dataset,
        rsv_ax,
        indices=range(nr_of_data_svd_vectors),
        show_legend=False,
        cycler=svd_cycler,
        use_svd_number=use_svd_number,
    )
    if show_data_svd_legend is True:
        rsv_ax.legend(
            title="singular value number" if use_svd_number else "singular_value_index",
            loc="lower right",
            bbox_to_anchor=(1.13, 1),
        )
    fig.suptitle(title, fontsize=16)

    if linlog:
        data_ax.set_xscale("symlog", linthresh=linthresh)
        data_ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))
    return fig, (data_ax, lsv_ax, sv_ax, rsv_ax)


def _plot_single_trace(
    data_array: xr.DataArray,
    x_dim: Hashable,
    *,
    title: str = "Single trace data",
    linlog: bool = False,
    linthresh: float = 1,
    figsize: tuple[float, float] = (15, 10),
) -> tuple[Figure, Axis]:
    """Plot single trace data in case ``plot_data_overview`` gets passed ingle trace data.

    Parameters
    ----------
    data_array : xr.DataArray
        DataArray containing only data  of a single trace.
    x_dim : Hashable
        Name of the x dimension.
    title : str
        Title to add to the figure. Defaults to "Data overview".
    linlog : bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    figsize : tuple[float, float]
        Size of the figure (N, M) in inches. Defaults to (15, 10).

    Returns
    -------
    tuple[Figure, Axis]
        Figure and axis which can then be refined by the user.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    data_array.plot.line(x=x_dim, ax=ax)
    fig.suptitle(title, fontsize=16)

    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))
    return fig, ax
