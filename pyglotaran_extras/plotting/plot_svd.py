"""Module containing SVD plotting functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.io.prepare_dataset import add_svd_to_dataset

from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import MinorSymLogLocator
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import shift_time_axis_by_irf_location

if TYPE_CHECKING:
    from typing import Sequence

    import xarray as xr
    from cycler import Cycler
    from matplotlib.axis import Axis
    from matplotlib.pyplot import Axes


def plot_svd(
    res: xr.Dataset,
    axes: Axes,
    linlog: bool = False,
    linthresh: float = 1,
    cycler: Cycler | None = PlotStyle().cycler,
    nr_of_data_svd_vectors: int = 4,
    nr_of_residual_svd_vectors: int = 2,
    show_data_svd_legend: bool = True,
    show_residual_svd_legend: bool = True,
    irf_location: float | None = None,
) -> None:
    """Plot SVD (Singular Value Decomposition) of data and residual.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    axes : Axes
        Axes to plot the SVDs on (needs to be at least 2x3).
    linlog : bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    nr_of_data_svd_vectors: int
        Number of data SVD vector to plot. Defaults to 4.
    nr_of_residual_svd_vectors: int
        Number of residual SVD vector to plot. Defaults to 2.
    show_data_svd_legend: bool
        Whether or not to show the data SVD legend. Defaults to True.
    show_residual_svd_legend: bool
        Whether or not to show the residual SVD legend. Defaults to True.
    irf_location:  float | None
        Location of the ``irf`` by which the time axis will get shifted. If it is None the time
        axis will not be shifted. Defaults to None.
    """
    if "weighted_residual" in res:
        add_svd_to_dataset(dataset=res, name="weighted_residual")
    else:
        add_svd_to_dataset(dataset=res, name="residual")
    plot_lsv_residual(
        res,
        axes[0, 0],
        linlog=linlog,
        linthresh=linthresh,
        cycler=cycler,
        indices=range(nr_of_residual_svd_vectors),
        show_legend=show_residual_svd_legend,
        irf_location=irf_location,
    )
    plot_rsv_residual(
        res,
        axes[0, 1],
        cycler=cycler,
        indices=range(nr_of_residual_svd_vectors),
        show_legend=show_residual_svd_legend,
    )
    plot_sv_residual(res, axes[0, 2], cycler=cycler)
    add_svd_to_dataset(dataset=res, name="data")
    plot_lsv_data(
        res,
        axes[1, 0],
        linlog=linlog,
        linthresh=linthresh,
        cycler=cycler,
        indices=range(nr_of_data_svd_vectors),
        show_legend=show_data_svd_legend,
        irf_location=irf_location,
    )
    plot_rsv_data(
        res,
        axes[1, 1],
        cycler=cycler,
        indices=range(nr_of_data_svd_vectors),
        show_legend=show_data_svd_legend,
    )
    plot_sv_data(res, axes[1, 2], cycler=cycler)


def plot_lsv_data(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(4),
    linlog: bool = False,
    linthresh: float = 1,
    cycler: Cycler | None = PlotStyle().cycler,
    show_legend: bool = True,
    irf_location: float | None = None,
) -> None:
    """Plot left singular vectors (time) of the data matrix.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    indices : Sequence[int]
        Indices of the singular vector to plot. Defaults to range(4).
    linlog : bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_legend: bool
        Whether or not to show the legend. Defaults to True.
    irf_location:  float | None
        Location of the ``irf`` by which the time axis will get shifted. If it is None the time
        axis will not be shifted. Defaults to None.
    """
    add_cycler_if_not_none(ax, cycler)
    dLSV = res.data_left_singular_vectors
    dLSV = shift_time_axis_by_irf_location(dLSV, irf_location)
    _plot_svd_vectors(dLSV, indices, "left_singular_value_index", ax, show_legend)
    ax.set_title("data. LSV")
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))


def plot_rsv_data(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(4),
    cycler: Cycler | None = PlotStyle().cycler,
    show_legend: bool = True,
) -> None:
    """Plot right singular vectors (spectra) of the data matrix.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    indices : Sequence[int]
        Indices of the singular vector to plot. Defaults to range(4).
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_legend: bool
        Whether or not to show the legend. Defaults to True.
    """
    add_cycler_if_not_none(ax, cycler)
    dRSV = res.data_right_singular_vectors
    _plot_svd_vectors(dRSV, indices, "right_singular_value_index", ax, show_legend)
    ax.set_title("data. RSV")


def plot_sv_data(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(10),
    cycler: Cycler | None = PlotStyle().cycler,
) -> None:
    """Plot singular values of the data matrix.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    indices : Sequence[int]
        Indices of the singular vector to plot. Defaults to range(4).
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    """
    add_cycler_if_not_none(ax, cycler)
    dSV = res.data_singular_values
    dSV.sel(singular_value_index=indices[: len(dSV.singular_value_index)]).plot.line(
        "ro-", yscale="log", ax=ax
    )
    ax.set_title("data. log(SV)")


def plot_lsv_residual(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(2),
    linlog: bool = False,
    linthresh: float = 1,
    cycler: Cycler | None = PlotStyle().cycler,
    show_legend: bool = True,
    irf_location: float | None = None,
) -> None:
    """Plot left singular vectors (time) of the residual matrix.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    indices : Sequence[int]
        Indices of the singular vector to plot. Defaults to range(4).
    linlog : bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_legend: bool
        Whether or not to show the legend. Defaults to True.
    irf_location:  float | None
        Location of the ``irf`` by which the time axis will get shifted. If it is None the time
        axis will not be shifted. Defaults to None.
    """
    add_cycler_if_not_none(ax, cycler)
    if "weighted_residual_left_singular_vectors" in res:
        rLSV = res.weighted_residual_left_singular_vectors
    else:
        rLSV = res.residual_left_singular_vectors
    rLSV = shift_time_axis_by_irf_location(rLSV, irf_location)
    _plot_svd_vectors(rLSV, indices, "left_singular_value_index", ax, show_legend)
    ax.set_title("res. LSV")
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))


def plot_rsv_residual(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(2),
    cycler: Cycler | None = PlotStyle().cycler,
    show_legend: bool = True,
) -> None:
    """Plot right singular vectors (spectra) of the residual matrix.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    indices : Sequence[int]
        Indices of the singular vector to plot. Defaults to range(4).
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_legend: bool
        Whether or not to show the legend. Defaults to True.
    """
    add_cycler_if_not_none(ax, cycler)
    if "weighted_residual_right_singular_vectors" in res:
        rRSV = res.weighted_residual_right_singular_vectors
    else:
        rRSV = res.residual_right_singular_vectors
    _plot_svd_vectors(rRSV, indices, "right_singular_value_index", ax, show_legend)
    ax.set_title("res. RSV")


def plot_sv_residual(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(10),
    cycler: Cycler | None = PlotStyle().cycler,
) -> None:
    """Plot singular values of the residual matrix.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    indices : Sequence[int]
        Indices of the singular vector to plot. Defaults to range(4).
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    """
    add_cycler_if_not_none(ax, cycler)
    if "weighted_residual_singular_values" in res:
        rSV = res.weighted_residual_singular_values
    else:
        rSV = res.residual_singular_values
    rSV.sel(singular_value_index=indices[: len(rSV.singular_value_index)]).plot.line(
        "ro-", yscale="log", ax=ax
    )
    ax.set_title("res. log(SV)")


def _plot_svd_vectors(
    vector_data: xr.DataArray,
    indices: Sequence[int],
    sv_index_dim: str,
    ax: Axis,
    show_legend: bool,
) -> None:
    """Plot SVD vectors with decreasing zorder on axis ``ax``.

    Parameters
    ----------
    vector_data: xr.DataArray
        DataArray containing the SVD vector data.
    indices: Sequence[int]
        Indices of the singular vector to plot.
    sv_index_dim: str
        Name of the singular value index dimension.
    ax: Axis
        Axis to plot on.
    show_legend: bool
        Whether or not to show the legend.

    See Also
    --------
    plot_lsv_data
    plot_rsv_data
    plot_lsv_residual
    plot_rsv_residual
    """
    max_index = len(getattr(vector_data, sv_index_dim))
    values = vector_data.isel(**{sv_index_dim: indices[:max_index]})
    x_dim = vector_data.dims[1]
    if x_dim == sv_index_dim:
        values = values.T
        x_dim = vector_data.dims[0]
    for zorder, label, value in zip(range(100)[::-1], indices[:max_index], values):
        value.plot.line(x=x_dim, ax=ax, zorder=zorder, label=label)
    if show_legend is True:
        ax.legend(title=sv_index_dim)
