"""Module containing SVD plotting functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.io.prepare_dataset import add_svd_to_dataset

from pyglotaran_extras.plotting.style import PlotStyle

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
    cycler: Cycler = PlotStyle().cycler,
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
    cycler : Cycler
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    """
    if "weighted_residual" in res:
        add_svd_to_dataset(dataset=res, name="weighted_residual")
    else:
        add_svd_to_dataset(dataset=res, name="residual")
    plot_lsv_residual(res, axes[0, 0], linlog=linlog, linthresh=linthresh, cycler=cycler)
    plot_rsv_residual(res, axes[0, 1], cycler=cycler)
    plot_sv_residual(res, axes[0, 2], cycler=cycler)
    add_svd_to_dataset(dataset=res, name="data")
    plot_lsv_data(res, axes[1, 0], linlog=linlog, linthresh=linthresh, cycler=cycler)
    plot_rsv_data(res, axes[1, 1], cycler=cycler)
    plot_sv_data(res, axes[1, 2], cycler=cycler)


def plot_lsv_data(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(4),
    linlog: bool = False,
    linthresh: float = 1,
    cycler: Cycler = PlotStyle().cycler,
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
    cycler : Cycler
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    """
    ax.set_prop_cycle(cycler)
    dLSV = res.data_left_singular_vectors
    dLSV.isel(left_singular_value_index=indices[: len(dLSV.left_singular_value_index)]).plot.line(
        x="time", ax=ax
    )
    ax.set_title("data. LSV")
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)


def plot_rsv_data(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(4),
    cycler: Cycler = PlotStyle().cycler,
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
    cycler : Cycler
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    """
    ax.set_prop_cycle(cycler)
    dRSV = res.data_right_singular_vectors
    dRSV.isel(
        right_singular_value_index=indices[: len(dRSV.right_singular_value_index)]
    ).plot.line(x="spectral", ax=ax)
    ax.set_title("data. RSV")


def plot_sv_data(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(10),
    cycler: Cycler = PlotStyle().cycler,
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
    cycler : Cycler
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    """
    ax.set_prop_cycle(cycler)
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
    cycler: Cycler = PlotStyle().cycler,
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
    cycler : Cycler
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    """
    ax.set_prop_cycle(cycler)
    if "weighted_residual_left_singular_vectors" in res:
        rLSV = res.weighted_residual_left_singular_vectors
    else:
        rLSV = res.residual_left_singular_vectors
    rLSV.isel(left_singular_value_index=indices[: len(rLSV.left_singular_value_index)]).plot.line(
        x=rLSV.dims[0], ax=ax
    )
    ax.set_title("res. LSV")
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)


def plot_rsv_residual(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(2),
    cycler: Cycler = PlotStyle().cycler,
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
    cycler : Cycler
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    """
    ax.set_prop_cycle(cycler)
    if "weighted_residual_right_singular_vectors" in res:
        rRSV = res.weighted_residual_right_singular_vectors
    else:
        rRSV = res.residual_right_singular_vectors
    rRSV.isel(
        right_singular_value_index=indices[: len(rRSV.right_singular_value_index)]
    ).plot.line(x=rRSV.dims[1], ax=ax)
    ax.set_title("res. RSV")


def plot_sv_residual(
    res: xr.Dataset,
    ax: Axis,
    indices: Sequence[int] = range(10),
    cycler: Cycler = PlotStyle().cycler,
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
    cycler : Cycler
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    """
    ax.set_prop_cycle(cycler)
    if "weighted_residual_singular_values" in res:
        rSV = res.weighted_residual_singular_values
    else:
        rSV = res.residual_singular_values
    rSV.sel(singular_value_index=indices[: len(rSV.singular_value_index)]).plot.line(
        "ro-", yscale="log", ax=ax
    )
    ax.set_title("res. log(SV)")
