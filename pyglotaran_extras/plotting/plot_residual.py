"""Module containing residual plot functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyglotaran_extras.plotting.plot_irf_dispersion_center import _plot_irf_dispersion_center
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import MinorSymLogLocator
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import shift_time_axis_by_irf_location

if TYPE_CHECKING:
    import xarray as xr
    from cycler import Cycler
    from matplotlib.axis import Axis


def plot_residual(
    res: xr.Dataset,
    ax: Axis,
    linlog: bool = False,
    linthresh: float = 1,
    show_data: bool | None = False,
    cycler: Cycler | None = PlotStyle().cycler,
    show_irf_dispersion_center: bool = True,
    irf_location: float | None = None,
) -> None:
    """Plot data or residual on a 2D contour plot.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    linlog : bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    show_data: bool | None
        Whether to show the input data or residual. If set to ``None`` the plot is skipped
        which improves plotting performance for big datasets. Defaults to False.
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_irf_dispersion_center: bool
        Whether to show the the IRF dispersion center as overlay on the residual/data plot.
        Defaults to True.
    irf_location:  float | None
        Location of the ``irf`` by which the time axis will get shifted. If it is None the time
        axis will not be shifted. Defaults to None.
    """
    if show_data is None:
        ax.text(
            0.5,
            0.5,
            "Skipped",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=24,
        )
        return

    add_cycler_if_not_none(ax, cycler)
    data = res.data if show_data else res.residual
    data = shift_time_axis_by_irf_location(data, irf_location)
    title = "dataset" if show_data else "residual"
    shape = np.array(data.shape)
    # Handle different dimensionality of data
    if min(shape) == 1:
        dims = data.coords.dims
        data.plot.line(x=dims[shape.argmax()], ax=ax)
    elif min(shape) < 5:
        data.plot(x="time", ax=ax)
    else:
        data.plot(x="time", ax=ax, add_colorbar=False)
    if show_irf_dispersion_center is True:
        _plot_irf_dispersion_center(
            res, ax=ax, spectral_axis="y", cycler=cycler, irf_location=irf_location
        )
        ax.set_xlabel("time")
        ax.legend()
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))
    ax.set_title(title)
