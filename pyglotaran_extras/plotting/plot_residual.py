"""Module containing residual plot functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyglotaran_extras.plotting.style import PlotStyle

if TYPE_CHECKING:
    import xarray as xr
    from cycler import Cycler
    from matplotlib.axis import Axis


def plot_residual(
    res: xr.Dataset,
    ax: Axis,
    linlog: bool = False,
    linthresh: float = 1,
    show_data: bool = False,
    cycler: Cycler = PlotStyle().cycler,
) -> None:
    """Plot data ro residual and 2D contour plot.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    linlog : bool
        Weather to use 'symlog' scale or not, by default False
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero., by default 1
    show_data : bool
        Weather to show the data or the residual., by default False
    cycler : Cycler
        Plot style cycler to use., by default PlotStyle().cycler
    """
    ax.set_prop_cycle(cycler)
    data = res.data if show_data else res.residual
    title = "dataset" if show_data else "residual"
    shape = np.array(data.shape)
    dims = data.coords.dims
    if min(shape) == 1:
        # res.data.plot.line(x=dims[shape.argmax()], ax=ax)
        # res.fitted_data.plot.line(x=dims[shape.argmax()], ax=ax)
        data.plot.line(x=dims[shape.argmax()], ax=ax)
    elif min(shape) < 5:
        data.plot(x="time", ax=ax)
    else:
        data.plot(x="time", ax=ax, add_colorbar=False)
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_title(title)
