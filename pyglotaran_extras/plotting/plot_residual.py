from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axis import Axis


def plot_residual(
    res: xr.Dataset, ax: Axis, linlog: bool = False, linthresh: float = 1, show_data: bool = False
) -> None:
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
