from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from pyglotaran_extras.plotting.plot_svd import plot_lsv_data
from pyglotaran_extras.plotting.plot_svd import plot_rsv_data
from pyglotaran_extras.plotting.plot_svd import plot_sv_data

__all__ = ["plot_data_overview"]

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes
    from xarray import Dataset


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
