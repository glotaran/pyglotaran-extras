"""Module containing data plotting functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import matplotlib.pyplot as plt
from glotaran.io.prepare_dataset import add_svd_to_dataset
from matplotlib.axis import Axis

from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.plotting.plot_svd import plot_lsv_data
from pyglotaran_extras.plotting.plot_svd import plot_rsv_data
from pyglotaran_extras.plotting.plot_svd import plot_sv_data

__all__ = ["plot_data_overview"]

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import DatasetConvertible


def plot_data_overview(
    dataset: DatasetConvertible,
    title: str = "Data overview",
    linlog: bool = False,
    linthresh: float = 1,
    figsize: tuple[int, int] = (15, 10),
    nr_of_data_svd_vectors: int = 4,
    show_data_svd_legend: bool = True,
) -> tuple[Figure, Axes]:
    """Plot data as filled contour plot and SVD components.

    Parameters
    ----------
    dataset : DatasetConvertible
        Dataset containing data and SVD of the data.
    title : str
        Title to add to the figure. Defaults to "Data overview".
    linlog : bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh : float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    figsize : tuple[int, int]
        Size of the figure (N, M) in inches. Defaults to (15, 10).
    nr_of_data_svd_vectors: int
        Number of data SVD vector to plot. Defaults to 4.
    show_data_svd_legend: bool
        Whether or not to show the data SVD legend. Defaults to True.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes which can then be refined by the user.
    """
    dataset = load_data(dataset)

    fig = plt.figure(figsize=figsize)
    data_ax = cast(Axis, plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3, fig=fig))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    lsv_ax = cast(Axis, plt.subplot2grid((4, 3), (3, 0), fig=fig))
    sv_ax = cast(Axis, plt.subplot2grid((4, 3), (3, 1), fig=fig))
    rsv_ax = cast(Axis, plt.subplot2grid((4, 3), (3, 2), fig=fig))

    if len(dataset.data.time) > 1:
        dataset.data.plot(x="time", ax=data_ax, center=False)
    else:
        dataset.data.plot(ax=data_ax)

    add_svd_to_dataset(dataset=dataset, name="data")
    plot_lsv_data(dataset, lsv_ax, indices=range(nr_of_data_svd_vectors), show_legend=False)
    plot_sv_data(dataset, sv_ax)
    plot_rsv_data(dataset, rsv_ax, indices=range(nr_of_data_svd_vectors), show_legend=False)
    if show_data_svd_legend is True:
        rsv_ax.legend(title="singular value index", loc="lower right", bbox_to_anchor=(1.13, 1))
    fig.suptitle(title, fontsize=16)

    if linlog:
        data_ax.set_xscale("symlog", linthresh=linthresh)
    return fig, (data_ax, lsv_ax, sv_ax, rsv_ax)
