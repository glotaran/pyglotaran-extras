"""Module containing guidance spectra plotting functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none

if TYPE_CHECKING:
    from cycler import Cycler
    from glotaran.project.result import Result
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import DatasetConvertible


def plot_guidance(
    result: DatasetConvertible | Result,
    figsize: tuple[float, float] = (15, 5),
    title: str = "Guidance Overview",
    y_label: str = "a.u.",
    cycler: Cycler | None = PlotStyle().cycler,
) -> tuple[Figure, Axes]:
    """Plot overview for a guidance spectrum.

    Parameters
    ----------
    result: DatasetConvertible | Result
        Result from a pyglotaran optimization as dataset, Path or Result object.
    figsize: tuple[float, float]
        Size of the figure (N, M) in inches. Defaults to (15, 5)
    title: str
        Title to add to the figure. Defaults to "Guidance Overview"
    y_label: str
        Label used for the y-axis of each subplot. Defaults to "a.u."
    cycler: Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes which can then be refined by the user.
    """
    res = load_data(result, _stacklevel=3)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for axis in axes:
        add_cycler_if_not_none(axis, cycler)

    res.data.plot(x="spectral", ax=axes[0], label="data")
    res.fitted_data.plot(x="spectral", ax=axes[0], label="fit")
    res.residual.plot(x="spectral", ax=axes[1], label="residual")

    for axis in axes:
        axis.set_ylabel(y_label)
    axes[0].legend()
    axes[0].set_title("Fit quality")
    axes[1].set_title("Residual")
    fig.suptitle(title, fontsize=28)
    plt.tight_layout()

    return fig, axes
