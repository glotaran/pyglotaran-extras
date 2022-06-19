"""Module containing if dispersion plotting functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import matplotlib.pyplot as plt
from xarray import DataArray

from pyglotaran_extras.io.utils import result_dataset_mapping
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import extract_irf_dispersion

if TYPE_CHECKING:

    from cycler import Cycler
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure

    from pyglotaran_extras.types import ResultLike


def plot_irf_dispersions(
    result: ResultLike,
    figsize: tuple[int, int] = (12, 8),
    cycler: Cycler = PlotStyle().cycler,
) -> tuple[Figure, Axis]:
    """Plot the dispersion of the IRF for one or multiple datasets.

    Parameters
    ----------
    result: ResultLike
        Data structure which can be converted to a mapping.
    figsize: tuple[int, int]
        Size of the figure (N, M) in inches. Defaults to (12, 8).
    cycler: Cycler
        Plot style cycler to use. Defaults to PlotStyle().cycler

    Returns
    -------
    tuple[Figure, Axis]
        Figure object which contains the plots and the Axis.
    """
    result_map = result_dataset_mapping(result)
    fig, axis = plt.subplots(1, figsize=figsize)
    add_cycler_if_not_none(axis, cycler)
    for dataset_name, dataset in result_map.items():
        irf_dispersion = cast(DataArray, extract_irf_dispersion(dataset, as_dataarray=True))
        irf_dispersion.plot(x="spectral", ax=axis, label=dataset_name)
    axis.legend()

    return fig, axis
