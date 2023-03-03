"""Module containing IRF dispersion plotting functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Tuple
from typing import cast

import matplotlib.pyplot as plt
import xarray as xr

from pyglotaran_extras.io.utils import result_dataset_mapping
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import extract_irf_dispersion_center

if TYPE_CHECKING:
    from typing import Literal

    from cycler import Cycler
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure

    from pyglotaran_extras.types import ResultLike


def plot_irf_dispersion_center(
    result: ResultLike,
    ax: Axis | None = None,
    figsize: tuple[float, float] = (12, 8),
    cycler: Cycler | None = PlotStyle().cycler,
    irf_location: float | None = None,
) -> tuple[Figure, Axis] | None:
    """Plot the IRF dispersion center over the spectral dimension for one or multiple datasets.

    Parameters
    ----------
    result: ResultLike
        Data structure which can be converted to a mapping.
    ax : Axis | None
        Axis to plot on. Defaults to None which means that a new figure and axis will be created.
    figsize: tuple[float, float]
        Size of the figure (N, M) in inches. Defaults to (12, 8).
    cycler: Cycler
        Plot style cycler to use. Defaults to PlotStyle().cycler
    irf_location:  float | None
        Location of the ``irf`` by which the time axis will get shifted. If it is None the time
        axis will not be shifted. Defaults to None.

    Returns
    -------
    tuple[Figure, Axis] | None
        Figure object which contains the plots and the Axis,
        if ``ax`` is not None nothing will be returned.
    """
    result_map = result_dataset_mapping(result)
    if ax is None:
        fig, axis = cast(Tuple[Figure, Axis], plt.subplots(1, figsize=figsize))
    else:
        axis = ax
    for dataset_name, dataset in result_map.items():
        _plot_irf_dispersion_center(
            dataset,
            axis,
            spectral_axis="x",
            cycler=cycler,
            label=dataset_name,
            irf_location=irf_location,
        )
    axis.legend()

    if ax is None:
        fig.suptitle("Instrument Response Functions", fontsize=16)
        return fig, axis
    return None


def _plot_irf_dispersion_center(
    res: xr.Dataset,
    ax: Axis,
    *,
    spectral_axis: Literal["x", "y"] = "x",
    cycler: Cycler | None = PlotStyle().cycler,
    label: str = "IRF",
    irf_location: float | None = None,
) -> None:
    """Plot the IRF dispersion center on an Axis ``ax``.

    This is an internal function to be used by higher level functions.

    Parameters
    ----------
    res: xr.Dataset
        Dataset containing the IRF data.
    ax: Axis
        Axis to plot on.
    spectral_axis: Literal["x", "y"]
        Direct of the spectral axis in the plot. Defaults to "x"
    cycler: Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    label: str
        Plot label for the IRF shown in the legend. Defaults to "IRF"
    irf_location:  float | None
        Location of the ``irf`` by which the time axis (here values) will get shifted.
        If it is None the time axis will not be shifted. Defaults to None.
    """
    add_cycler_if_not_none(ax, cycler)
    irf = cast(xr.DataArray, extract_irf_dispersion_center(res, as_dataarray=True))
    if irf_location is not None:
        (irf - irf_location).plot(ax=ax, label=label, **{spectral_axis: "spectral"})
    else:
        irf.plot(ax=ax, label=label, **{spectral_axis: "spectral"})
