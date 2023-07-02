"""Module containing concentration plot functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import MinorSymLogLocator
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import get_shifted_traces

if TYPE_CHECKING:
    import xarray as xr
    from cycler import Cycler
    from matplotlib.axis import Axis


def plot_concentrations(
    res: xr.Dataset,
    ax: Axis,
    center_λ: float | None,
    linlog: bool = False,
    linthresh: float = 1,
    linscale: float = 1,
    main_irf_nr: int = 0,
    cycler: Cycler | None = PlotStyle().cycler,
    title: str = "Concentrations",
) -> None:
    """Plot traces on the given axis ``ax``.

    Parameters
    ----------
    res: xr.Dataset
        Result dataset from a pyglotaran optimization.
    ax: Axis
        Axis to plot the traces on
    center_λ: float | None
        Center wavelength (λ in nm)
    linlog: bool
        Whether to use 'symlog' scale or not. Defaults to False.
    linthresh: float
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 1.
    linscale: float
        This allows the linear range (-linthresh to linthresh) to be stretched
        relative to the logarithmic range.
        Its value is the number of decades to use for each half of the linear range.
        For example, when linscale == 1.0 (the default), the space used for the
        positive and negative halves of the linear range will be equal to one
        decade in the logarithmic range. Defaults to 1.
    main_irf_nr: int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks. Defaults to 0.
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().data_cycler_solid.
    title: str
        Title used for the plot axis. Defaults to "Concentrations".

    See Also
    --------
    get_shifted_traces
    """
    add_cycler_if_not_none(ax, cycler)
    traces = get_shifted_traces(res, center_λ, main_irf_nr)

    if "spectral" in traces.coords:
        traces.sel(spectral=center_λ, method="nearest").plot.line(x="time", ax=ax)
    else:
        traces.plot.line(x="time", ax=ax)
    ax.set_title(title)

    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh, linscale=linscale)
        ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))
