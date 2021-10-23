from __future__ import annotations

from typing import TYPE_CHECKING

from pyglotaran_extras.plotting.style import PlotStyle
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
    cycler: Cycler = PlotStyle().cycler,
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
        Whether to use 'symlog' scale or not, by default False
    linthresh: int
        A single float which defines the range (-x, x), within which the plot is linear.
        This avoids having the plot go to infinity around zero., by default 1
    linscale: int
        This allows the linear range (-linthresh to linthresh) to be stretched
        relative to the logarithmic range.
        Its value is the number of decades to use for each half of the linear range.
        For example, when linscale == 1.0 (the default), the space used for the
        positive and negative halves of the linear range will be equal to one
        decade in the logarithmic range., by default 1
    main_irf_nr: int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks , by default 0
    cycler : Cycler
        Plot style cycler to use., by default PlotStyle().data_cycler_solid

    See Also
    --------
    get_shifted_traces
    """
    ax.set_prop_cycle(cycler)
    traces = get_shifted_traces(res, center_λ, main_irf_nr)

    if "spectral" in traces.coords:
        traces.sel(spectral=center_λ, method="nearest").plot.line(x="time", ax=ax)
    else:
        traces.plot.line(x="time", ax=ax)

    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh, linscale=linscale)
