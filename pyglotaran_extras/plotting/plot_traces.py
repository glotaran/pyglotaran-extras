from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from pyglotaran_extras.plotting.style import PlotStyle

if TYPE_CHECKING:
    import xarray as xr


def get_shifted_traces(
    res: xr.Dataset, center_λ: float | None = None, main_irf_nr: int = 0
) -> xr.DataArray:
    """Shift traces by the position of the main ``irf``.

    Parameters
    ----------
    res: xr.Dataset
        Result dataset from a pyglotaran optimization.
    center_λ: float|None
        Center wavelength (λ in nm), by default None
    main_irf_nr: int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks , by default 0

    Returns
    -------
    xr.DataArray
        Traces shifted by the ``irf``s location, to align the at 0.

    Raises
    ------
    ValueError
        If no known concentration was found in the result.
    """
    if "species_concentration" in res:
        traces = res.species_concentration
    elif "species_associated_concentrations" in res:
        traces = res.species_associated_concentrations
    else:
        raise ValueError(f"No concentrations in result:\n{res}")
    times = traces.coords["time"]
    if center_λ is None:  # center wavelength (λ in nm)
        center_λ = min(res.dims["spectral"], round(res.dims["spectral"] / 2))

    if "irf_center_location" in res:
        center_dispersion = res.center_dispersion
        irf_loc = center_dispersion.sel(spectral=center_λ, method="nearest").item()
    elif "center_dispersion_1" in res:
        # legacy compatibility pyglotaran<0.5.0
        center_dispersion = res.center_dispersion_1
        irf_loc = center_dispersion.sel(spectral=center_λ, method="nearest").item()
    elif "irf_center" in res:
        irf_loc = res.irf_center
    else:
        irf_loc = min(times)

    if hasattr(irf_loc, "shape") and len(irf_loc.shape) > 0:
        irf_loc = irf_loc[main_irf_nr]

    times_shifted = times - irf_loc
    return traces.assign_coords(time=times_shifted)


def calculate_x_ranges(res, linrange):
    print(f"{res=}")
    print(f"{linrange=}")
    print("Not yet implemented")
    pass


def plot_traces(
    res: xr.Dataset,
    ax: plt.Axes,
    center_λ: float | None,
    linlog: bool = False,
    linthresh: float = 1,
    linscale: float = 1,
    main_irf_nr: int = 0,
) -> None:
    """Plot traces on the given axis ``ax``

    Parameters
    ----------
    res: xr.Dataset
        Result dataset from a pyglotaran optimization.
    ax: plt.Axes
        Axes to plot the traces on
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

    See Also
    --------
    get_shifted_traces
    """
    traces = get_shifted_traces(res, center_λ, main_irf_nr)
    plot_style = PlotStyle()
    plt.rc("axes", prop_cycle=plot_style.cycler)

    if "spectral" in traces.coords:
        traces.sel(spectral=center_λ, method="nearest").plot.line(x="time", ax=ax)
    else:
        traces.plot.line(x="time", ax=ax)

    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh, linscale=linscale)
