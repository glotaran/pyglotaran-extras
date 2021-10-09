from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr


def extract_irf_location(
    res: xr.Dataset, center_λ: float | None = None, main_irf_nr: int = 0
) -> float:
    """Determine location of the ``irf``, which can be used to shift plots.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset from a pyglotaran optimization.
    center_λ: float | None
        Center wavelength (λ in nm)
    main_irf_nr : int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks , by default 0

    Returns
    -------
    float
        Location if the ``irf``
    """
    times = res.coords["time"]
    if center_λ is None:  # center wavelength (λ in nm)
        center_λ = min(res.dims["spectral"], round(res.dims["spectral"] / 2))

    if "irf_center_location" in res:
        irf_center_location = res.irf_center_location
        irf_loc = irf_center_location.sel(spectral=center_λ, method="nearest")
    elif "center_dispersion_1" in res:
        # legacy compatibility pyglotaran<0.5.0
        center_dispersion = res.center_dispersion_1
        irf_loc = center_dispersion.sel(spectral=center_λ, method="nearest").item()
    elif "irf_center" in res:
        irf_loc = res.irf_center
    else:
        irf_loc = min(times)

    if hasattr(irf_loc, "shape") and len(irf_loc.shape) > 0:
        irf_loc = irf_loc[main_irf_nr].item()

    return irf_loc
