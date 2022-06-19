"""Module containing plotting utility functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import xarray as xr

from pyglotaran_extras.io.utils import result_dataset_mapping

if TYPE_CHECKING:
    from typing import Iterable

    from cycler import Cycler
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import ResultLike


class PlotDuplicationWarning(UserWarning):
    """Warning given when there are more subplots than datapoints."""


def select_irf_nr(irf_dispersion: xr.DataArray, main_irf_nr: int = 0) -> xr.DataArray | float:
    """Select a subset of the IRF dispersion data where ``irf_nr==main_irf_nr``.

    Parameters
    ----------
    irf_dispersion: xr.DataArray
        Data Variable from a result dataset which contains the IRF dispersion data.
    main_irf_nr: int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks. Defaults to 0.

    Returns
    -------
    xr.DataArray | float
        DataArray only containing the IRF dispersion data for the main IRF.

    Raises
    ------
    ValueError
        If ``irf_nr`` is not in the coordinates
    """
    if "irf_nr" in irf_dispersion.sizes:
        if main_irf_nr >= irf_dispersion.sizes["irf_nr"]:
            raise ValueError(
                f"The value {main_irf_nr=} is not a valid value for "
                f"irf_nr, needs to be smaller than {irf_dispersion.sizes['irf_nr']}."
            )
        irf_dispersion = irf_dispersion.sel(irf_nr=main_irf_nr)
    if irf_dispersion.size == 1:
        irf_dispersion = irf_dispersion.item()
    return irf_dispersion


def extract_irf(
    res: xr.Dataset, main_irf_nr: int = 0, *, as_dataarray: bool = True
) -> xr.DataArray | float:
    """Extract the IRF data from a result dataset where ``irf_nr==main_irf_nr``.

    Parameters
    ----------
    res: xr.Dataset
        Result dataset from a pyglotaran optimization.
    main_irf_nr: int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks. Defaults to 0.
    as_dataarray: bool
        Ensure that the returned data are xr.DataArray instead of a float, even if the
        dispersion is none existent or constant. Defaults to True


    Returns
    -------
    xr.DataArray | float
        IRF dispersion data as float or xr.DataArray
    """
    if "irf_center_location" in res:
        return select_irf_nr(res.irf_center_location, main_irf_nr=main_irf_nr)
    if "center_dispersion_1" in res:
        # legacy compatibility pyglotaran<0.5.0
        return select_irf_nr(res.center_dispersion_1, main_irf_nr=main_irf_nr)

    # No/constant dispersion
    if "irf_center" in res:
        irf_dispersion = select_irf_nr(res.irf_center, main_irf_nr=main_irf_nr)
    else:
        irf_dispersion = min(res.coords["time"]).item()

    if as_dataarray is True:
        spectral = res.coords["spectral"].values
        return xr.DataArray(irf_dispersion * np.ones(spectral.shape), {"spectral": spectral})
    return irf_dispersion


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
        parametrized with multiple peaks. Defaults to 0.

    Returns
    -------
    float
        Location of the ``irf``
    """
    irf_dispersion = extract_irf(res=res, main_irf_nr=main_irf_nr, as_dataarray=False)
    if isinstance(irf_dispersion, xr.DataArray):
        if center_λ is None:  # center wavelength (λ in nm)
            center_λ = min(res.dims["spectral"], round(res.dims["spectral"] / 2))
        return irf_dispersion.sel(spectral=center_λ, method="nearest").item()

    return irf_dispersion


def maximum_coordinate_range(
    result: ResultLike, coord_name: str = "spectral"
) -> tuple[float, float]:
    """Calculate the minimal and maximal values of a coordinate across datasets.

    Parameters
    ----------
    result : ResultLike
        Data structure which can be converted to a mapping.
    coord_name : str
        Name of the coordinate to calculate the maximal range for.

    Returns
    -------
    tuple[float, float]
        Minimal and maximal values across datasets

    See Also
    --------
    plot_fit_overview
    """
    result_map = result_dataset_mapping(result)
    minima = []
    maxima = []
    for dataset in result_map.values():
        coord = dataset.coords[coord_name].values
        minima.append(coord.min())
        maxima.append(coord.max())
    return min(minima), max(maxima)


def add_unique_figure_legend(fig: Figure, axes: Axes) -> None:
    """Add a legend with unique elements sorted by label to a figure.

    The handles and labels are extracted from the ``axes``

    Parameters
    ----------
    fig : Figure
        Figure to add the legend to.
    axes : Axes
        Axes plotted on the figure.

    See Also
    --------
    plot_fit_overview
    """
    handles = []
    labels = []
    for ax in axes.flatten():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles += ax_handles
        labels += ax_labels
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    unique.sort(key=lambda entry: entry[1])
    fig.legend(*zip(*unique))


def select_plot_wavelengths(
    result: ResultLike,
    axes_shape: tuple[int, int] = (4, 4),
    wavelength_range: tuple[float, float] | None = None,
    equidistant_wavelengths: bool = True,
) -> Iterable[float]:
    """Select wavelengths to be used in ``plot_fit_overview`` from a result.

    Parameters
    ----------
    result: ResultLike
        Data structure which can be converted to a mapping of datasets.
    axes_shape: tuple[int, int]
        Shape of the plot grid (N, M). Defaults to (4, 4).
    wavelength_range: tuple[float, float]
        Tuple of minimum and maximum values to calculate the the wavelengths
        used for plotting. If not provided the values will be tetermined over all datasets.
        Defaults to None.
    equidistant_wavelengths: bool
        Whether or not wavelengths should be selected based on equidistant values
        or equidistant indices (only supported for a single dataset).
        Since in general multiple datasets will have. Defaults to True.

    Returns
    -------
    Iterable[float]
        Wavelength which should be used for each subplot by ``plot_fit_overview``.

    See Also
    --------
    maximum_coordinate_range
    """
    result_map = result_dataset_mapping(result)
    nr_of_plots = np.prod(axes_shape)

    if wavelength_range is None:
        wavelength_range = maximum_coordinate_range(result_map)

    if equidistant_wavelengths:
        return np.linspace(*wavelength_range, num=nr_of_plots)

    first_dataset = next(iter(result_map.keys()))
    if len(result_map) > 1:
        warn(
            UserWarning(
                "Calculating plot wavelengths is only supported, for a single dataset."
                f"The dataset {first_dataset!r}, will be used to calculate the selected "
                "wavelengths.To mute this warning call "
                f"'{select_plot_wavelengths.__name__}' with only one dataset."
            ),
            stacklevel=2,
        )
    spectral_coords = result_map[first_dataset].coords["spectral"]
    spectral_coords = spectral_coords[
        (wavelength_range[0] <= spectral_coords) & (spectral_coords <= wavelength_range[1])
    ]
    spectral_indices = np.linspace(0, len(spectral_coords) - 1, num=nr_of_plots, dtype=np.int64)
    return spectral_coords[spectral_indices].values


def extract_dataset_scale(res: xr.Dataset, divide_by_scale: bool = True) -> float:
    """Extract 'dataset_scale' attribute from optimization result dataset.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset from a pyglotaran optimization.
    divide_by_scale : bool
        Whether or not to divide the data by the dataset scale used for optimization.
        Defaults to True.

    Returns
    -------
    float
        Dataset scale extracted from ``res`` falls back to 1 if not present.
    """
    scale = 1
    if divide_by_scale is True:
        if "dataset_scale" in res.attrs:
            scale = res.attrs["dataset_scale"]
        else:
            warn(
                UserWarning(
                    "Diving data by dataset scales is only supported by results from "
                    "'pyglotaran>=0.5.0'. Please upgrade pyglotaran and recreate the "
                    "result to plot."
                )
            )
    return scale


def get_shifted_traces(
    res: xr.Dataset, center_λ: float | None = None, main_irf_nr: int = 0
) -> xr.DataArray:
    """Shift traces by the position of the main ``irf``.

    Parameters
    ----------
    res: xr.Dataset
        Result dataset from a pyglotaran optimization.
    center_λ: float|None
        Center wavelength (λ in nm). Defaults to None.
    main_irf_nr: int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks. Defaults to 0.

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
    irf_loc = extract_irf_location(res, center_λ, main_irf_nr)

    times_shifted = traces.coords["time"] - irf_loc
    return traces.assign_coords(time=times_shifted)


def add_cycler_if_not_none(axis: Axis, cycler: Cycler | None) -> None:
    """Add cycler to and axis if it is not None.

    This is a convenience function that allow to opt out of using
    a cycler, which is needed to run a plotting function in a loop
    where the cycler is controlled from the outside.


    Parameters
    ----------
    axis: Axis
        Axis to plot the data and fits on.
    cycler: Cycler | None
        Plot style cycler to use.
    """
    if cycler is not None:
        axis.set_prop_cycle(cycler)
