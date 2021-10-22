from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from pyglotaran_extras.io.utils import result_dataset_mapping

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

from typing import Iterable

from pyglotaran_extras.types import ResultLike


class PlotDuplicationWarning(UserWarning):
    """Warning given when there are more subplots than datapoints."""


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
        Shape of the plot grid (N, M), by default (4, 4)
    wavelength_range: tuple[float, float]
        Tuple of minimum and maximum values to calculate the the wavelengths
        used for plotting. If not provided the values will be tetermined over all datasets.
        , by default None
    equidistant_wavelengths: bool
        Whether or not wavelengths should be selected based on equidistant values
        or equidistant indices (only supported for a single dataset).
        Since in general multiple datasets will have , by default True

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
        , by default True

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
    irf_loc = extract_irf_location(res, center_λ, main_irf_nr)

    times_shifted = traces.coords["time"] - irf_loc
    return traces.assign_coords(time=times_shifted)
