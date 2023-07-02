"""Module containing plotting utility functionality."""
from __future__ import annotations

from math import ceil
from math import log
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Iterable
from warnings import warn

import numpy as np
import xarray as xr
from matplotlib.ticker import Locator

from pyglotaran_extras.inspect.utils import pretty_format_numerical_iterable
from pyglotaran_extras.io.utils import result_dataset_mapping

if TYPE_CHECKING:
    from typing import Callable
    from typing import Hashable
    from typing import Literal
    from typing import Mapping

    from cycler import Cycler
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import BuiltinSubPlotLabelFormatFunctionKey
    from pyglotaran_extras.types import ResultLike
    from pyglotaran_extras.types import SubPlotLabelCoord


class PlotDuplicationWarning(UserWarning):
    """Warning given when there are more subplots than datapoints."""


def select_irf_dispersion_center_by_index(
    irf_dispersion: xr.DataArray, main_irf_nr: int = 0
) -> xr.DataArray | float:
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


def extract_irf_dispersion_center(
    res: xr.Dataset, main_irf_nr: int = 0, *, as_dataarray: bool = True
) -> xr.DataArray | float:
    """Extract the IRF dispersion center data from a result dataset where ``irf_nr==irf_nr_index``.

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
        return select_irf_dispersion_center_by_index(
            res.irf_center_location, main_irf_nr=main_irf_nr
        )
    if "center_dispersion_1" in res:
        # legacy compatibility pyglotaran<0.5.0
        return select_irf_dispersion_center_by_index(
            res.center_dispersion_1, main_irf_nr=main_irf_nr
        )

    # No/constant dispersion
    if "irf_center" in res:
        irf_dispersion_center = select_irf_dispersion_center_by_index(
            res.irf_center, main_irf_nr=main_irf_nr
        )
    else:
        irf_dispersion_center = min(res.coords["time"]).item()

    if as_dataarray is True:
        spectral = res.coords["spectral"].values
        return xr.DataArray(
            irf_dispersion_center * np.ones(spectral.shape), {"spectral": spectral}
        )
    return irf_dispersion_center


def extract_irf_location(
    res: xr.Dataset, center_λ: float | None = None, main_irf_nr: int | None = 0
) -> float:
    """Determine location of the ``irf``, which can be used to shift plots.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset from a pyglotaran optimization.
    center_λ: float | None
        Center wavelength (λ in nm)
    main_irf_nr : int
        Index of the main ``irf`` component when using an ``irf`` parametrized with multiple peaks.
        If it is none ``None`` the location will be 0. Defaults to 0.

    Returns
    -------
    float
        Location of the ``irf``
    """
    if main_irf_nr is None:
        return 0
    irf_dispersion_center = extract_irf_dispersion_center(
        res=res, main_irf_nr=main_irf_nr, as_dataarray=False
    )
    if isinstance(irf_dispersion_center, xr.DataArray):
        if center_λ is None:  # center wavelength (λ in nm)
            center_λ = min(res.dims["spectral"], round(res.dims["spectral"] / 2))
        return irf_dispersion_center.sel(spectral=center_λ, method="nearest").item()

    return irf_dispersion_center


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


def shift_time_axis_by_irf_location(
    plot_data: xr.DataArray,
    irf_location: float | None,
) -> xr.DataArray:
    """Shift ``plot_data`` 'time' axis  by the position of the main ``irf``.

    Parameters
    ----------
    plot_data: xr.DataArray
        Data to plot.
    irf_location:  float | None
        Location of the ``irf``, if the value is None the original ``plot_data`` will be returned.

    Returns
    -------
    xr.DataArray
        ``plot_data`` with the time axis shifted by the position of the main ``irf``.

    Raises
    ------
    ValueError
        If ``plot_data`` does not have a time axis.

    See Also
    --------
    extract_irf_location
    """
    if irf_location is None:
        return plot_data

    if "time" not in plot_data.coords:
        raise ValueError("plot_data need to have a 'time' axis.")

    times_shifted = plot_data.coords["time"] - irf_location
    return plot_data.assign_coords(time=times_shifted)


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

    irf_location = extract_irf_location(res, center_λ, main_irf_nr)
    return shift_time_axis_by_irf_location(traces, irf_location)


def ensure_axes_array(axes: Axis | Axes) -> Axes:
    """Ensure that axes have flatten method even if it is a single axis.

    Parameters
    ----------
    axes: Axis | Axes
        Axis or Axes to convert for API consistency.

    Returns
    -------
    Axes
        Numpy ndarray of axes.
    """
    # We can't use `Axis` in isinstance so we check for the np.ndarray attribute of `Axes`
    if hasattr(axes, "flatten") is False:
        axes = np.array([axes])
    return axes


def add_cycler_if_not_none(axis: Axis | Axes, cycler: Cycler | None) -> None:
    """Add cycler to and axis if it is not None.

    This is a convenience function that allow to opt out of using
    a cycler, which is needed to run a plotting function in a loop
    where the cycler is controlled from the outside.


    Parameters
    ----------
    axis: Axis | Axes
        Axis to plot on.
    cycler: Cycler | None
        Plot style cycler to use.
    """
    if cycler is not None:
        axis = ensure_axes_array(axis)
        for ax in axis.flatten():
            ax.set_prop_cycle(cycler)


def abs_max(
    data: xr.DataArray, *, result_dims: Hashable | Iterable[Hashable] = ()
) -> xr.DataArray:
    """Calculate the absolute maximum values of ``data`` along all dims except ``result_dims``.

    Parameters
    ----------
    data: xr.DataArray
        Data for which the absolute maximum should be calculated.
    result_dims: Hashable | Iterable[Hashable]
        Dimensions of ``data`` which should be preserved and part of the resulting DataArray.
        Defaults to () which results in using the absolute maximum of all values.

    Returns
    -------
    xr.DataArray
        Absolute maximum values of ``data`` with dimensions ``result_dims``.
    """
    if not isinstance(result_dims, Iterable):
        result_dims = (result_dims,)
    reduce_dims = (dim for dim in data.dims if dim not in result_dims)
    return np.abs(data).max(dim=reduce_dims)


def calculate_ticks_in_units_of_pi(
    values: np.ndarray | xr.DataArray, *, step_size: float = 0.5
) -> tuple[Iterable[float], Iterable[str]]:
    """Calculate tick values and labels in units of Pi.

    Parameters
    ----------
    values: np.ndarray
        Values which the ticks should be calculated for.
    step_size: float
        Step size of the ticks in units of pi. Defaults to 0.5

    Returns
    -------
    tuple[Iterable[float], Iterable[str]]
        Tick values and tick labels

    See Also
    --------
    pyglotaran_extras.plotting.plot_doas.plot_doas

    Examples
    --------
    If you have a case study that uses a ``damped-oscillation`` megacomplex you can plot the
    ``damped_oscillation_phase`` with y-tick in units of Pi by the following code given that the
    dataset is saved under ``dataset.nc``.

    .. code-block:: python
        import matplotlib.pyplot as plt

        from glotaran.io import load_dataset
        from pyglotaran_extras.plotting.utils import calculate_ticks_in_units_of_pi

        dataset = load_dataset("dataset.nc")

        fig, ax = plt.subplots(1, 1)

        damped_oscillation_phase = dataset["damped_oscillation_phase"].sel(
            damped_oscillation=["osc1"]
        )
        damped_oscillation_phase.plot.line(x="spectral", ax=ax)

        ax.set_yticks(
            *calculate_ticks_in_units_of_pi(damped_oscillation_phase), rotation="horizontal"
        )
    """
    values = np.array(values)
    int_values_over_pi = np.round(values / np.pi / step_size)
    tick_labels = np.arange(int_values_over_pi.min(), int_values_over_pi.max() + 1) * step_size
    return tick_labels * np.pi, (
        str(val) for val in pretty_format_numerical_iterable(tick_labels, decimal_places=1)
    )


def not_single_element_dims(data_array: xr.DataArray) -> list[Hashable]:
    """Names of dimensions in ``data`` which don't have a size equal to one.

    This helper function is for example used to determine if a data only have a single trace,
    since this requires different plotting code (e.g. ``data_array.plot.line(x="time")``).

    Parameters
    ----------
    data_array: xr.DataArray
        DataArray to check if it has only a single dimension.

    Returns
    -------
    list[Hashable]
        Names of dimensions in ``data`` which don't have a size equal to one.
    """
    return [dim for dim, values in data_array.coords.items() if values.size != 1]


class MinorSymLogLocator(Locator):
    """Dynamically find minor tick positions based on major ticks for a symlog scaling.

    Ref.: https://stackoverflow.com/a/45696768
    """

    def __init__(self, linthresh: float, nints: int = 10) -> None:
        """Ticks will be placed between the major ticks.

        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.

        Parameters
        ----------
        linthresh : float
            A single float which defines the range (-x, x), within which the plot is linear.
        nints : int
            Number of minor tick between major ticks. Defaults to 10
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self) -> list[float]:
        """Return the locations of the ticks.

        Returns
        -------
        list[float]
            Minor ticks position.
        """
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]  # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0.0 and (
            (majorlocs[0] != self.linthresh and dmlower > self.linthresh)
            or (dmlower == self.linthresh and majorlocs[0] < 0)
        ):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] * 10.0)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] - self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0.0 and (
            (np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh)
            or (dmupper == self.linthresh and majorlocs[-1] > 0)
        ):
            majorlocs = np.append(majorlocs, majorlocs[-1] * 10.0)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1] + self.linthresh)

        # iterate through minor locs
        minorlocs: list[float] = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin: float, vmax: float) -> None:
        """Return the values of the located ticks given **vmin** and **vmax** (not implemented).

        Parameters
        ----------
        vmin : float
            Minimum value.
        vmax : float
            Maximum value.

        Raises
        ------
        NotImplementedError
            Not used
        """
        raise NotImplementedError(f"Cannot get tick locations for a {type(self)} type.")


def format_sub_plot_number_upper_case_letter(sub_plot_number: int, size: None | int = None) -> str:
    """Format ``sub_plot_number`` into an upper case letter, that can be used as label.

    Parameters
    ----------
    sub_plot_number : int
        Number of the subplot starting at One.
    size : None | int
        Size of the axes array (number of plots). Defaults to None

    Returns
    -------
    str
        Upper case label for a sub plot.

    Examples
    --------
    >>> print(format_sub_plot_number_upper_case_letter(1))
    A

    >>> print(format_sub_plot_number_upper_case_letter(26))
    Z

    >>> print(format_sub_plot_number_upper_case_letter(27))
    AA

    >>> print(format_sub_plot_number_upper_case_letter(1, 26))
    AA

    >>> print(format_sub_plot_number_upper_case_letter(2, 26))
    AB

    >>> print(format_sub_plot_number_upper_case_letter(26, 26))
    AZ

    >>> print(format_sub_plot_number_upper_case_letter(27, 50))
    BA

    See Also
    --------
    BuiltinLabelFormatFunctions
    add_subplot_labels
    """
    sub_plot_number -= 1
    if size is not None and size > 26:
        return "".join(
            format_sub_plot_number_upper_case_letter(((sub_plot_number // (26**i)) % 26) + 1)
            for i in reversed(range(1, ceil(log(size, 26))))
        ) + format_sub_plot_number_upper_case_letter((sub_plot_number % 26) + 1)
    if sub_plot_number < 26:
        return chr(ord("A") + sub_plot_number)
    return format_sub_plot_number_upper_case_letter(
        sub_plot_number // 26
    ) + format_sub_plot_number_upper_case_letter((sub_plot_number % 26) + 1)


BuiltinSubPlotLabelFormatFunctions: Mapping[
    str, Callable[[int, int | None], str]
] = MappingProxyType(
    {
        "number": lambda x, y: f"{x}",
        "upper_case_letter": format_sub_plot_number_upper_case_letter,
        "lower_case_letter": lambda x, y: format_sub_plot_number_upper_case_letter(x, y).lower(),
    }
)


def get_subplot_label_format_function(
    format_function: BuiltinSubPlotLabelFormatFunctionKey | Callable[[int, int | None], str]
) -> Callable[[int, int | None], str]:
    """Get subplot label function from ``BuiltinSubPlotLabelFormatFunctions`` if it is a key.

    This function is mainly needed for typing reasons.

    Parameters
    ----------
    format_function : BuiltinSubPlotLabelFormatFunctionKey | Callable[[int, int  |  None], str]
        Key ``BuiltinSubPlotLabelFormatFunctions`` to retrieve builtin function or user defined
        format function.

    Returns
    -------
    Callable[[int, int | None], str]
        Function to format subplot label.
    """
    if isinstance(format_function, str) and format_function in BuiltinSubPlotLabelFormatFunctions:
        return BuiltinSubPlotLabelFormatFunctions[format_function]
    return format_function  # type:ignore[return-value]


def add_subplot_labels(
    axes: Axis | Axes,
    *,
    label_position: tuple[float, float] = (-0.05, 1.05),
    label_coords: SubPlotLabelCoord = "axes fraction",
    direction: Literal["row", "column"] = "row",
    label_format_template: str = "{}",
    label_format_function: BuiltinSubPlotLabelFormatFunctionKey
    | Callable[[int, int | None], str] = "number",
    fontsize: int = 16,
) -> None:
    """Add labels to all subplots in ``axes`` in a consistent manner.

    Parameters
    ----------
    axes : Axis | Axes
        Axes (subplots) on which the labels should be added.
    label_position : tuple[float, float]
        Position of the label in ``label_coords`` coordinates.
    label_coords : SubPlotLabelCoord
        Coordinate system used for ``label_position``. Defaults to "axes fraction"
    direction : Literal["row", "column"]
        Direct in which the axes should be iterated in. Defaults to "row"
    label_format_template : str
        Template string to inject the return value of ``label_format_function`` into.
        Defaults to "{}"
    label_format_function : BuiltinSubPlotLabelFormatFunctionKey | Callable[[int, int | None], str]
        Function to calculate the label for the axis index and ``axes`` size. Defaults to "number"
    fontsize : int
        Font size used for the label. Defaults to 16
    """
    axes = ensure_axes_array(axes)
    format_function = get_subplot_label_format_function(label_format_function)
    if direction == "column":
        axes = axes.T
    for i, ax in enumerate(axes.flatten(), start=1):
        ax.annotate(
            label_format_template.format(format_function(i, axes.size)),
            xy=label_position,
            xycoords=label_coords,
            fontsize=fontsize,
        )
