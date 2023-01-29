"""Module containing DOAS (Damped Oscillation) plotting functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from cycler import Cycler

from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import abs_max
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import calculate_ticks_in_units_of_pi
from pyglotaran_extras.plotting.utils import extract_irf_location
from pyglotaran_extras.plotting.utils import shift_time_axis_by_irf_location

if TYPE_CHECKING:
    from glotaran.project.result import Result
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import DatasetConvertible


def plot_doas(
    dataset: DatasetConvertible | Result,
    *,
    damped_oscillation: list[str] | None = None,
    time_range: tuple[float, float] | None = None,
    spectral: float = 0,
    center_λ: float | None = None,
    main_irf_nr: int = 0,
    normalize: bool = True,
    figsize: tuple[int, int] = (20, 5),
    show_zero_line: bool = True,
    cycler: Cycler | None = PlotStyle().cycler,
    oscillation_type: Literal["cos", "sin"] = "cos",
    title: str | None = "Damped oscillations",
) -> tuple[Figure, Axes]:
    """Plot DOAS (Damped Oscillation) related data of the optimization result.

    Parameters
    ----------
    dataset: DatasetConvertible | Result
        Result dataset from a pyglotaran optimization.
    damped_oscillation: list[str] | None
        List of oscillation names which should be plotted.
        Defaults to None which means that all oscillations will be plotted.
    time_range: tuple[float, float] | None
        Start and end time for the Oscillation plot. Defaults to None which means that
        the full time range is used.
    spectral: float
        Value of the spectral axis that should be used to select the data for the Oscillation
        plot this value does not need to be an exact existing value and only has effect if the
        IRF has dispersion. Defaults to 0 which means that the Oscillation plot at lowest
        spectral value will be shown.
    center_λ: float | None = None,
        Center wavelength (λ in nm) to select the IRF location by which the Oscillation
        plot is shifted. Defaults to None.
    main_irf_nr: int
        Index of the main ``irf`` component when using an ``irf``
        parametrized with multiple peaks. Defaults to 0.
    normalize: bool
        Whether or not to normalize the DOAS spectra plot. If the DOAS spectra is normalized,
        the Oscillation is scaled with the reciprocal of the normalization to compensate for this.
        Defaults to True.
    figsize: tuple[int, int]
        Size of the figure (N, M) in inches. Defaults to (20, 5)
    show_zero_line: bool
        Whether or not to add a horizontal line at zero. Defaults to True
    cycler: Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler
    oscillation_type: Literal["cos", "sin"]
        Type of the oscillation to show in the oscillation plot. Defaults to "cos"
    title: str | None
        Title of the figure. Defaults to "Damped oscillations"

    Returns
    -------
    tuple[Figure, Axes]
        Figure object which contains the plots and the Axes.

    See Also
    --------
    calculate_ticks_in_units_of_pi
    """
    dataset = load_data(dataset, _stacklevel=3)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    add_cycler_if_not_none(axes, cycler)

    time_sel_kwargs = (
        {"time": slice(time_range[0], time_range[1])} if time_range is not None else {}
    )
    osc_sel_kwargs = (
        {"damped_oscillation": damped_oscillation} if damped_oscillation is not None else {}
    )

    irf_location = extract_irf_location(dataset, center_λ, main_irf_nr)

    oscillations = shift_time_axis_by_irf_location(
        dataset[f"damped_oscillation_{oscillation_type}"]
        .sel(spectral=spectral, method="nearest")
        .sel(**osc_sel_kwargs),
        irf_location,
    )
    oscillations_spectra = dataset["damped_oscillation_associated_spectra"].sel(**osc_sel_kwargs)

    damped_oscillation_phase = dataset["damped_oscillation_phase"].sel(**osc_sel_kwargs)

    osc_max = abs_max((oscillations - 1), result_dims="damped_oscillation")
    spectra_max = abs_max(oscillations_spectra, result_dims="damped_oscillation")
    scales = np.sqrt(osc_max * spectra_max)

    norm_factor = scales.max() if normalize is True else 1

    ((oscillations - 1) / osc_max * scales * norm_factor).sel(**time_sel_kwargs).plot.line(
        x="time", ax=axes[0]
    )

    (oscillations_spectra / spectra_max * scales / norm_factor).plot.line(x="spectral", ax=axes[1])

    damped_oscillation_phase.plot.line(x="spectral", ax=axes[2])

    axes[0].set_title(f"{oscillation_type.capitalize()} Oscillations")
    axes[1].set_title("Spectra")
    axes[2].set_title("Phases")

    axes[1].set_ylabel("Normalized DOAS" if normalize is True else "DOAS")

    axes[2].set_yticks(
        *calculate_ticks_in_units_of_pi(damped_oscillation_phase), rotation="horizontal"
    )
    axes[2].set_ylabel("Phase (π)")

    axes[1].get_legend().remove()
    axes[2].get_legend().remove()

    if show_zero_line is True:
        [ax.axhline(0, color="k", linewidth=1) for ax in axes.flatten()]

    if title:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    return fig, axes
