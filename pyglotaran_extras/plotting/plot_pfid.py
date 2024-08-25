"""Module containing PFID (Perturbed Free Induction Decay) plotting functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import abs_max
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.plotting.utils import calculate_ticks_in_units_of_pi
from pyglotaran_extras.plotting.utils import condense_numbered_strings
from pyglotaran_extras.plotting.utils import extract_irf_location
from pyglotaran_extras.plotting.utils import shift_time_axis_by_irf_location

if TYPE_CHECKING:
    from cycler import Cycler
    from glotaran.project.result import Result
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import DatasetConvertible


def plot_pfid(  # noqa: C901
    dataset: DatasetConvertible | Result,
    *,
    names: list[str] | None = None,
    oscillation_type: Literal["cos", "sin"] = "cos",
    show_clps: bool = False,
    time_range: tuple[float, float] | None = None,
    spectral: float | None = None,
    main_irf_nr: int | None = 0,
    normalize: bool = False,
    figsize: tuple[float, float] | None = None,
    show_zero_line: bool = True,
    cycler: Cycler | None = PlotStyle().cycler,
    title: str | None = "Perturbed Free Induction Decays",
    legend_format_string: str = r"{label}: $\nu$={frequency:.0f}, $\gamma$={rate:.1f}",
) -> tuple[Figure, Axes]:
    r"""Plot PFID (Perturbed Free Induction Decay) related data of the optimization result.

    Parameters
    ----------
    dataset : DatasetConvertible | Result
        Result dataset from a pyglotaran optimization.
    names : list[str] | None
        List of oscillation names which should be plotted.
        Defaults to None which means that all oscillations will be plotted.
    oscillation_type : Literal["cos", "sin"]
        Type of the oscillation to show in the oscillation plot. Defaults to "cos"
    show_clps : bool, optional
        If True, plots additional conditional linear parameters (CLPs), in which case the
        oscillation_type argument is ignored (both cos and sin are plotted). Defaults to False.
    time_range : tuple[float, float] | None
        Start and end time for the Oscillation plot, if ``main_irf_nr`` is not None the value are
        relative to the IRF location. Defaults to None which means that the full time range is
        used.
    spectral : float | None
        Value of the spectral axis that should be used to select the data for the Oscillation
        plot this value does not need to be an exact existing value and only has effect if the
        IRF has dispersion. If None the Oscillation plot at lowest spectral value will be shown.
    main_irf_nr : int | None
        Index of the main ``irf`` component when using an ``irf`` parametrized with multiple peaks
        and is used to shift the time axis. If it is none ``None`` the shifting will be
        deactivated. Defaults to 0.
    normalize : bool
        Whether or not to normalize the PFID spectra plot. If the PFID spectra is normalized,
        the Oscillation is scaled with the reciprocal of the normalization to compensate for this.
        Defaults to False.
    figsize : tuple[float, float] | None
        Size of the figure (N, M) in inches. Defaults to None which then uses
        (20,5) if show_clps=False, (20, 10) if show_clps=True.
    show_zero_line : bool
        Whether or not to add a horizontal line at zero. Defaults to True
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler
    title : str | None
        Title of the figure. Defaults to "Perturbed Free Induction Decays"
    legend_format_string : str
        Format string for each entry in the legend of the oscillation plot. Possible values which
        can be replaced are ``label`` (label of the oscillation in the model definition),
        ``frequency`` (ν) and ``rate`` (γ). Use ``""`` to remove the legend. Defaults to
        ``r"{label}: $\nu$={frequency:.0f}, $\gamma$={rate:.1f}"``

    Returns
    -------
    tuple[Figure, Axes]
        Figure object which contains the plots and the Axes.

    See Also
    --------
    calculate_ticks_in_units_of_pi
    """
    dataset = load_data(dataset, _stacklevel=3)
    if figsize is None:
        figsize = (20, 10) if show_clps else (20, 5)
    fig, axes = plt.subplots(2 if show_clps else 1, 3, figsize=figsize)

    add_cycler_if_not_none(axes, cycler)

    time_sel_kwargs = {"time": slice(time_range[0], time_range[1])} if time_range else {}
    names = dataset.pfid.to_numpy() if names is None else names
    osc_sel_kwargs = {"pfid": names}
    if spectral is None and "spectral" in dataset.coords:
        spectral = dataset.coords["spectral"].min().item()
    irf_location = extract_irf_location(dataset, spectral, main_irf_nr)

    pfid_phase = dataset["pfid_phase"].sel(**osc_sel_kwargs)
    oscillations_spectra = dataset["pfid_associated_spectra"].sel(**osc_sel_kwargs)
    oscillation_types = ["cos", "sin"] if show_clps else [oscillation_type]

    for axes_index, osc_type in enumerate(oscillation_types):
        oscillations = dataset[f"pfid_{osc_type}"]

        if "spectral" in oscillations.coords:
            oscillations = oscillations.sel(spectral=spectral, method="nearest")

        oscillations = shift_time_axis_by_irf_location(
            oscillations.sel(**osc_sel_kwargs), irf_location, _internal_call=True
        )
        osc_max = abs_max(oscillations, result_dims="pfid")
        spectra_max = abs_max(oscillations_spectra, result_dims="pfid")
        scales = np.sqrt(osc_max * spectra_max)
        norm_factor = scales.max() if normalize else 1

        oscillations_to_plot = (oscillations / osc_max * scales * norm_factor).sel(
            **time_sel_kwargs
        )

        for oscillation_label in oscillations_to_plot.pfid.to_numpy():
            oscillation = oscillations_to_plot.sel(pfid=[oscillation_label])
            frequency = oscillation.pfid_frequency.item()
            rate = oscillation.pfid_rate.item()
            oscillation.plot.line(
                x="time",
                ax=axes[axes_index, 0] if show_clps else axes[0],
                label=legend_format_string.format(
                    label=oscillation_label, frequency=frequency, rate=rate
                ),
            )

    spectra_ax = axes[0, 2] if show_clps else axes[1]
    phases_ax = axes[1, 2] if show_clps else axes[2]

    (oscillations_spectra / spectra_max * scales / norm_factor).plot.line(
        x="spectral", ax=spectra_ax
    )
    pfid_phase.plot.line(x="spectral", ax=phases_ax)

    spectra_ax.set_title("Spectra")
    spectra_ax.set_ylabel("Normalized PFID" if normalize else "PFID")

    phases_ax.set_title("Phases")
    phases_ax.set_yticks(*calculate_ticks_in_units_of_pi(pfid_phase), rotation="horizontal")
    phases_ax.set_ylabel("Phase (π)")

    if show_clps:
        for axes_index, osc_type in enumerate(oscillation_types):
            for pfidname in names:
                clp_label = f"{pfidname}_{osc_type}"
                dataset["clp"].sel(clp_label=clp_label).plot(ax=axes[axes_index, 1])
            axes[axes_index, 1].set_title(f"clps={condense_numbered_strings(names)}")

        axes[0, 0].set_title(f"Cos Oscillations {spectral}")
        axes[1, 0].set_title(f"Sin Oscillations {spectral}")
    else:
        axes[0].set_title(f"{oscillation_type.capitalize()} Oscillations {spectral}")

    for ax in axes.flatten():
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    if legend_format_string:
        axes[0].legend() if not show_clps else axes[0, 0].legend()

    if show_zero_line:
        [ax.axhline(0, color="k", linewidth=1) for ax in axes.flatten()]

    if title:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    return fig, axes
