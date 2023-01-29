"""Module containing coherent artifact plot functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.plotting.utils import abs_max
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none

if TYPE_CHECKING:
    from cycler import Cycler
    from glotaran.project.result import Result
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import DatasetConvertible


def plot_coherent_artifact(
    dataset: DatasetConvertible | Result,
    *,
    time_range: tuple[float, float] | None = None,
    spectral: float = 0,
    normalize: bool = True,
    figsize: tuple[int, int] = (18, 7),
    show_zero_line: bool = True,
    cycler: Cycler | None = None,
    title: str | None = "Coherent Artifact",
) -> tuple[Figure, Axes]:
    """Plot coherent artifact as IRF derivative components over time and IRFAS over spectral dim.

    The IRFAS are the IRF (Instrument Response Function) Associated Spectra.

    Parameters
    ----------
    dataset: DatasetConvertible | Result
        Result dataset from a pyglotaran optimization.
    time_range: tuple[float, float] | None
        Start and end time for the IRF derivative plot. Defaults to None which means that
        the full time range is used.
    spectral: float
        Value of the spectral axis that should be used to select the data for the IRF derivative
        plot this value does not need to be an exact existing value and only has effect if the
        IRF has dispersion. Defaults to 0 which means that the IRF derivative plot at lowest
        spectral value will be shown.
    normalize: bool
        Whether or not to normalize the IRF derivative plot. If the IRF derivative is normalized,
        the IRFAS is scaled with the reciprocal of the normalization to compensate for this.
        Defaults to True.
    figsize: tuple[int, int]
        Size of the figure (N, M) in inches. Defaults to (18, 7).
    show_zero_line: bool
        Whether or not to add a horizontal line at zero. Defaults to True.
    cycler: Cycler | None
        Plot style cycler to use. Defaults to None, which means that the matplotlib default style
        will be used.
    title: str | None
        Title of the figure. Defaults to "Coherent Artifact".

    Returns
    -------
    tuple[Figure, Axes]
        Figure object which contains the plots and the Axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    add_cycler_if_not_none(axes, cycler)
    dataset = load_data(dataset, _stacklevel=3)

    if (
        "coherent_artifact_response" not in dataset
        or "coherent_artifact_associated_spectra" not in dataset
    ):
        warn(
            UserWarning(f"Dataset does not contain coherent artifact data:\n {dataset.data_vars}"),
            stacklevel=2,
        )
        return fig, axes

    irf_max = abs_max(dataset.coherent_artifact_response, result_dims=("coherent_artifact_order"))
    irfas_max = abs_max(
        dataset.coherent_artifact_associated_spectra, result_dims=("coherent_artifact_order")
    )
    scales = np.sqrt(irfas_max * irf_max)
    norm_factor = 1
    irf_y_label = "amplitude"
    irfas_y_label = "ΔA"

    if normalize is True:
        norm_factor = scales.max()
        irf_y_label = f"normalized {irf_y_label}"

    plot_slice_irf = (
        dataset.coherent_artifact_response.sel(spectral=spectral, method="nearest")
        / irf_max
        * scales
        / norm_factor
    )
    irf_sel_kwargs = (
        {"time": slice(time_range[0], time_range[1])} if time_range is not None else {}
    )
    plot_slice_irf.sel(**irf_sel_kwargs).plot.line(x="time", ax=axes[0])
    axes[0].set_title("IRF Derivatives")
    axes[0].set_ylabel(f"{irf_y_label} (a.u.)")

    plot_slice_irfas = (
        dataset.coherent_artifact_associated_spectra / irfas_max * scales * norm_factor
    )
    plot_slice_irfas.plot.line(x="spectral", ax=axes[1])
    axes[1].get_legend().remove()
    axes[1].set_title("IRFAS")
    axes[1].set_ylabel(f"{irfas_y_label} (mOD)")

    if show_zero_line is True:
        axes[0].axhline(0, color="k", linewidth=1)
        axes[1].axhline(0, color="k", linewidth=1)

    #
    if dataset.coords["coherent_artifact_order"][0] == 1:
        axes[0].legend(
            [f"{int(ax_label)-1}" for ax_label in dataset.coords["coherent_artifact_order"]],
            title="coherent_artifact_order",
        )
    if title:
        fig.suptitle(title, fontsize=16)
    return fig, axes
