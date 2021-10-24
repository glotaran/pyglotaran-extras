from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import matplotlib.pyplot as plt

from pyglotaran_extras.deprecation.deprecation_utils import FIG_ONLY_WARNING
from pyglotaran_extras.deprecation.deprecation_utils import PyglotaranExtrasApiDeprecationWarning
from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.plotting.style import PlotStyle

if TYPE_CHECKING:
    from cycler import Cycler
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import DatasetConvertible


def plot_doas(
    result: DatasetConvertible,
    figsize: tuple[int, int] = (25, 25),
    cycler: Cycler = PlotStyle().cycler,
    figure_only: bool = True,
) -> Figure | tuple[Figure, Axes]:
    """Plot damped oscillations (DOAS).

    Parameters
    ----------
    result: DatasetConvertible
        Result from a pyglotaran optimization as dataset, Path or Result object.
    figsize : tuple[int, int]
        Size of the figure (N, M) in inches., by default (18, 16)
    cycler : Cycler
        Plot style cycler to use., by default PlotStyle().cycler
    figure_only: bool
        Whether or not to only return the figure.
        This is a deprecation helper argument to transition to a consistent return value
        consisting of the :class:`Figure` and the :class:`Axes`, by default True

    Returns
    -------
    Figure|tuple[Figure, Axes]
        If ``figure_only`` is True, Figure object which contains the plots (deprecated).
        If ``figure_only`` is False, Figure object which contains the plots and the Axes.
    """
    dataset = load_data(result)

    # Create M x N plotting grid
    M = 6
    N = 3

    fig, axes = plt.subplots(M, N, figsize=figsize)

    for ax in axes.flatten():
        ax.set_prop_cycle(cycler)

    # Plot data
    dataset.species_associated_spectra.plot.line(x="spectral", ax=axes[0, 0])
    dataset.decay_associated_spectra.plot.line(x="spectral", ax=axes[0, 1])

    if "spectral" in dataset.species_concentration.coords:
        dataset.species_concentration.isel(spectral=0).plot.line(x="time", ax=axes[1, 0])
    else:
        dataset.species_concentration.plot.line(x="time", ax=axes[1, 0])
    axes[1, 0].set_xscale("symlog", linthreshx=1)

    if "dampened_oscillation_associated_spectra" in dataset:
        dataset.dampened_oscillation_cos.isel(spectral=0).sel(time=slice(-1, 10)).plot.line(
            x="time", ax=axes[1, 1]
        )
        dataset.dampened_oscillation_associated_spectra.plot.line(x="spectral", ax=axes[2, 0])
        dataset.dampened_oscillation_phase.plot.line(x="spectral", ax=axes[2, 1])

    dataset.residual_left_singular_vectors.isel(left_singular_value_index=0).plot(ax=axes[0, 2])
    dataset.residual_singular_values.plot.line("ro-", yscale="log", ax=axes[1, 2])
    dataset.residual_right_singular_vectors.isel(right_singular_value_index=0).plot(ax=axes[2, 2])

    interval = int(dataset.spectral.size / 11)
    for i in range(0):
        axi = axes[i % 3, int(i / 3) + 3]
        index = (i + 1) * interval
        dataset.data.isel(spectral=index).plot(ax=axi)
        dataset.residual.isel(spectral=index).plot(ax=axi)
        dataset.fitted_data.isel(spectral=index).plot(ax=axi)

    plt.tight_layout(pad=5, w_pad=2.0, h_pad=2.0)
    if figure_only is True:
        warn(PyglotaranExtrasApiDeprecationWarning(FIG_ONLY_WARNING), stacklevel=2)
        return fig
    else:
        return fig, axes
