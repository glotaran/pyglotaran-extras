from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyglotaran_extras.plotting.style import PlotStyle

if TYPE_CHECKING:
    import xarray as xr
    from cycler import Cycler
    from matplotlib.axis import Axis
    from matplotlib.pyplot import Axes


def plot_spectra(res: xr.Dataset, axes: Axes, cycler: Cycler = PlotStyle().cycler) -> None:
    plot_sas(res, axes[0, 0])
    plot_das(res, axes[0, 1])
    plot_norm_sas(res, axes[1, 0])
    plot_norm_das(res, axes[1, 1])


def plot_sas(
    res: xr.Dataset, ax: Axis, title: str = "SAS", cycler: Cycler = PlotStyle().cycler
) -> None:
    ax.set_prop_cycle(cycler)
    keys = [
        v for v in res.data_vars if v.startswith(("species_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        sas = res[key]
        sas.plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()


def plot_norm_sas(
    res: xr.Dataset, ax: Axis, title: str = "norm SAS", cycler: Cycler = PlotStyle().cycler
) -> None:
    ax.set_prop_cycle(cycler)
    keys = [
        v for v in res.data_vars if v.startswith(("species_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        sas = res[key]
        # sas = res.species_associated_spectra
        (sas / np.abs(sas).max(dim="spectral")).plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()


def plot_das(
    res: xr.Dataset, ax: Axis, title: str = "DAS", cycler: Cycler = PlotStyle().cycler
) -> None:
    ax.set_prop_cycle(cycler)
    keys = [
        v for v in res.data_vars if v.startswith(("decay_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        das = res[key]
        das.plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()


def plot_norm_das(
    res: xr.Dataset, ax: Axis, title: str = "norm DAS", cycler: Cycler = PlotStyle().cycler
) -> None:
    ax.set_prop_cycle(cycler)
    keys = [
        v for v in res.data_vars if v.startswith(("decay_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        das = res[key]
        (das / np.abs(das).max(dim="spectral")).plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()
