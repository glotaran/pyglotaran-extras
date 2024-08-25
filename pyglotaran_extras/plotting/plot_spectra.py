"""Module containing spectra plotting functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyglotaran_extras.config.plot_config import use_plot_config
from pyglotaran_extras.plotting.style import PlotStyle
from pyglotaran_extras.plotting.utils import add_cycler_if_not_none
from pyglotaran_extras.types import Unset

if TYPE_CHECKING:
    import xarray as xr
    from cycler import Cycler
    from matplotlib.axis import Axis
    from matplotlib.pyplot import Axes

    from pyglotaran_extras.types import UnsetType


@use_plot_config(exclude_from_config=("cycler", "das_cycler"))
def plot_spectra(
    res: xr.Dataset,
    axes: Axes,
    cycler: Cycler | None = PlotStyle().cycler,
    show_zero_line: bool = True,
    das_cycler: Cycler | None | UnsetType = Unset,
) -> None:
    """Plot spectra such as SAS and DAS as well as their normalize version on ``axes``.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    axes : Axes
        Axes to plot the spectra on (needs to be at least 2x2).
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_zero_line : bool
        Whether or not to add a horizontal line at zero. Defaults to True.
    das_cycler : Cycler | None | UnsetType
        Plot style cycler to use for DAS plots. Defaults to ``Unset`` which means that the value
        of ``cycler`` is used.
    """
    if das_cycler is Unset:
        das_cycler = cycler

    plot_sas(res, axes[0, 0], cycler=cycler, show_zero_line=show_zero_line)
    plot_das(res, axes[0, 1], cycler=das_cycler, show_zero_line=show_zero_line)
    plot_norm_sas(res, axes[1, 0], cycler=cycler, show_zero_line=show_zero_line)
    plot_norm_das(res, axes[1, 1], cycler=das_cycler, show_zero_line=show_zero_line)


@use_plot_config(exclude_from_config=("cycler",))
def plot_sas(
    res: xr.Dataset,
    ax: Axis,
    title: str = "SAS",
    cycler: Cycler | None = PlotStyle().cycler,
    show_zero_line: bool = True,
) -> None:
    """Plot SAS (Species Associated Spectra) on ``ax``.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    title : str
        Title of the plot. Defaults to "SAS".
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_zero_line : bool
        Whether or not to add a horizontal line at zero. Defaults to True.
    """
    add_cycler_if_not_none(ax, cycler)
    keys = [
        v for v in res.data_vars if v.startswith(("species_associated_spectra", "species_spectra"))
    ]
    for key in reversed(keys):
        sas = res[key]
        for zorder, species in zip(range(100)[::-1], sas.coords["species"], strict=False):
            sas.sel(species=species).plot.line(x="spectral", ax=ax, zorder=zorder)
        ax.set_title(title)
    if show_zero_line is True:
        ax.axhline(0, color="k", linewidth=1)


@use_plot_config(exclude_from_config=("cycler",))
def plot_norm_sas(
    res: xr.Dataset,
    ax: Axis,
    title: str = "norm SAS",
    cycler: Cycler | None = PlotStyle().cycler,
    show_zero_line: bool = True,
) -> None:
    """Plot normalized SAS (Species Associated Spectra) on ``ax``.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    title : str
        Title of the plot. Defaults to "norm SAS".
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_zero_line : bool
        Whether or not to add a horizontal line at zero. Defaults to True.
    """
    add_cycler_if_not_none(ax, cycler)
    keys = [
        v for v in res.data_vars if v.startswith(("species_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        sas = res[key]
        for zorder, species in zip(range(100)[::-1], sas.coords["species"], strict=False):
            (sas / np.abs(sas).max(dim="spectral")).sel(species=species).plot.line(
                x="spectral", ax=ax, zorder=zorder
            )
        ax.set_title(title)
    if show_zero_line is True:
        ax.axhline(0, color="k", linewidth=1)


@use_plot_config(exclude_from_config=("cycler",))
def plot_das(
    res: xr.Dataset,
    ax: Axis,
    title: str = "DAS",
    cycler: Cycler | None = PlotStyle().cycler,
    show_zero_line: bool = True,
) -> None:
    """Plot DAS (Decay Associated Spectra) on ``ax``.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    title : str
        Title of the plot. Defaults to "DAS".
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_zero_line : bool
        Whether or not to add a horizontal line at zero. Defaults to True.
    """
    add_cycler_if_not_none(ax, cycler)
    keys = [
        v for v in res.data_vars if v.startswith(("decay_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        das = res[key]
        das.plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()
    if show_zero_line is True:
        ax.axhline(0, color="k", linewidth=1)


@use_plot_config(exclude_from_config=("cycler",))
def plot_norm_das(
    res: xr.Dataset,
    ax: Axis,
    title: str = "norm DAS",
    cycler: Cycler | None = PlotStyle().cycler,
    show_zero_line: bool = True,
) -> None:
    """Plot normalized DAS (Decay Associated Spectra) on ``ax``.

    Parameters
    ----------
    res : xr.Dataset
        Result dataset
    ax : Axis
        Axis to plot on.
    title : str
        Title of the plot. Defaults to "norm DAS".
    cycler : Cycler | None
        Plot style cycler to use. Defaults to PlotStyle().cycler.
    show_zero_line : bool
        Whether or not to add a horizontal line at zero. Defaults to True.
    """
    add_cycler_if_not_none(ax, cycler)
    keys = [
        v for v in res.data_vars if v.startswith(("decay_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        das = res[key]
        (das / np.abs(das).max(dim="spectral")).plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()
    if show_zero_line is True:
        ax.axhline(0, color="k", linewidth=1)
