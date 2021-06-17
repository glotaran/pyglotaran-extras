from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.plotting.plot_residual import plot_residual
from pyglotaran_extras.plotting.plot_spectra import plot_spectra
from pyglotaran_extras.plotting.plot_svd import plot_svd
from pyglotaran_extras.plotting.plot_traces import plot_traces
from pyglotaran_extras.plotting.style import PlotStyle


def plot_overview(result, center_λ=None, linlog=True, linthresh=1, show_data=False):

    res = load_data(result)

    # Plot dimensions
    M = 4
    N = 3
    fig, ax = plt.subplots(M, N, figsize=(18, 16), constrained_layout=True)

    plot_style = PlotStyle()
    plt.rc("axes", prop_cycle=plot_style.cycler)

    if center_λ is None:  # center wavelength (λ in nm)
        center_λ = min(res.dims["spectral"], round(res.dims["spectral"] / 2))

    # First and second row: concentrations - SAS/EAS - DAS
    plot_traces(res, ax[0, 0], center_λ, linlog=linlog, linthresh=linthresh)
    plot_spectra(res, ax[0:2, 1:3])
    plot_svd(res, ax[2:4, 0:3], linlog=linlog, linthresh=linthresh)
    plot_residual(res, ax[1, 0], linlog=linlog, linthresh=linthresh, show_data=show_data)
    plot_style.set_default_colors()
    plot_style.set_default_fontsize()
    plt.rc("axes", prop_cycle=plot_style.cycler)
    # plt.tight_layout(pad=3, w_pad=4.0, h_pad=4.0)
    return fig


def plot_simple_overview(res, title=None):
    """simple plotting function derived from code from pyglotaran_extras"""
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=16)
    sas = res.species_associated_spectra
    traces = res.species_concentration
    if "spectral" in traces.coords:
        traces.sel(spectral=res.spectral.values[0], method="nearest").plot.line(
            x="time", ax=ax[0, 0]
        )
    else:
        traces.plot.line(x="time", ax=ax[0, 0])
    sas.plot.line(x="spectral", ax=ax[0, 1])
    rLSV = res.residual_left_singular_vectors
    rLSV.isel(left_singular_value_index=range(min(2, len(rLSV)))).plot.line(x="time", ax=ax[1, 0])

    ax[1, 0].set_title("res. LSV")
    rRSV = res.residual_right_singular_vectors
    rRSV.isel(right_singular_value_index=range(min(2, len(rRSV)))).plot.line(
        x="spectral", ax=ax[1, 1]
    )

    ax[1, 1].set_title("res. RSV")
    res.data.plot(x="time", ax=ax[0, 2])
    ax[0, 2].set_title("data")
    res.residual.plot(x="time", ax=ax[1, 2])
    ax[1, 2].set_title("residual")
    plt.show(block=False)
    return fig


if __name__ == "__main__":
    import sys

    result_path = Path(sys.argv[1])
    res = xr.open_dataset(result_path)
    print(res)

    fig = plot_overview(res)
    if len(sys.argv) > 2:
        fig.savefig(sys.argv[2], bbox_inches="tight")
        print(f"Saved figure to: {sys.argv[2]}")
    else:
        plt.show(block=False)
        input("press <ENTER> to continue")
