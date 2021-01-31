from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from ..io.load_data import load_data
from .plot_residual import plot_residual
from .plot_spectra import plot_spectra
from .plot_svd import plot_svd
from .plot_traces import plot_traces
from .style import PlotStyle


def plot_overview(result, center_λ=None, linlog=True, linthresh=1):

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
    plot_residual(res, ax[1, 0], linlog=linlog, linthresh=linthresh)
    plot_style.set_default_colors()
    plot_style.set_default_fontsize()
    plt.rc("axes", prop_cycle=plot_style.cycler)
    # plt.tight_layout(pad=3, w_pad=4.0, h_pad=4.0)
    plt.draw()
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
