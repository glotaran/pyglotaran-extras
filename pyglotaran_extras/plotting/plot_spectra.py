import numpy as np


def plot_spectra(res, axes):
    plot_sas(res, axes[0, 0])
    plot_das(res, axes[0, 1])
    plot_norm_sas(res, axes[1, 0])
    plot_norm_das(res, axes[1, 1])


def plot_sas(res, ax, title="SAS"):
    keys = [
        v for v in res.data_vars if v.startswith(("species_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        sas = res[key]
        sas.plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()


def plot_norm_sas(res, ax, title="norm SAS"):
    keys = [
        v for v in res.data_vars if v.startswith(("species_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        sas = res[key]
        # sas = res.species_associated_spectra
        (sas / np.abs(sas).max(dim="spectral")).plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()


def plot_das(res, ax, title="DAS"):
    keys = [
        v for v in res.data_vars if v.startswith(("decay_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        das = res[key]
        das.plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()


def plot_norm_das(res, ax, title="norm DAS"):
    keys = [
        v for v in res.data_vars if v.startswith(("decay_associated_spectra", "species_spectra"))
    ]
    for key in keys:
        das = res[key]
        (das / np.abs(das).max(dim="spectral")).plot.line(x="spectral", ax=ax)
        ax.set_title(title)
        ax.get_legend().remove()
