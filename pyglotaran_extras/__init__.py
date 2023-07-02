"""Pyglotaran extension package with convenience functionality such as plotting."""
from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.io.setup_case_study import setup_case_study
from pyglotaran_extras.plotting.plot_coherent_artifact import plot_coherent_artifact
from pyglotaran_extras.plotting.plot_data import plot_data_overview
from pyglotaran_extras.plotting.plot_doas import plot_doas
from pyglotaran_extras.plotting.plot_guidance import plot_guidance
from pyglotaran_extras.plotting.plot_irf_dispersion_center import plot_irf_dispersion_center
from pyglotaran_extras.plotting.plot_overview import plot_overview
from pyglotaran_extras.plotting.plot_overview import plot_simple_overview
from pyglotaran_extras.plotting.plot_traces import plot_fitted_traces
from pyglotaran_extras.plotting.plot_traces import select_plot_wavelengths
from pyglotaran_extras.plotting.utils import add_subplot_labels

__all__ = [
    "load_data",
    "setup_case_study",
    "plot_coherent_artifact",
    "plot_data_overview",
    "plot_doas",
    "plot_guidance",
    "plot_irf_dispersion_center",
    "plot_overview",
    "plot_simple_overview",
    "plot_fitted_traces",
    "select_plot_wavelengths",
    "add_subplot_labels",
]

__version__ = "0.7.1"
