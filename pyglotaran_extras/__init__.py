"""Pyglotaran extension package with convenience functionality such as plotting."""

from __future__ import annotations

from pyglotaran_extras.config.config import _find_script_dir_at_import
from pyglotaran_extras.config.config import create_config_schema
from pyglotaran_extras.config.config import load_config
from pyglotaran_extras.config.plot_config import PerFunctionPlotConfig
from pyglotaran_extras.config.plot_config import plot_config_context
from pyglotaran_extras.config.plot_config import use_plot_config
from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.io.setup_case_study import setup_case_study
from pyglotaran_extras.plotting.plot_coherent_artifact import plot_coherent_artifact
from pyglotaran_extras.plotting.plot_concentrations import plot_concentrations
from pyglotaran_extras.plotting.plot_data import plot_data_overview
from pyglotaran_extras.plotting.plot_doas import plot_doas
from pyglotaran_extras.plotting.plot_guidance import plot_guidance
from pyglotaran_extras.plotting.plot_irf_dispersion_center import plot_irf_dispersion_center
from pyglotaran_extras.plotting.plot_overview import plot_overview
from pyglotaran_extras.plotting.plot_overview import plot_simple_overview
from pyglotaran_extras.plotting.plot_residual import plot_residual
from pyglotaran_extras.plotting.plot_spectra import plot_das
from pyglotaran_extras.plotting.plot_spectra import plot_norm_das
from pyglotaran_extras.plotting.plot_spectra import plot_norm_sas
from pyglotaran_extras.plotting.plot_spectra import plot_sas
from pyglotaran_extras.plotting.plot_spectra import plot_spectra
from pyglotaran_extras.plotting.plot_svd import plot_lsv_data
from pyglotaran_extras.plotting.plot_svd import plot_lsv_residual
from pyglotaran_extras.plotting.plot_svd import plot_rsv_data
from pyglotaran_extras.plotting.plot_svd import plot_rsv_residual
from pyglotaran_extras.plotting.plot_svd import plot_sv_data
from pyglotaran_extras.plotting.plot_svd import plot_sv_residual
from pyglotaran_extras.plotting.plot_svd import plot_svd
from pyglotaran_extras.plotting.plot_traces import plot_fitted_traces
from pyglotaran_extras.plotting.plot_traces import select_plot_wavelengths
from pyglotaran_extras.plotting.utils import add_subplot_labels

__all__ = [
    "load_data",
    "setup_case_study",
    "plot_coherent_artifact",
    "plot_concentrations",
    "plot_data_overview",
    "plot_doas",
    "plot_guidance",
    "plot_irf_dispersion_center",
    "plot_overview",
    "plot_simple_overview",
    "plot_residual",
    "plot_das",
    "plot_norm_das",
    "plot_norm_sas",
    "plot_sas",
    "plot_spectra",
    "plot_lsv_data",
    "plot_lsv_residual",
    "plot_rsv_data",
    "plot_rsv_residual",
    "plot_sv_data",
    "plot_sv_residual",
    "plot_svd",
    "plot_fitted_traces",
    "select_plot_wavelengths",
    "add_subplot_labels",
    # Config
    "PerFunctionPlotConfig",
    "plot_config_context",
    "use_plot_config",
    "create_config_schema",
    "CONFIG",
]

__version__ = "0.7.3"

SCRIPT_DIR = _find_script_dir_at_import(__file__)
"""User script dir determined during import."""
CONFIG = load_config(SCRIPT_DIR)
"""Global config instance."""
