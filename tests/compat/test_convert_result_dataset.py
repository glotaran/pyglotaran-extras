"""Tests for ``pyglotaran_extras.compat.convert_result_dataset``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from glotaran.optimization.objective import OptimizationResult
from glotaran.optimization.objective import OptimizationResultMetaData

from pyglotaran_extras.compat.convert_result_dataset import DAMPED_OSCILLATION_ELEMENT_UID
from pyglotaran_extras.compat.convert_result_dataset import build_compat_dataset
from pyglotaran_extras.plotting.plot_spectra import plot_spectra
from pyglotaran_extras.plotting.utils import extract_dataset_scale


def test_global_result_exposes_legacy_spectra_names() -> None:
    """Global element data is exposed under the legacy SAS and concentration names."""
    time = np.arange(3)
    spectral = np.arange(2)
    amplitude_label = ["s1", "s2"]
    input_data = xr.DataArray(
        np.zeros((2, 3)),
        dims=("spectral", "time"),
        coords={"spectral": spectral, "time": time},
    )
    element_data = xr.Dataset(
        {
            "global_concentrations": xr.DataArray(
                np.ones((3, 2)),
                dims=("time", "amplitude_label"),
                coords={"time": time, "amplitude_label": amplitude_label},
            ),
            "model_concentrations": xr.DataArray(
                np.ones((2, 2)),
                dims=("spectral", "amplitude_label"),
                coords={"spectral": spectral, "amplitude_label": amplitude_label},
            ),
        }
    )
    result = OptimizationResult(
        input_data=input_data,
        residuals=xr.zeros_like(input_data),
        elements={"dataset": element_data},
        fit_decomposition=None,
        meta=OptimizationResultMetaData(
            global_dimension="time",
            model_dimension="spectral",
            root_mean_square_error=0,
        ),
    )

    converted = build_compat_dataset(result)

    assert converted["species_concentration"].dims == ("time", "species_model")
    assert converted["species_spectra"].dims == ("spectral", "species_model")

    _, axes = plt.subplots(2, 2)
    plot_spectra(converted, axes)
    assert len(axes[0, 1].lines) == 3
    assert len(axes[1, 1].lines) == 3


def test_damped_oscillation_labels_stay_selectable() -> None:
    """Damped oscillations keep their labels on the renamed dimension."""
    time = np.arange(3)
    spectral = np.arange(2)
    oscillation = ["osc1", "osc2"]
    input_data = xr.DataArray(
        np.zeros((2, 3)),
        dims=("spectral", "time"),
        coords={"spectral": spectral, "time": time},
    )
    element_data = xr.Dataset(
        {
            "cos_concentrations": xr.DataArray(np.ones((3, 2)), dims=("time", "oscillation")),
            "amplitudes": xr.DataArray(np.ones((2, 2)), dims=("spectral", "oscillation")),
        },
        coords={"time": time, "spectral": spectral, "oscillation": oscillation},
        attrs={"element_uid": DAMPED_OSCILLATION_ELEMENT_UID},
    )
    result = OptimizationResult(
        input_data=input_data,
        residuals=xr.zeros_like(input_data),
        elements={"doas": element_data},
        fit_decomposition=None,
        meta=OptimizationResultMetaData(
            global_dimension="time",
            model_dimension="spectral",
            root_mean_square_error=0,
        ),
    )

    converted = build_compat_dataset(result)

    assert "oscillation" not in converted.dims
    assert converted["damped_oscillation_associated_spectra"].sel(
        damped_oscillation="osc2"
    ).dims == ("spectral",)


def test_dataset_scale_is_preserved() -> None:
    """The v0.8 dataset scale is available to the plotting functions."""
    input_data = xr.DataArray(
        np.zeros((2, 3)),
        dims=("spectral", "time"),
        coords={"spectral": np.arange(2), "time": np.arange(3)},
    )
    result = OptimizationResult(
        input_data=input_data,
        residuals=xr.zeros_like(input_data),
        elements={},
        fit_decomposition=None,
        meta=OptimizationResultMetaData(
            global_dimension="time",
            model_dimension="spectral",
            root_mean_square_error=0,
            scale=2.5,
        ),
    )

    assert extract_dataset_scale(build_compat_dataset(result)) == 2.5
