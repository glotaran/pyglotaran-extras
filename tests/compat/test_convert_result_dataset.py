from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from glotaran.optimization.objective import OptimizationResult
from glotaran.optimization.objective import OptimizationResultMetaData

from pyglotaran_extras.compat.convert_result_dataset import build_compat_dataset
from pyglotaran_extras.plotting.plot_spectra import plot_spectra


def test_global_result_exposes_legacy_spectra_names() -> None:
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