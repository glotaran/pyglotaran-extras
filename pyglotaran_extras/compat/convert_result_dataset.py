"""Convert a new pyglotaran (result) dataset to a version compatible with pyglotaran-extras."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr
from glotaran.optimization.objective import OptimizationResult
from glotaran.project.result import Result

from pyglotaran_extras.compat.compat_result import CompatResult

if TYPE_CHECKING:
    pass


# Element UID constants
KINETIC_ELEMENT_UID = "glotaran.builtin.elements.kinetic.element.KineticElement"
DAMPED_OSCILLATION_ELEMENT_UID = (
    "glotaran.builtin.elements.damped_oscillation.element.DampedOscillationElement"
)
DAMPED_OSCILLATION_ELEMENT_UID = (
    "glotaran.builtin.elements.damped_oscillation.element.DampedOscillationElement"
)
COHERENT_ARTIFACT_ELEMENT_UID = (
    "glotaran.builtin.elements.coherent_artifact.element.CoherentArtifactElement"
)

def add_kinetic_data(
    target_dataset: xr.Dataset,
    optimization_result: OptimizationResult,
    element_name: str,
) -> None:
    """Add kinetic element data to the target dataset in-place.

    Parameters
    ----------
    target_dataset : xr.Dataset
        The target dataset to add data to (modified in-place).
    optimization_result : OptimizationResult
        The optimization result containing element data.
    element_name : str
        The name of the kinetic element.
    """
    if element_name not in optimization_result.elements:
        return

    element_ds = optimization_result.elements[element_name]

    # Species concentration
    target_dataset[f"species_concentration_{element_name}"] = element_ds[
        "concentrations"
    ].rename({"compartment": f"species_{element_name}"})
    # Also add without suffix for backward compatibility (first element only)
    if "species_concentration" not in target_dataset:
        target_dataset["species_concentration"] = element_ds["concentrations"].rename({"compartment": f"species_{element_name}"})

    # Species associated spectra (SAS)
    target_dataset[f"species_associated_spectra_{element_name}"] = element_ds["amplitudes"].rename({"compartment": f"species_{element_name}"})

    # Decay associated spectra (DAS) / kinetic associated
    target_dataset[f"decay_associated_spectra_{element_name}"] = element_ds["kinetic_amplitudes"].isel(activation=0)


def add_damped_oscillation_data(
    target_dataset: xr.Dataset,
    optimization_result: OptimizationResult,
    element_name: str,
) -> None:
    """Add damped oscillation element data to the target dataset in-place.

    Parameters
    ----------
    target_dataset : xr.Dataset
        The target dataset to add data to (modified in-place).
    optimization_result : OptimizationResult
        The optimization result containing element data.
    element_name : str
        The name of the damped oscillation element.
    """
    if element_name not in optimization_result.elements:
        return

    element_ds = optimization_result.elements[element_name]

    # Variable mapping from new names to old names
    var_mapping = {
        "cos_concentrations": "damped_oscillation_cos",
        "sin_concentrations": "damped_oscillation_sin",
        "amplitudes": "damped_oscillation_associated_spectra",
        "phase_amplitudes": "damped_oscillation_phase",
    }

    # Coordinate/dimension renaming
    coord_mapping = {
        "oscillation": "damped_oscillation",
    }

    for src_name, dst_name in var_mapping.items():
        if src_name in element_ds:
            da = element_ds[src_name]
            # Rename dimensions if needed
            rename_dims = {k: v for k, v in coord_mapping.items() if k in da.dims}
            if rename_dims:
                da = da.rename(rename_dims)
            target_dataset[dst_name] = da

    # Copy oscillation-related coordinates
    if "oscillation" in element_ds.coords:
        target_dataset.coords["damped_oscillation"] = element_ds.coords["oscillation"]
    if "oscillation_frequency" in element_ds.coords:
        # Need to handle this as a coordinate on the damped_oscillation dimension
        freq_data = element_ds.coords["oscillation_frequency"]
        if "oscillation" in freq_data.dims:
            freq_data = freq_data.rename({"oscillation": "damped_oscillation"})
        target_dataset.coords["damped_oscillation_frequency"] = freq_data
    if "oscillation_rate" in element_ds.coords:
        rate_data = element_ds.coords["oscillation_rate"]
        if "oscillation" in rate_data.dims:
            rate_data = rate_data.rename({"oscillation": "damped_oscillation"})
        target_dataset.coords["damped_oscillation_rate"] = rate_data

def add_coherent_artifact_data(
    target_dataset: xr.Dataset,
    optimization_result: OptimizationResult,
    coherent_artifact_name: str,
) -> None:
    """Add coherent artifact element data to the target dataset in-place.

    Parameters
    ----------
    target_dataset : xr.Dataset
        The target dataset to add data to (modified in-place).
    optimization_result : OptimizationResult
        The optimization result containing element data.
    coherent_artifact_name : str
        The name of the coherent artifact element.
    """
    if coherent_artifact_name not in optimization_result.elements:
        return

    element_ds = optimization_result.elements[coherent_artifact_name]

    # Coherent artifact concentrations
    target_dataset[f"coherent_artifact_response"] = element_ds[
        "concentrations"
    ].rename({"derivative": "coherent_artifact_order"})
    # Coherent artifact amplitudes
    target_dataset[f"coherent_artifact_associated_spectra"] = element_ds[
        "amplitudes"
    ].rename({"derivative": "coherent_artifact_order"})

def add_activation_data(
    target_dataset: xr.Dataset,
    optimization_result: OptimizationResult,
    activation_name: str,
) -> None:
    """Add activation (IRF) data to the target dataset in-place.

    Parameters
    ----------
    target_dataset : xr.Dataset
        The target dataset to add data to (modified in-place).
    optimization_result : OptimizationResult
        The optimization result containing activation data.
    activation_name : str
        The name of the activation.
    """
    if activation_name not in optimization_result.activations:
        return

    activation_ds = optimization_result.activations[activation_name]

    # IRF center
    if "center" in activation_ds.attrs:
        target_dataset["irf_center"] = activation_ds.attrs["center"]

    # IRF width
    if "width" in activation_ds.attrs:
        target_dataset["irf_width"] = activation_ds.attrs["width"]

    # IRF scale
    if "scale" in activation_ds.attrs:
        target_dataset["irf_scale"] = activation_ds.attrs["scale"]

    # IRF function
    if "trace" in activation_ds:
        target_dataset["irf"] = activation_ds["trace"]

    # IRF dispersion (center location)
    if "center" in activation_ds:
        target_dataset["center_dispersion_1"] = activation_ds["center"]
        target_dataset["center_dispersion_1"] = xr.DataArray([activation_ds["center"]], dims=["irf_nr","spectral"])


def build_compat_dataset(optimization_result: OptimizationResult) -> xr.Dataset:
    """Build a v0.7-compatible flat dataset from an OptimizationResult.

    Parameters
    ----------
    optimization_result : OptimizationResult
        The optimization result to convert.

    Returns
    -------
    xr.Dataset
        A flat dataset compatible with v0.7 plotting functions.
    """
    # Start with core data
    target_ds = xr.Dataset()

    # Copy coordinates from input_data
    input_data = optimization_result.input_data
    if isinstance(input_data, xr.DataArray):
        for coord_name, coord_data in input_data.coords.items():
            target_ds.coords[coord_name] = coord_data
        target_ds["data"] = input_data
    else:
        # It's a Dataset
        for coord_name, coord_data in input_data.coords.items():
            target_ds.coords[coord_name] = coord_data
        # Copy data variable(s) - typically there's a main data variable
        if "data" in input_data:
            target_ds["data"] = input_data["data"]
        else:
            # Use the first data variable
            for var_name in input_data.data_vars:
                target_ds["data"] = input_data[var_name]
                break

    # Add residuals
    if optimization_result.residuals is not None:
        if isinstance(optimization_result.residuals, xr.DataArray):
            target_ds["residual"] = optimization_result.residuals
        else:
            # It's a Dataset - extract the residual variable
            if "residual" in optimization_result.residuals:
                target_ds["residual"] = optimization_result.residuals["residual"]
            else:
                for var_name in optimization_result.residuals.data_vars:
                    target_ds["residual"] = optimization_result.residuals[var_name]
                    break

    # Add fitted data (computed property)
    fitted = optimization_result.fitted_data
    if fitted is not None:
        if isinstance(fitted, xr.DataArray):
            target_ds["fitted_data"] = fitted
        else:
            if "fitted_data" in fitted:
                target_ds["fitted_data"] = fitted["fitted_data"]
            else:
                for var_name in fitted.data_vars:
                    target_ds["fitted_data"] = fitted[var_name]
                    break

    # Process elements based on their type
    for element_name, element_ds in optimization_result.elements.items():
        element_uid = element_ds.attrs.get("element_uid", "")

        if element_uid == KINETIC_ELEMENT_UID:
            add_kinetic_data(target_ds, optimization_result, element_name)
        elif element_uid == DAMPED_OSCILLATION_ELEMENT_UID:
            add_damped_oscillation_data(target_ds, optimization_result, element_name)
        elif element_uid == COHERENT_ARTIFACT_ELEMENT_UID:
            add_coherent_artifact_data(target_ds, optimization_result, element_name)
        # Add more element types here as needed

    # Process first activation for IRF data
    if optimization_result.activations:
        first_activation_name = next(iter(optimization_result.activations.keys()))
        add_activation_data(target_ds, optimization_result, first_activation_name)

    # Set dataset attributes from meta
    meta = optimization_result.meta
    target_ds.attrs["root_mean_square_error"] = meta.root_mean_square_error
    if meta.weighted_root_mean_square_error is not None:
        target_ds.attrs["weighted_root_mean_square_error"] = meta.weighted_root_mean_square_error
    else:
        target_ds.attrs["weighted_root_mean_square_error"] = meta.root_mean_square_error
    target_ds.attrs["global_dimension"] = meta.global_dimension
    target_ds.attrs["model_dimension"] = meta.model_dimension

    return target_ds


def convert(
    input: xr.Dataset | Result | OptimizationResult,
) -> xr.Dataset | CompatResult:
    """Convert a glotaran Result, OptimizationResult, or xarray Dataset to a compatible format.

    Parameters
    ----------
    input : xr.Dataset or Result or OptimizationResult
        The input object to be converted.

    Returns
    -------
    xr.Dataset or CompatResult
        The converted object.

    Raises
    ------
    ValueError
        If the input is not a Result, OptimizationResult, or a Dataset.

    Examples
    --------
    >>> result = Result(...)
    >>> converted_result = convert(result)
    >>> opt_result = result.optimization_results["dataset1"]
    >>> converted_dataset = convert(opt_result)
    >>> dataset = xr.open_dataset('input_dataset.nc')
    >>> converted_dataset = convert(dataset)
    """
    if isinstance(input, Result):
        return convert_result(input)
    if isinstance(input, OptimizationResult):
        return build_compat_dataset(input)
    if isinstance(input, xr.Dataset):
        return convert_dataset(input)
    msg = "input must be either a Result, OptimizationResult, or a Dataset"
    raise ValueError(msg)


def convert_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Convert a single dataset (legacy function for backward compatibility).

    This function is kept for cases where a single dataset is passed directly.
    For new code, use convert_result() with a full Result object.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to convert.

    Returns
    -------
    xr.Dataset
        The converted dataset.
    """
    # For a standalone dataset, we can only do minimal conversion
    # This is a fallback for legacy usage
    converted_ds = dataset.copy()

    # Rename fit to fitted_data if present
    if "fit" in converted_ds.data_vars:
        converted_ds["fitted_data"] = converted_ds["fit"]

    # Ensure weighted_root_mean_square_error is set
    if (
        "weighted_root_mean_square_error" not in converted_ds.attrs
        and "root_mean_square_error" in converted_ds.attrs
    ):
        converted_ds.attrs["weighted_root_mean_square_error"] = converted_ds.attrs[
            "root_mean_square_error"
        ]

    return converted_ds


def convert_result(result: Result) -> CompatResult:
    """Convert the result format used in v0.8 to a v0.7-compatible format.

    Parameters
    ----------
    result : Result
        The v0.8 Result object to convert.

    Returns
    -------
    CompatResult
        A CompatResult with flat datasets accessible via the `data` property.
    """
    converted_result = CompatResult.from_result(result)

    # Build flat datasets for each optimization result
    compat_datasets = {}
    for key, opt_result in result.optimization_results.items():
        compat_datasets[key] = build_compat_dataset(opt_result)

    # Store the converted datasets
    converted_result._compat_datasets = compat_datasets

    return converted_result
