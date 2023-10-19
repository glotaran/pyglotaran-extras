"""Convert a new pyglotaran (result) dataset to a version compatible with pyglotaran-extras."""
import copy
import xarray as xr
from glotaran.project.result import Result


def _adjust_estimations_to_spectra(ds: xr.Dataset, *, cleanup: bool = False) -> None:
    """Adjust the estimations to spectra names and flatten data."""
    # Check if the variable exists in the dataset
    if "kinetic_associated_estimation" in ds:
        # Extract the data
        data = ds["kinetic_associated_estimation"].to_numpy()
        # Reshape the data to match the desired shape
        for activation_id in range(data.shape[0]):
            reshaped_data = data[activation_id, :, :]
            # Update the dataset with the reshaped data
            ds[f"decay_associated_spectra_mc{activation_id+1}"] = (
                ("spectral", f"component_mc{activation_id+1}"),
                reshaped_data,
            )
        if cleanup:
            # Remove the original variable after adjustment
            ds = ds.drop_vars("kinetic_associated_estimation")
    if "species_associated_estimation" in ds:
        ds["species_associated_spectra"] = ds["species_associated_estimation"]
        if cleanup:
            ds = ds.drop_vars("species_associated_estimation")


def _adjust_activation_to_irf(ds: xr.Dataset, *, cleanup: bool = False) -> None:
    if "gaussian_activation_center" in ds:
        values = ds.gaussian_activation_center.to_numpy().flatten()
        ds["irf_center"] = values[0]
        if cleanup:
            # ds = ds.drop_vars("gaussian_activation_center")  # noqa: ERA001
            pass  # only cleanup if we have converted all irf aspects
    if "gaussian_activation_width" in ds:
        values = ds.gaussian_activation_width.to_numpy().flatten()
        ds["irf_width"] = values[0]
        if cleanup:
            # ds = ds.drop_vars("gaussian_activation_width")  # noqa: ERA001
            pass
    if "gaussian_activation_scale" in ds:
        values = ds.gaussian_activation_scale.to_numpy().flatten()
        ds["irf_scale"] = values[0]
        if cleanup:
            # ds = ds.drop_vars("gaussian_activation_scale")  # noqa: ERA001
            pass
    if "gaussian_activation_function" in ds:
        values = ds.gaussian_activation_function.to_numpy().flatten()
        ds["irf"] = values
        if cleanup:
            # ds = ds.drop_vars("gaussian_activation_function")  # noqa: ERA001
            pass


def convert(input: xr.Dataset | Result, cleanup: bool = False) -> xr.Dataset | Result:
    """Convert a glotaran Result or xarray Dataset to a different format.

    Parameters
    ----------
    input : xr.Dataset or Result
        The input object to be converted.
    cleanup : bool, optional
        Whether or not to perform cleanup after the conversion. Default is False.

    Returns
    -------
    xr.Dataset or Result
        The converted object.

    Raises
    ------
    ValueError
        If the input is not a Result or a Dataset.

    Examples
    --------
    >>> result = Result(...)
    >>> converted_result = convert(result, cleanup=True)
    >>> dataset = xr.open_dataset('input_dataset.nc')
    >>> converted_dataset = convert(dataset)
    """
    if isinstance(input, Result):
        return convert_result(input, cleanup=cleanup)
    if isinstance(input, xr.Dataset):
        return convert_dataset(input, cleanup=cleanup)
    msg = "input must be either a Result or a Dataset"
    raise ValueError(msg)


def convert_dataset(dataset: xr.Dataset, cleanup: bool = False) -> xr.Dataset:
    """Convert the dataset format used in staging (to be v0.8) to the format of main (v0.7)."""

    # Create a copy of the staging dataset to avoid modifying the original
    converted_ds = dataset.copy()

    _adjust_activation_to_irf(converted_ds, cleanup=cleanup)
    _adjust_estimations_to_spectra(converted_ds, cleanup=cleanup)

    # variable_mapping = {"species_associated_estimation": "species_associated_spectra"} # noqa: ERA001, E501
    # converted_ds = converted_ds.rename_vars({**variable_mapping}) # noqa: ERA001

    # more conversions
    return converted_ds


def convert_result(result: Result, cleanup: bool = False) -> Result:
    """Convert the result format used in staging (to be v0.8) to the format of main (v0.7)."""

    converted_result = copy.copy(result)

    # convert the datasets
    for key in converted_result.data:
        converted_result.data[key] = convert_dataset(converted_result.data[key], cleanup=cleanup)

    # convert the parameters
    return converted_result
