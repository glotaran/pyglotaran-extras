"""Convert a new pyglotaran (result) dataset to a version compatible with pyglotaran-extras."""
import xarray as xr


def adjust_estimations_to_spectra(ds: xr.Dataset, *, cleanup: bool = False) -> xr.Dataset:
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
    return ds


def convert(dataset: xr.Dataset, cleanup: bool = False) -> xr.Dataset:
    """Convert the dataset format used in staging (to be v0.8) to the format of main (v0.7)."""

    # Create a copy of the staging dataset to avoid modifying the original
    converted_ds = dataset.copy()

    # variable_mapping = {"species_associated_estimation": "species_associated_spectra"} # noqa: ERA001, E501
    # converted_ds = converted_ds.rename_vars({**variable_mapping}) # noqa: ERA001

    # more conversions
    return adjust_estimations_to_spectra(converted_ds, cleanup=cleanup)
