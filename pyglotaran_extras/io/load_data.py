from __future__ import annotations

from pathlib import Path

import xarray as xr
from glotaran.io import load_dataset
from glotaran.project.result import Result

from pyglotaran_extras.types import DatasetConvertible


def load_data(result: DatasetConvertible, dataset_name: str | None = None) -> xr.Dataset:
    """Extract a single dataset from a :class:`DatasetConvertible` object.

    Parameters
    ----------
    result : DatasetConvertible
        Result class instance, xarray Dataset or path to a dataset file.
    dataset_name : str, optional
        Name of a specific dataset contained in ``result``, if not provided
        the first dataset will be extracted., by default None

    Returns
    -------
    xr.Dataset
        Extracted dataset.

    Raises
    ------
    TypeError
        If ``result`` isn't a :class:`DatasetConvertible` object.
    """
    if isinstance(result, xr.Dataset):
        return result
    if isinstance(result, xr.DataArray):
        return result.to_dataset(name="data")
    if isinstance(result, Result):
        if dataset_name is not None:
            return result.data[dataset_name]
        keys = list(result.data)
        return result.data[keys[0]]
    if isinstance(result, (str, Path)):
        return load_data(load_dataset(result))
    raise TypeError(f"Result needs to be of type {DatasetConvertible!r}, but was {result!r}.")
