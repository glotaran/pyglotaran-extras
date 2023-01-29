"""Data loading utility module."""
from __future__ import annotations

from pathlib import Path
from warnings import warn

import xarray as xr
from glotaran.io import load_dataset
from glotaran.project.result import Result

from pyglotaran_extras.types import DatasetConvertible


def load_data(
    result: DatasetConvertible | Result, dataset_name: str | None = None, *, _stacklevel: int = 2
) -> xr.Dataset:
    """Extract a single dataset from a :class:`DatasetConvertible` object.

    Parameters
    ----------
    result : DatasetConvertible | Result
        Result class instance, xarray Dataset or path to a dataset file.
    dataset_name : str, optional
        Name of a specific dataset contained in ``result``, if not provided
        the first dataset will be extracted. Defaults to None.
    _stacklevel: int
        Stacklevel of the warning which is raised when ``result`` is of class ``Result``,
        contains multiple datasets and no ``dataset_name`` is provided. Changing this value is
        only required if you use this function inside of another function. Defaults to 2

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
        if len(keys) > 1:
            warn(
                UserWarning(
                    f"Result contains multiple datasets, auto selecting {keys[0]!r}.\n"
                    f"Pass the dataset set you want to plot (e.g. result.data[{keys[0]!r}]) , "
                    f"to deactivate this Warning.\nPossible dataset names are: {keys}"
                ),
                stacklevel=_stacklevel,
            )
        return result.data[keys[0]]
    if isinstance(result, (str, Path)):
        return load_data(load_dataset(result))
    raise TypeError(f"Result needs to be of type {DatasetConvertible!r}, but was {result!r}.")
