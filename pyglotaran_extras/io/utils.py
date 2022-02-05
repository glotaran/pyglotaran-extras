"""Io utility module."""
from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

import xarray as xr
from glotaran.project.result import Result

from pyglotaran_extras.io.load_data import load_data
from pyglotaran_extras.types import ResultLike


def result_dataset_mapping(result: ResultLike) -> Mapping[str, xr.Dataset]:
    """Convert a ``ResultLike`` object to a per dataset mapping of result like data.

    Parameters
    ----------
    result : ResultLike
        Data structure which can be converted to a mapping.

    Returns
    -------
    Mapping[str, Dataset]
        Per dataset mapping of result like data.

    Raises
    ------
    TypeError
        If any value of a ``result`` isn't of :class:`DatasetConvertible`.
    TypeError
        If ``result`` isn't a :class:`ResultLike` object.
    """
    result_mapping = {}
    if isinstance(result, Result):
        return result.data
    if isinstance(result, (xr.Dataset, xr.DataArray, Path, str)):
        return {"dataset": load_data(result)}
    if isinstance(result, Sequence):
        for index, value in enumerate(result):
            result_mapping[f"dataset{index}"] = load_data(value)
        return result_mapping
    if isinstance(result, Mapping):
        for key, value in result.items():
            result_mapping[key] = load_data(value)
        return result_mapping
    raise TypeError(f"Result needs to be of type {ResultLike!r}, but was {result!r}.")
