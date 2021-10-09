from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

from glotaran.io import load_dataset
from glotaran.project.result import Result
from xarray import DataArray
from xarray import Dataset

from pyglotaran_extras.types import DatasetConvertible
from pyglotaran_extras.types import ResultLike


def result_dataset_mapping(result: ResultLike) -> Mapping[str, Dataset]:
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
        If any element of a ``result`` sequence isn't :class:`DatasetConvertible`.
    TypeError
        If any element of a ``result`` mapping isn't :class:`DatasetConvertible`.
    TypeError
        If ``result`` isn't a :class:`ResultLike` object.
    """

    if isinstance(result, Result):
        return result.data
    if isinstance(result, Dataset):
        return {"dataset": result}
    if isinstance(result, Sequence):
        result_mapping = {}
        for index, element in enumerate(result):
            if isinstance(element, (Path, str)):
                element = load_dataset(element)
            result_mapping[f"dataset{index}"] = element
            if not isinstance(element, (Dataset, DataArray)):
                raise TypeError(
                    f"Elements of result need to be of type {DatasetConvertible!r}."
                    f", but were {result!r}."
                )
        return result_mapping
    if isinstance(result, Mapping):
        result_mapping = {}
        for key, element in result.items():
            if isinstance(element, (Path, str)):
                element = load_dataset(element)
            result_mapping[key] = element
            if not isinstance(element, (Dataset, DataArray)):
                raise TypeError(
                    f"Elements of result need to be of type {DatasetConvertible!r}."
                    f", but were {result!r}."
                )
        return result_mapping
    raise TypeError(f"Result needs to be of type {ResultLike!r}, but was {result!r}.")
