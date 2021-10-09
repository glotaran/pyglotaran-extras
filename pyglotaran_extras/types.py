from __future__ import annotations

from pathlib import Path
from typing import Mapping
from typing import Sequence
from typing import Union

from glotaran.project.result import Result
from xarray import Dataset

DatasetConvertible = Union[Dataset, str, Path]
"""Types of data which can be converted to a dataset."""
ResultLike = Union[
    Result, DatasetConvertible, Mapping[str, DatasetConvertible], Sequence[DatasetConvertible]
]
"""Result like data which can be converted to a per dataset mapping."""
