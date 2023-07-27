"""Module containing type definitions."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import Tuple
from typing import Union

import xarray as xr
from glotaran.project.result import Result

if TYPE_CHECKING:
    if sys.version_info < (3, 10):
        from typing_extensions import TypeAlias
    else:
        from typing import TypeAlias


DatasetConvertible = Union[xr.Dataset, xr.DataArray, str, Path]
"""Types of data which can be converted to a dataset."""
ResultLike = Union[
    Result, DatasetConvertible, Mapping[str, DatasetConvertible], Sequence[DatasetConvertible]
]
"""Result like data which can be converted to a per dataset mapping."""


BuiltinSubPlotLabelFormatFunctionKey: TypeAlias = Literal[
    "number", "upper_case_letter", "lower_case_letter"
]
"""Key supported by ``BuiltinLabelFormatFunctions``."""

SubPlotLabelCoordStrs: TypeAlias = Literal[
    "figure points",
    "figure pixels",
    "figure fraction",
    "subfigure points",
    "subfigure pixels",
    "subfigure fraction",
    "axes points",
    "axes pixels",
    "axes fraction",
    "data",
    "polar",
]

SubPlotLabelCoord: TypeAlias = Union[
    SubPlotLabelCoordStrs, Tuple[SubPlotLabelCoordStrs, SubPlotLabelCoordStrs]
]
