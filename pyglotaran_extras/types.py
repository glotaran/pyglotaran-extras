"""Module containing type definitions."""
from __future__ import annotations

from pathlib import Path
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import TypeAlias
from typing import Union

import xarray as xr
from glotaran.project.result import Result

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

SubPlotLabelCoord: TypeAlias = (
    SubPlotLabelCoordStrs | tuple[SubPlotLabelCoordStrs, SubPlotLabelCoordStrs]
)
