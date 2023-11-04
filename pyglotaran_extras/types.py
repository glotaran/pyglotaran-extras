"""Module containing type definitions."""
from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from typing import TypeAlias

import xarray as xr
from glotaran.project.result import Result


class UnsetType:
    """Type for the ``Unset`` singleton."""

    def __repr__(self) -> str:  # noqa: DOC
        """Representation of instances in editors."""
        return "Unset"


Unset = UnsetType()
"""Singleton to use when arguments are not provided where None is a valid value."""

DatasetConvertible: TypeAlias = xr.Dataset | xr.DataArray | str | Path
"""Types of data which can be converted to a dataset."""
ResultLike: TypeAlias = (
    Result | DatasetConvertible | Mapping[str, DatasetConvertible] | Sequence[DatasetConvertible]
)
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
