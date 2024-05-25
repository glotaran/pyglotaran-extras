"""Module containing plot configuration."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import RootModel
from pydantic import model_serializer
from pydantic import model_validator
from pydantic_core import PydanticUndefined


class PlotLabelOverRideValue(BaseModel):
    """Value of ``PlotLabelOverRideMap``."""

    target_name: str
    axis: Literal["x", "y", "both"] = "both"

    @model_serializer
    def serialize(self) -> dict[str, Any] | str:
        """Serialize supporting short notation.

        Returns
        -------
        dict[str, Any] | str
        """
        if self.axis == "both":
            return self.target_name
        return {"target_name": self.target_name, "axis": self.axis}


def _add_short_notation_to_schema(json_schema: dict[str, Any]) -> None:  # noqa: DOC
    """Update json schema to support short notation for ``PlotLabelOverRideValue``."""
    orig_additional_properties = json_schema["additionalProperties"]
    json_schema["additionalProperties"] = {
        "anyOf": [orig_additional_properties, {"type": "string"}]
    }


class PlotLabelOverRideMap(RootModel, Mapping):
    """Mapping to override axis labels."""

    model_config = ConfigDict(json_schema_extra=_add_short_notation_to_schema)

    root: dict[str, PlotLabelOverRideValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def parse(cls, values: dict[str, Any]) -> dict[str, PlotLabelOverRideValue]:  # noqa: DOC
        """Parse ``axis_label_override`` dictionary supporting verbose and short notation.

        Parameters
        ----------
        values : dict[str, Any]
            Dict that initializes the class.

        Returns
        -------
        dict[str, PlotLabelOverRideValue]
        """
        if values is PydanticUndefined or values is None:
            return {}
        parsed_values: dict[str, PlotLabelOverRideValue] = {}
        for key, value in values.items():
            if isinstance(value, str):
                parsed_values[key] = PlotLabelOverRideValue(target_name=value)
            else:
                parsed_values[key] = PlotLabelOverRideValue.model_validate(value)
        return parsed_values

    def __iter__(self) -> Iterator[str]:  # noqa: DOC
        """Iterate over items."""
        return iter(self.root)

    def __len__(self) -> int:  # noqa: DOC
        """Get number of items."""
        return len(self.root)

    def __getitem__(self, item_label: str) -> PlotLabelOverRideValue:  # noqa: DOC
        """Access items."""
        return self.root[item_label]


class PerFunctionPlotConfig(BaseModel):
    """Per function plot configuration."""

    model_config = ConfigDict(extra="forbid")

    default_args_override: dict[str, Any] = Field(
        default_factory=dict,
        description="Default arguments to use if not specified in function call.",
    )
    axis_label_override: PlotLabelOverRideMap = Field(default_factory=PlotLabelOverRideMap)

    def merge(self, other: PerFunctionPlotConfig) -> PerFunctionPlotConfig:
        """Merge two ``PerFunctionPlotConfig``'s where ``other`` overrides values.

        Parameters
        ----------
        other : PerFunctionPlotConfig
            Other ``PerFunctionPlotConfig`` to merge in.

        Returns
        -------
        PerFunctionPlotConfig
        """
        self_dict = self.model_dump()
        other_dict = other.model_dump()
        return PerFunctionPlotConfig(
            default_args_override=(
                self_dict["default_args_override"] | other_dict["default_args_override"]
            ),
            axis_label_override=(
                self_dict["axis_label_override"] | other_dict["axis_label_override"]
            ),
        )

    def find_override_kwargs(self, not_user_provided_kwargs: set[str]) -> dict[str, Any]:
        """Config key word arguments that were not provided by the user and are safe to override.

        Parameters
        ----------
        not_user_provided_kwargs : set[str]
            Set of keyword arguments that were provided by the user and thus should not be
            overridden.

        Returns
        -------
        dict[str, Any]
        """
        return {
            k: self.default_args_override[k]
            for k in self.default_args_override
            if k in not_user_provided_kwargs
        }

    def update_axes_labels(
        self, axes: Axes | Iterable[Axes] | np.ndarray[Axes, np.dtype[Any]] | None
    ) -> None:
        """Apply label overrides to ``axes``.

        Parameters
        ----------
        axes : Axes | Iterable[Axes] | np.ndarray[Axes, np.dtype[Any]] | None
            Axes to apply the override to.
        """
        if axes is None:
            return
        if isinstance(axes, Axes):
            orig_x_label = axes.get_xlabel()
            orig_y_label = axes.get_ylabel()

            if orig_x_label in self.axis_label_override and (
                override_item := self.axis_label_override[orig_x_label]
            ).axis in ("x", "both"):
                axes.set_xlabel(override_item.target_name)

            if orig_y_label in self.axis_label_override and (
                override_item := self.axis_label_override[orig_y_label]
            ).axis in ("y", "both"):
                axes.set_ylabel(override_item.target_name)

        elif isinstance(axes, np.ndarray):
            for ax in axes.flatten():
                self.update_axes_labels(ax)
        else:
            for ax in axes:
                self.update_axes_labels(ax)
