"""Module containing plot configuration."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any
from typing import Literal

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
            if isinstance(value, PlotLabelOverRideValue):
                parsed_values[key] = value
            elif isinstance(value, str):
                parsed_values[key] = PlotLabelOverRideValue(target_name=value)
            else:
                parsed_values[key] = PlotLabelOverRideValue(**value)
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
