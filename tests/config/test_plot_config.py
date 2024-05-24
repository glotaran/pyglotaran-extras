"""Tests for ``pyglotaran_extras.config.plot_config``."""

from __future__ import annotations

from io import StringIO
from textwrap import dedent
from typing import Any

import pytest
from jsonschema import ValidationError as SchemaValidationError
from jsonschema import validate
from pydantic import ValidationError as PydanticValidationError
from ruamel.yaml import YAML

from pyglotaran_extras.config.plot_config import PlotLabelOverRideMap
from pyglotaran_extras.config.plot_config import PlotLabelOverRideValue


def test_plot_label_over_ride_value_serialization():
    """Short notation is used if axis has default value."""
    assert PlotLabelOverRideValue(target_name="New Label").model_dump() == "New Label"
    assert PlotLabelOverRideValue(target_name="New Label", axis="x").model_dump() == {
        "target_name": "New Label",
        "axis": "x",
    }


def test_plot_label_over_ride_map():
    """PlotLabelOverRideMap behaves like a mapping and the schema allows short notation."""
    axis_label_override: dict[str, Any] = YAML().load(
        StringIO(
            dedent(
                """
                Old Label: New Label
                Old Y Label:
                    target_name: New Label
                    axis: y
                """
            )
        )
    )
    override_map = PlotLabelOverRideMap(axis_label_override)

    assert len(override_map) == 2

    assert override_map["Old Label"] == PlotLabelOverRideValue(target_name="New Label")
    assert override_map["Old Y Label"] == PlotLabelOverRideValue(target_name="New Label", axis="y")

    override_map_pydantic_init = PlotLabelOverRideMap(
        {"Old Label": PlotLabelOverRideValue(target_name="New Label")}
    )
    assert override_map_pydantic_init["Old Label"] == PlotLabelOverRideValue(
        target_name="New Label"
    )

    validate(instance=axis_label_override, schema=PlotLabelOverRideMap.model_json_schema())

    for map_item_tuple, expected in zip(
        override_map.items(), axis_label_override.items(), strict=True
    ):
        assert (map_item_tuple[0], map_item_tuple[1].model_dump()) == expected

    with pytest.raises(SchemaValidationError) as execinfo:
        validate(
            instance={"Old Y Label": {"axis": "y"}},
            schema=PlotLabelOverRideMap.model_json_schema(),
        )

    assert str(execinfo.value).startswith("'target_name' is a required property")

    assert PlotLabelOverRideMap().model_dump() == {}
    with pytest.raises(PydanticValidationError) as execinfo:
        PlotLabelOverRideMap({"invalid": {"invalid": 1}})

    assert (
        "target_name\n  Field required [type=missing, input_value={'invalid': 1}, input_type=dict]"
        in str(execinfo.value)
    )
