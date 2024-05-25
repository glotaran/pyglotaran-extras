"""Tests for ``pyglotaran_extras.config.plot_config``."""

from __future__ import annotations

from io import StringIO
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
import pytest
from jsonschema import ValidationError as SchemaValidationError
from jsonschema import validate
from pydantic import ValidationError as PydanticValidationError
from ruamel.yaml import YAML

from pyglotaran_extras.config.plot_config import PerFunctionPlotConfig
from pyglotaran_extras.config.plot_config import PlotConfig
from pyglotaran_extras.config.plot_config import PlotLabelOverRideMap
from pyglotaran_extras.config.plot_config import PlotLabelOverRideValue
from tests import TEST_DATA


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
        PlotLabelOverRideMap.model_validate({"invalid": {"invalid": 1}})

    assert (
        "target_name\n  Field required [type=missing, input_value={'invalid': 1}, input_type=dict]"
        in str(execinfo.value)
    )


def test_per_function_plot_config():
    """Initialize with correct defaults and validate correctly."""
    function_config_data: dict[str, Any] = YAML().load(
        StringIO(
            dedent(
                """
                default_args_override:
                    test_arg: true
                axis_label_override:
                    "Old Label": "New Label"
                """
            )
        )
    )
    function_config = PerFunctionPlotConfig.model_validate(function_config_data)

    validate(instance=function_config_data, schema=PerFunctionPlotConfig.model_json_schema())

    assert function_config.default_args_override["test_arg"] is True
    assert function_config.axis_label_override["Old Label"] == PlotLabelOverRideValue(
        target_name="New Label"
    )

    with pytest.raises(SchemaValidationError) as execinfo:
        validate(
            instance={"unknown": 1},
            schema=PerFunctionPlotConfig.model_json_schema(),
        )

    assert str(execinfo.value).startswith(
        "Additional properties are not allowed ('unknown' was unexpected)"
    )

    assert PerFunctionPlotConfig().model_dump() == {
        "default_args_override": {},
        "axis_label_override": {},
    }
    with pytest.raises(PydanticValidationError) as execinfo:
        PerFunctionPlotConfig.model_validate({"unknown": 1})

    assert (
        "1 validation error for PerFunctionPlotConfig\n"
        "unknown\n"
        "  Extra inputs are not permitted [type=extra_forbidden, input_value=1, input_type=int]"
        in str(execinfo.value)
    )


def test_per_function_plot_config_merge():
    """Values with same key get updated and other values stay the same."""
    original_config = PerFunctionPlotConfig(
        default_args_override={"test_arg": "to_be_changed", "not_updated": "same"},
        axis_label_override={"Old Label": "to_be_changed", "not_updated": "same"},
    )
    update_config = PerFunctionPlotConfig(
        default_args_override={"test_arg": "changed"},
        axis_label_override={"Old Label": "changed"},
    )

    merged_config = original_config.merge(update_config)

    assert merged_config.default_args_override["test_arg"] == "changed"
    assert merged_config.default_args_override["not_updated"] == "same"
    assert merged_config.axis_label_override["Old Label"] == PlotLabelOverRideValue(
        target_name="changed"
    )
    assert merged_config.axis_label_override["not_updated"] == PlotLabelOverRideValue(
        target_name="same"
    )


def test_per_function_plot_find_override_kwargs():
    """Only get kwargs that were not provided by the user and are known."""
    original_config = PerFunctionPlotConfig(
        default_args_override={"test_arg": "to_be_changed", "not_updated": "same"},
    )

    no_override_kwargs = original_config.find_override_kwargs(set())
    assert no_override_kwargs == {}

    override_kwargs = original_config.find_override_kwargs({"not_updated", "unknown_arg"})
    assert override_kwargs == {"not_updated": "same"}


def test_per_function_plot_update_axes_labels():
    """Only labels where the axis and current label match get updated."""

    def create_test_ax():
        ax = plt.subplot()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax

    simple_config = PerFunctionPlotConfig(axis_label_override={"x": "new x", "y": "new y"})

    simple_config.update_axes_labels(None)

    ax_both = create_test_ax()
    simple_config.update_axes_labels(ax_both)

    assert ax_both.get_xlabel() == "new x"
    assert ax_both.get_ylabel() == "new y"

    ax_explicit = create_test_ax()

    PerFunctionPlotConfig(
        axis_label_override=PlotLabelOverRideMap(
            {
                "x": PlotLabelOverRideValue(target_name="new x", axis="x"),
                "y": PlotLabelOverRideValue(target_name="new y", axis="y"),
            }
        )
    ).update_axes_labels(ax_explicit)
    assert ax_explicit.get_xlabel() == "new x"
    assert ax_explicit.get_ylabel() == "new y"

    ax_mismatch = create_test_ax()

    PerFunctionPlotConfig(
        axis_label_override=PlotLabelOverRideMap(
            {
                "x": PlotLabelOverRideValue(target_name="new x", axis="y"),
                "y": PlotLabelOverRideValue(target_name="new y", axis="x"),
            }
        )
    ).update_axes_labels(ax_mismatch)
    assert ax_mismatch.get_xlabel() == "x"
    assert ax_mismatch.get_ylabel() == "y"

    _, np_axes = plt.subplots(1, 2)

    assert np_axes.shape == (2,)

    np_axes[0].set_xlabel("x")
    np_axes[0].set_ylabel("y-keep")
    np_axes[1].set_xlabel("x-keep")
    np_axes[1].set_ylabel("y")

    simple_config.update_axes_labels(np_axes)

    assert np_axes[0].get_xlabel() == "new x"
    assert np_axes[0].get_ylabel() == "y-keep"
    assert np_axes[1].get_xlabel() == "x-keep"
    assert np_axes[1].get_ylabel() == "new y"

    _, ax0 = plt.subplots(1, 1)
    _, ax1 = plt.subplots(1, 1)

    iterable_axes = (ax0, ax1)

    iterable_axes[0].set_xlabel("x")
    iterable_axes[0].set_ylabel("y-keep")
    iterable_axes[1].set_xlabel("x-keep")
    iterable_axes[1].set_ylabel("y")

    simple_config.update_axes_labels(iterable_axes)

    assert iterable_axes[0].get_xlabel() == "new x"
    assert iterable_axes[0].get_ylabel() == "y-keep"
    assert iterable_axes[1].get_xlabel() == "x-keep"
    assert iterable_axes[1].get_ylabel() == "new y"


def test_plot_config():
    """Initialize with correct defaults and validate correctly."""
    plot_config_values: dict[str, Any] = YAML().load(
        StringIO(
            dedent(
                """
                general:
                    default_args_override:
                        test_arg: true
                    axis_label_override:
                        "Old Label": "New Label"

                test_func:
                    default_args_override:
                        test_arg: false
                    axis_label_override:
                        "Old Y Label":
                            target_name: "New Y Label"
                            axis: y
                """
            )
        )
    )
    plot_config = PlotConfig.model_validate(plot_config_values)

    assert plot_config.model_extra is not None

    assert plot_config.general == PerFunctionPlotConfig(
        default_args_override={"test_arg": True},
        axis_label_override={"Old Label": "New Label"},
    )
    assert plot_config.model_extra["test_func"] == PerFunctionPlotConfig(
        default_args_override={"test_arg": False},
        axis_label_override={"Old Y Label": {"target_name": "New Y Label", "axis": "y"}},
    )

    with pytest.raises(PydanticValidationError) as execinfo:
        PerFunctionPlotConfig.model_validate({"test_func": {"unknown": 1}})

    assert (
        "1 validation error for PerFunctionPlotConfig\n"
        "test_func\n"
        "  Extra inputs are not permitted "
        "[type=extra_forbidden, input_value={'unknown': 1}, input_type=dict]"
        in str(execinfo.value)
    )


def test_plot_config_merge():
    """Merging creates the expected output.

    - Update fields that are present in original and update
    - Keep fields that are not present in the update
    - Add field that is only present in update
    """
    original_values, update_values, expected_values = tuple(
        YAML().load_all(StringIO((TEST_DATA / "config/plot_config_merge.yml").read_text()))
    )
    original = PlotConfig.model_validate(original_values)
    update = PlotConfig.model_validate(update_values)
    expected = PlotConfig.model_validate(expected_values)

    assert original.merge(update) == expected


def test_plot_config_get_function_config():
    """The generated config updates the general config with the test func config.

    - Update fields that are present in general and test_fun config
    - Keep fields that are not present in the test_func config
    - Add field that is only present in test_func config
    """
    plot_config_values = YAML().load(
        StringIO((TEST_DATA / "config/pyglotaran_extras_config.yml").read_text())
    )
    plot_config = PlotConfig.model_validate(plot_config_values["plotting"])

    assert plot_config.get_function_config("test_func") == PerFunctionPlotConfig(
        default_args_override={
            "will_update_arg": "test_func arg",
            "will_be_kept_arg": "general arg",
            "will_be_added_arg": "new arg",
        },
        axis_label_override={
            "will_update_label": "test_func label",
            "will_be_kept_label": "general label",
            "will_be_added_label": "new label",
        },
    )
