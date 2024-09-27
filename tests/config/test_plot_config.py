"""Tests for ``pyglotaran_extras.config.plot_config``."""

from __future__ import annotations

from functools import wraps
from io import StringIO
from textwrap import dedent
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import pytest
import xarray as xr
from jsonschema import ValidationError as SchemaValidationError
from jsonschema import validate
from pydantic import ValidationError as PydanticValidationError
from ruamel.yaml import YAML

from pyglotaran_extras.config.plot_config import PerFunctionPlotConfig
from pyglotaran_extras.config.plot_config import PlotConfig
from pyglotaran_extras.config.plot_config import PlotLabelOverrideMap
from pyglotaran_extras.config.plot_config import PlotLabelOverrideValue
from pyglotaran_extras.config.plot_config import extract_default_kwargs
from pyglotaran_extras.config.plot_config import find_axes
from pyglotaran_extras.config.plot_config import find_not_user_provided_kwargs
from pyglotaran_extras.config.plot_config import plot_config_context
from pyglotaran_extras.config.plot_config import use_plot_config
from tests import TEST_DATA
from tests.conftest import generator_is_exhausted

if TYPE_CHECKING:
    from typing import Literal

    from matplotlib.axes import Axes

    from pyglotaran_extras.config.config import Config


def test_plot_label_override_value_serialization():
    """Short notation is used if axis has default value."""
    assert PlotLabelOverrideValue(target_name="New Label").model_dump() == "New Label"
    assert PlotLabelOverrideValue(target_name="New Label", axis="x").model_dump() == {
        "target_name": "New Label",
        "axis": "x",
    }


def test_plot_label_override_map():
    """PlotLabelOverrideMap behaves like a mapping and the schema allows short notation."""
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
    override_map = PlotLabelOverrideMap(axis_label_override)

    assert len(override_map) == 2

    assert override_map["Old Label"] == PlotLabelOverrideValue(target_name="New Label")
    assert override_map["Old Y Label"] == PlotLabelOverrideValue(target_name="New Label", axis="y")

    override_map_pydantic_init = PlotLabelOverrideMap(
        {"Old Label": PlotLabelOverrideValue(target_name="New Label")}
    )
    assert override_map_pydantic_init["Old Label"] == PlotLabelOverrideValue(
        target_name="New Label"
    )

    validate(instance=axis_label_override, schema=PlotLabelOverrideMap.model_json_schema())

    for map_item_tuple, expected in zip(
        override_map.items(), axis_label_override.items(), strict=True
    ):
        assert (map_item_tuple[0], map_item_tuple[1].model_dump()) == expected

    with pytest.raises(SchemaValidationError) as execinfo:
        validate(
            instance={"Old Y Label": {"axis": "y"}},
            schema=PlotLabelOverrideMap.model_json_schema(),
        )

    assert str(execinfo.value).startswith("'target_name' is a required property")

    assert PlotLabelOverrideMap().model_dump() == {}
    with pytest.raises(PydanticValidationError) as execinfo:
        PlotLabelOverrideMap.model_validate({"invalid": {"invalid": 1}})

    assert (
        "target_name\n  Field required [type=missing, input_value={'invalid': 1}, input_type=dict]"
        in str(execinfo.value)
    )


@pytest.mark.parametrize(
    ("matplotlib_label", "axis_name", "expected"),
    [
        ("not_found", "x", None),
        ("not_found", "y", None),
        ("no_user_newline", "x", "no_user_newline value"),
        ("no\n_user_newline", "x", "no_user_newline value"),
        ("with_\nuser_\nnewline", "x", "with_user_newline value"),
        ("with_user_newline", "x", "with_user_newline value"),
        ("x_only", "x", "x_only value"),
        ("x_only", "y", None),
        ("y_only", "x", None),
        ("y_only", "y", "y_only value"),
    ],
)
def test_plot_label_override_map_find_axis_label(
    matplotlib_label: str, axis_name: Literal["x", "y"], expected: str | None
):
    """Finding the correct label is agnostic to newlines injected by matplotlib."""
    override_map = PlotLabelOverrideMap.model_validate(
        {
            "no_user_newline": "no_user_newline value",
            "with_\nuser_\nnewline": "with_user_newline value",
            "x_only": {"target_name": "x_only value", "axis": "x"},
            "y_only": {"target_name": "y_only value", "axis": "y"},
        }
    )
    assert override_map.find_axis_label(matplotlib_label, axis_name) == expected


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
    assert function_config.axis_label_override["Old Label"] == PlotLabelOverrideValue(
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

    assert PerFunctionPlotConfig().model_dump() == {}

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
    assert merged_config.axis_label_override["Old Label"] == PlotLabelOverrideValue(
        target_name="changed"
    )
    assert merged_config.axis_label_override["not_updated"] == PlotLabelOverrideValue(
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

    def create_test_ax() -> Axes:
        _, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax

    simple_config = PerFunctionPlotConfig(axis_label_override={"x": "new x", "y": "new y"})

    ax_both = create_test_ax()
    simple_config.update_axes_labels(ax_both)

    assert ax_both.get_xlabel() == "new x"
    assert ax_both.get_ylabel() == "new y"

    ax_explicit = create_test_ax()

    PerFunctionPlotConfig(
        axis_label_override=PlotLabelOverrideMap(
            {
                "x": PlotLabelOverrideValue(target_name="new x", axis="x"),
                "y": PlotLabelOverrideValue(target_name="new y", axis="y"),
            }
        )
    ).update_axes_labels(ax_explicit)
    assert ax_explicit.get_xlabel() == "new x"
    assert ax_explicit.get_ylabel() == "new y"

    ax_mismatch = create_test_ax()

    PerFunctionPlotConfig(
        axis_label_override=PlotLabelOverrideMap(
            {
                "x": PlotLabelOverrideValue(target_name="new x", axis="y"),
                "y": PlotLabelOverrideValue(target_name="new y", axis="x"),
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


def test_plot_config_get_function_config(test_config_values: dict[str, Any]):
    """The generated config updates the general config with the test func config.

    - Update fields that are present in general and test_fun config
    - Keep fields that are not present in the test_func config
    - Add field that is only present in test_func config
    """
    plot_config = PlotConfig.model_validate(test_config_values["plotting"])

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


def test_extract_default_kwargs():
    """Extract argument that can be passed as kwargs."""

    def func(
        pos_arg: str,
        pos_arg_default: str = "pos_arg_default",
        /,
        normal_arg: str = "normal_arg",
        *,
        kw_only_arg: int = 1,
    ):
        r"""Test function.

        Parameters
        ----------
        pos_arg : str
            Not extracted
        pos_arg_default : str
            Not extracted. Defaults to "pos_arg_default"
        normal_arg : str
            A normal arg. Defaults to "normal_arg".
        kw_only_arg : int
            A keyword only arg with new line and escaped character in the docstring (\\nu).
            Defaults to 1.
        """

    assert extract_default_kwargs(func, ()) == {
        "normal_arg": {
            "default": "normal_arg",
            "annotation": "str",
            "docstring": 'A normal arg. Defaults to "normal_arg".',
        },
        "kw_only_arg": {
            "default": 1,
            "annotation": "int",
            "docstring": (
                r"A keyword only arg with new line and escaped character in the docstring (\\nu)."
                " Defaults to 1."
            ),
        },
    }

    assert extract_default_kwargs(func, ("kw_only_arg",)) == {
        "normal_arg": {
            "default": "normal_arg",
            "annotation": "str",
            "docstring": 'A normal arg. Defaults to "normal_arg".',
        },
    }

    def no_annotation(foo="bar"):
        pass

    assert extract_default_kwargs(no_annotation, ()) == {
        "foo": {"default": "bar", "annotation": "object", "docstring": None}
    }


def test_find_not_user_provided_kwargs():
    """Only find kwarg names for none user passed kwargs."""
    result = None

    def dec(func):
        default_kwargs = extract_default_kwargs(func, ())

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal result
            arg_names = func.__code__.co_varnames[: len(args)]
            result = find_not_user_provided_kwargs(default_kwargs, arg_names, kwargs)
            return func(*args, *kwargs)

        return wrapper

    @dec
    def func(
        pos_arg: str,
        arg1: str = "arg1",
        arg2: str = "arg2",
        *,
        kwarg1: int = 1,
        kwarg2: int = 1,
    ):
        return 1

    assert func("foo", "bar", kwarg2=2) == 1
    assert result == {"arg2", "kwarg1"}


def test_find_axes():
    """Get axes value from iterable of values."""

    data_array = xr.DataArray([[0, 1]], coords={"time": [0], "spectral": [500, 510]})
    data_set = xr.Dataset({"data": data_array})

    base_values = ["foo", True, 1.5, data_array, data_set]

    assert generator_is_exhausted(find_axes(base_values)) is True

    _, ax = plt.subplots()
    single_ax_gen = find_axes([*base_values, ax])

    assert next(single_ax_gen) is ax
    assert generator_is_exhausted(single_ax_gen) is True

    _, np_axes = plt.subplots(1, 2)

    assert np_axes.shape == (2,)

    np_axes_gen = find_axes([*base_values, np_axes])

    assert next(np_axes_gen) is np_axes[0]
    assert next(np_axes_gen) is np_axes[1]
    assert generator_is_exhausted(np_axes_gen) is True

    _, ax1 = plt.subplots()
    iterable_axes = (ax, ax1)
    iterable_axes_gen = find_axes([*base_values, iterable_axes])

    assert next(iterable_axes_gen) is ax
    assert next(iterable_axes_gen) is ax1
    assert generator_is_exhausted(iterable_axes_gen) is True

    multiple_axes_gen = find_axes([*base_values, ax, ax1])

    assert next(multiple_axes_gen) is ax
    assert next(multiple_axes_gen) is ax1
    assert generator_is_exhausted(multiple_axes_gen) is True


def test_use_plot_config(mock_config: tuple[Config, dict[str, Any]]):
    """Config is applied to functions with the ``use_plot_config`` decorator."""
    _, registry = mock_config

    assert registry == {}

    @use_plot_config()
    def test_func(
        will_update_arg="default update",
        will_be_kept_arg="default keep",
        will_be_added_arg="default add",
        not_in_config="not_in_config",
    ):
        kwargs = {
            "will_update_arg": will_update_arg,
            "will_be_kept_arg": will_be_kept_arg,
            "will_be_added_arg": will_be_added_arg,
            "not_in_config": not_in_config,
        }
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_xlabel("will_update_label")
        ax1.set_ylabel("will_be_kept_label")
        ax2.set_xlabel("will_be_added_label")
        ax2.set_ylabel("default")
        return fig, (ax1, ax2), kwargs

    assert "test_func" in registry

    _, (ax1_test_func, ax2_test_func), kwargs_test_func_no_user_args = test_func()

    assert ax1_test_func.get_xlabel() == "test_func label"
    assert ax1_test_func.get_ylabel() == "general label"
    assert ax2_test_func.get_xlabel() == "new label"
    assert ax2_test_func.get_ylabel() == "default"

    assert kwargs_test_func_no_user_args["will_update_arg"] == "test_func arg"
    assert kwargs_test_func_no_user_args["will_be_kept_arg"] == "general arg"
    assert kwargs_test_func_no_user_args["will_be_added_arg"] == "new arg"
    assert kwargs_test_func_no_user_args["not_in_config"] == "not_in_config"

    _, _, kwargs_test_func_user_args = test_func(
        will_update_arg="set by user", will_be_added_arg="added by user"
    )

    assert kwargs_test_func_user_args["will_update_arg"] == "set by user"
    assert kwargs_test_func_user_args["will_be_kept_arg"] == "general arg"
    assert kwargs_test_func_user_args["will_be_added_arg"] == "added by user"
    assert kwargs_test_func_no_user_args["not_in_config"] == "not_in_config"

    _, axes = plt.subplots(1, 2)

    @use_plot_config()
    def axes_iterable_arg(
        axes: tuple[Axes, Axes],
    ):
        (ax1, ax2) = axes
        ax1.set_xlabel("will_update_label")
        ax1.set_ylabel("will_be_kept_label")
        ax2.set_xlabel("will_be_added_label")
        ax2.set_ylabel("default")
        return (ax1, ax2)

    assert "axes_iterable_arg" in registry

    axes_iterable_arg((axes[0], axes[1]))

    assert axes[0].get_xlabel() == "will change label"
    assert axes[0].get_ylabel() == "general label"
    assert axes[1].get_xlabel() == "will_be_added_label"
    assert axes[1].get_ylabel() == "default"

    _, (ax1_arg, ax2_arg) = plt.subplots(1, 2)

    @use_plot_config()
    def multiple_axes_args(
        ax1: Axes,
        ax2: Axes,
    ):
        ax1.set_xlabel("will_update_label")
        ax1.set_ylabel("will_be_kept_label")
        ax2.set_xlabel("will_be_added_label")
        ax2.set_ylabel("default")
        return (ax1, ax2)

    assert "multiple_axes_args" in registry

    multiple_axes_args(ax1_arg, ax2_arg)

    assert ax1_arg.get_xlabel() == "will change label"
    assert ax1_arg.get_ylabel() == "general label"
    assert ax2_arg.get_xlabel() == "will_be_added_label"
    assert ax2_arg.get_ylabel() == "default"

    # Integration test that ``PlotLabelOverrideMap.find_axis_label`` is used
    _, ax_newline = plt.subplots()

    @use_plot_config()
    def newline_label(
        ax: Axes,
    ):
        ax.set_xlabel("will_\nupdate_label")
        ax.set_ylabel("will_be_\nkept_label")
        return ax

    assert "newline_label" in registry

    newline_label(ax_newline)

    assert ax_newline.get_xlabel() == "will change label"
    assert ax_newline.get_ylabel() == "general label"


@pytest.mark.usefixtures("mock_config")
def test_plot_config_context():
    """Context overrides resolved config values of the function."""
    import pyglotaran_extras

    source_file = pyglotaran_extras.CONFIG._source_files[0]
    original_test_func_config = pyglotaran_extras.CONFIG.plotting.get_function_config(
        "test_func"
    ).model_copy(deep=True)

    plot_config = PerFunctionPlotConfig(
        default_args_override={
            "will_be_added_arg": "test_func arg overridden by context arg",
            "added_by_context_arg": "added by context arg",
        },
        axis_label_override={
            "will_be_added_label": "test_func arg overridden by context label",
            "added_by_context_label": "added by context label",
        },
    )

    _, (ax1_arg, ax2_arg) = plt.subplots(1, 2)

    @use_plot_config()
    def test_func(
        ax1: Axes,
        ax2: Axes,
    ):
        ax1.set_xlabel("will_update_label")
        ax1.set_ylabel("will_be_kept_label")
        ax2.set_xlabel("will_be_added_label")
        ax2.set_ylabel("default")
        return (ax1, ax2)

    with plot_config_context(plot_config):
        test_func_config = pyglotaran_extras.CONFIG.plotting.get_function_config("test_func")

        # Force reload in use_plot_config by changing mtime of the file
        source_file.write_bytes(source_file.read_bytes())

        test_func(ax1_arg, ax2_arg)

    assert hasattr(pyglotaran_extras.CONFIG.plotting, "__context_config") is False

    assert test_func_config == PerFunctionPlotConfig(
        default_args_override={
            "will_update_arg": "test_func arg",
            "will_be_kept_arg": "general arg",
            "will_be_added_arg": "test_func arg overridden by context arg",
            "added_by_context_arg": "added by context arg",
        },
        axis_label_override={
            "will_update_label": "test_func label",
            "will_be_kept_label": "general label",
            "will_be_added_label": "test_func arg overridden by context label",
            "added_by_context_label": "added by context label",
        },
    )

    assert ax1_arg.get_xlabel() == "test_func label"
    assert ax1_arg.get_ylabel() == "general label"
    assert ax2_arg.get_xlabel() == "test_func arg overridden by context label"
    assert ax2_arg.get_ylabel() == "default"

    assert (
        pyglotaran_extras.CONFIG.plotting.get_function_config("test_func")
        == original_test_func_config
    )
