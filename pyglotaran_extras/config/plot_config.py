"""Module containing plot configuration."""

from __future__ import annotations

from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableMapping
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter
from inspect import getcallargs
from inspect import signature
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import TypeAlias
from typing import TypedDict
from typing import cast

import numpy as np
from docstring_parser import parse as parse_docstring
from matplotlib.axes import Axes
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import RootModel
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_serializer
from pydantic import model_validator
from pydantic_core import ErrorDetails
from pydantic_core import PydanticUndefined

from pyglotaran_extras.config.utils import add_yaml_repr

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyglotaran_extras.config.config import Config
    from pyglotaran_extras.types import Param
    from pyglotaran_extras.types import RetType


class DefaultKwarg(TypedDict):
    """Default value and type annotation of a kwarg extracted from the function signature."""

    default: Any
    annotation: str
    docstring: str | None


DefaultKwargs: TypeAlias = Mapping[str, DefaultKwarg]
__PlotFunctionRegistry: MutableMapping[str, DefaultKwargs] = {}


@add_yaml_repr
class PlotLabelOverrideValue(BaseModel):
    """Value of ``PlotLabelOverrideMap``."""

    model_config = ConfigDict(extra="forbid")

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
    """Update json schema to support short notation for ``PlotLabelOverrideValue``."""
    orig_additional_properties = json_schema["additionalProperties"]
    json_schema["additionalProperties"] = {
        "anyOf": [orig_additional_properties, {"type": "string"}]
    }


@add_yaml_repr
class PlotLabelOverrideMap(RootModel, Mapping):
    """Mapping to override axis labels."""

    model_config = ConfigDict(json_schema_extra=_add_short_notation_to_schema)

    root: dict[str, PlotLabelOverrideValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def parse(cls, values: dict[str, Any]) -> dict[str, PlotLabelOverrideValue]:  # noqa: DOC
        """Parse ``axis_label_override`` dictionary supporting verbose and short notation.

        Parameters
        ----------
        values : dict[str, Any]
            Dict that initializes the class.

        Returns
        -------
        dict[str, PlotLabelOverrideValue]
        """
        if values is PydanticUndefined or values is None:
            return {}
        errors: dict[str, ErrorDetails] = {}
        parsed_values: dict[str, PlotLabelOverrideValue] = {}
        for key, value in values.items():
            try:
                if isinstance(value, str):
                    parsed_values[key] = PlotLabelOverrideValue(target_name=value)
                else:
                    parsed_values[key] = PlotLabelOverrideValue.model_validate(value)
            except ValidationError as error:
                errors |= {str(e): e for e in error.errors()}
        if len(errors) > 0:
            raise ValidationError.from_exception_data(cls.__name__, line_errors=[*errors.values()])  # type:ignore[list-item]
        return parsed_values

    def __iter__(self) -> Iterator[str]:  # type:ignore[override] # noqa: DOC
        """Iterate over items."""
        return iter(self.root)

    def __len__(self) -> int:  # noqa: DOC
        """Get number of items."""
        return len(self.root)

    def __getitem__(self, item_label: str) -> PlotLabelOverrideValue:  # noqa: DOC
        """Access items."""
        return self.root[item_label]

    def __contains__(self, item_label: object) -> bool:  # noqa: DOC
        """Check if item is ``in`` the object."""
        return item_label in self.root

    def find_axis_label(self, matplotlib_label: str, axis_name: Literal["x", "y"]) -> str | None:
        """Find axis label even if ``matplotlib`` or the user added a newline in it.

        Parameters
        ----------
        matplotlib_label : str
            Label extracted from the ``matplotlib`` ``Axes`` with ``ax.get_xlabel()`` or
            ``ax.get_xlabel()``.
        axis_name : Literal["x", "y"]
            Name of the axis to find the label for.

        Returns
        -------
        str | None
            Mapped label value if found and None otherwise.
        """
        if matplotlib_label in self and self[matplotlib_label].axis in (axis_name, "both"):
            return self[matplotlib_label].target_name

        # If a label is too long to fit matplotlib inserts a newline which means we can not look it
        # up with string equality
        for key, value in self.root.items():
            if matplotlib_label.replace("\n", "") == key.replace("\n", "") and value.axis in (
                axis_name,
                "both",
            ):
                return value.target_name
        return None


@add_yaml_repr
class PerFunctionPlotConfig(BaseModel):
    """Per function plot configuration."""

    model_config = ConfigDict(extra="forbid")

    default_args_override: dict[str, Any] = Field(
        default_factory=dict,
        description="Default arguments to use if not specified in function call.",
    )
    axis_label_override: PlotLabelOverrideMap | dict[str, str] = Field(
        default_factory=PlotLabelOverrideMap
    )

    @field_validator("axis_label_override", mode="before")
    @classmethod
    def validate_axis_label_override(  # noqa: DOC
        cls, value: PlotLabelOverrideMap | dict[str, str]
    ) -> PlotLabelOverrideMap:
        """Ensure that ``axis_label_override`` gets converted into ``PlotLabelOverrideMap``."""
        return PlotLabelOverrideMap.model_validate(value)

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        """Serialize in a sparse manner leaving out empty values.

        Returns
        -------
        dict[str, Any]
        """
        serialized = {}
        if len(self.default_args_override) > 0:
            serialized["default_args_override"] = self.default_args_override
        if len(self.axis_label_override) > 0:
            serialized["axis_label_override"] = cast(
                PlotLabelOverrideMap, self.axis_label_override
            ).model_dump()
        return serialized

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
        return PerFunctionPlotConfig.model_validate(
            {
                "default_args_override": (
                    self_dict.pop("default_args_override", {})
                    | other_dict.pop("default_args_override", {})
                ),
                "axis_label_override": (
                    self_dict.pop("axis_label_override", {})
                    | other_dict.pop("axis_label_override", {})
                ),
            }
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

    def update_axes_labels(self, axes: Axes | Iterable[Axes]) -> None:
        """Apply label overrides to ``axes``.

        Parameters
        ----------
        axes : Axes | Iterable[Axes]
            Axes to apply the override to.
        """
        if isinstance(axes, Axes):
            self.update_axes_labels((axes,))
            return
        for ax in axes:
            if isinstance(ax, Axes):
                orig_x_label = ax.get_xlabel()
                orig_y_label = ax.get_ylabel()
                axis_label_override = cast(PlotLabelOverrideMap, self.axis_label_override)

                if (
                    override_label := axis_label_override.find_axis_label(orig_x_label, "x")
                ) is not None:
                    ax.set_xlabel(override_label)

                if (
                    override_label := axis_label_override.find_axis_label(orig_y_label, "y")
                ) is not None:
                    ax.set_ylabel(override_label)

            elif isinstance(ax, np.ndarray):
                self.update_axes_labels(ax.flatten())
            else:
                self.update_axes_labels(ax)


@add_yaml_repr
class PlotConfig(BaseModel):
    """Config for plot functions including default args and label overrides."""

    model_config = ConfigDict(extra="allow")

    general: PerFunctionPlotConfig = Field(
        default_factory=PerFunctionPlotConfig,
        description="Config that gets applied to all functions if not specified otherwise.",
    )

    @model_validator(mode="before")
    @classmethod
    def parse(cls, values: dict[str, Any]) -> dict[str, PerFunctionPlotConfig]:
        """Ensure the extra values are converted to ``PerFunctionPlotConfig``.

        Parameters
        ----------
        values : dict[str, Any]
            Dict that initializes the class.

        Returns
        -------
        dict[str, PerFunctionPlotConfig]

        Raises
        ------
        ValidationError
        """
        parsed_values = {}
        errors: dict[str, ErrorDetails] = {}
        for key, value in values.items():
            try:
                parsed_values[key] = PerFunctionPlotConfig.model_validate(value)
            except ValidationError as error:
                errors |= {str(e): {**e, "loc": (key, *e["loc"])} for e in error.errors()}
        if len(errors) > 0:
            raise ValidationError.from_exception_data(cls.__name__, line_errors=[*errors.values()])  # type:ignore[list-item]
        return parsed_values

    def get_function_config(self, function_name: str) -> PerFunctionPlotConfig:
        """Get config for a specific function.

        Parameters
        ----------
        function_name : str
            Name of the function to get the config for.

        Returns
        -------
        PerFunctionPlotConfig
        """
        function_config = self.general
        if self.model_extra is not None and function_name in self.model_extra:
            function_config = function_config.merge(self.model_extra[function_name])
        if hasattr(self, "__context_config"):
            function_config = function_config.merge(getattr(self, "__context_config"))
        return function_config

    def merge(self, other: PlotConfig) -> PlotConfig:  # noqa: C901
        """Merge two ``PlotConfig``'s where ``other`` overrides values.

        Parameters
        ----------
        other : PlotConfig
            Other ``PlotConfig`` to merge in.

        Returns
        -------
        PlotConfig
        """
        updated: dict[str, PerFunctionPlotConfig] = {}
        # Update general field
        for key in self.model_fields_set:
            updated[key] = cast(PerFunctionPlotConfig, getattr(self, key))
            if key in other.model_fields_set:
                updated[key] = updated[key].merge(cast(PerFunctionPlotConfig, getattr(other, key)))
        for key in other.model_fields_set:
            if key not in updated:
                updated[key] = getattr(other, key)
        # Update model_extra
        if self.model_extra is not None:
            for key, value in self.model_extra.items():
                updated[key] = cast(PerFunctionPlotConfig, value)
                if other.model_extra is not None and key in other.model_extra:
                    updated[key] = updated[key].merge(
                        cast(PerFunctionPlotConfig, other.model_extra[key])
                    )
        if other.model_extra is not None:
            for key, value in other.model_extra.items():
                if key not in updated:
                    updated[key] = value

        return PlotConfig.model_validate(updated)


def create_parameter_docstring_mapping(func: Callable[..., Any]) -> Mapping[str, str]:
    """Create a mapping of parameter names and they docstrings.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to create the parameter docstring mapping for.

    Returns
    -------
    Mapping[str, str]
    """
    param_docstring_mapping = {}
    for param in parse_docstring(func.__doc__ if func.__doc__ is not None else "").params:
        if param.description is not None:
            param_docstring_mapping[param.arg_name] = " ".join(param.description.splitlines())
    return param_docstring_mapping


def extract_default_kwargs(
    func: Callable[..., Any], exclude_kwargs: tuple[str, ...]
) -> DefaultKwargs:
    """Extract the default kwargs of ``func`` from its signature.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to extract the default args from.
    exclude_kwargs : tuple[str, ...]
        Names of keyword arguments that should be excluded.

    Returns
    -------
    DefaultKwargs

    See Also
    --------
    use_plot_config
    """
    sig = signature(func)
    param_docstring_mapping = create_parameter_docstring_mapping(func)
    return {
        k: {
            "default": v.default,
            "annotation": v.annotation if v.annotation is not Parameter.empty else "object",
            "docstring": param_docstring_mapping.get(k, None),
        }
        for k, v in sig.parameters.items()
        if k not in exclude_kwargs
        and v.default is not Parameter.empty
        and v.kind is not Parameter.POSITIONAL_ONLY
    }


def find_not_user_provided_kwargs(
    default_kwargs: DefaultKwargs, arg_names: Iterable[str], kwargs: Mapping[str, Any]
) -> set[str]:
    """Find which kwargs of a function were not provided by the user.

    Those kwargs can be overridden by config value.

    Parameters
    ----------
    default_kwargs : DefaultKwargs
        Default keyword arguments to the function.
    arg_names : Iterable[str]
        Names of the positional arguments passed when calling the function.
    kwargs : Mapping[str, Any]
        Kwargs passed when calling the function.

    Returns
    -------
    set[str]

    See Also
    --------
    extract_default_kwargs
    """
    return {k for k in default_kwargs if k not in kwargs and k not in arg_names}


def find_axes(
    values: Iterable[Any],
) -> Generator[Axes, None, None]:
    """Iterate over values and yield the values that are ``Axes``.

    Parameters
    ----------
    values : Iterable[Any]
        Values to look for an ``Axes`` values in.

    Yields
    ------
    Axes
    """
    for value in values:
        if isinstance(value, str):
            continue
        elif isinstance(value, Axes):
            yield value
        elif isinstance(value, np.ndarray):
            yield from find_axes(value.flatten())
        elif isinstance(value, Iterable):
            yield from find_axes(value)


def use_plot_config(  # noqa: DOC201, DOC203
    exclude_from_config: tuple[str, ...] = (),
) -> Callable[[Callable[Param, RetType]], Callable[Param, RetType]]:
    """Decorate plot functions to register it and enables auto use of config.

    Parameters
    ----------
    exclude_from_config : tuple[str, ...]
        Names of keyword argument with default for which the type can not be represent in the
        config. Defaults to ()
    """

    def outer_wrapper(func: Callable[Param, RetType]) -> Callable[Param, RetType]:  # noqa: DOC
        """Outer wrapper to allow for ``ignore_kwargs`` to be passed."""
        default_kwargs = extract_default_kwargs(func, exclude_from_config)
        __PlotFunctionRegistry[func.__name__] = default_kwargs

        @wraps(func)
        def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:  # noqa: DOC
            """Wrap function and apply config."""
            from pyglotaran_extras import CONFIG

            CONFIG.reload()

            arg_names = func.__code__.co_varnames[: len(args)]
            not_user_provided_kwargs = find_not_user_provided_kwargs(
                default_kwargs, arg_names, kwargs
            )
            function_config = CONFIG.plotting.get_function_config(func.__name__)
            override_kwargs = function_config.find_override_kwargs(not_user_provided_kwargs)
            updated_kwargs = kwargs | override_kwargs
            arg_axes = find_axes(getcallargs(func, *args, **updated_kwargs).values())
            return_values = func(*args, **updated_kwargs)
            function_config.update_axes_labels(arg_axes)

            if isinstance(return_values, Iterable):
                return_axes = find_axes(return_values)
                function_config.update_axes_labels(return_axes)

            return return_values

        return wrapper

    return outer_wrapper


@contextmanager
def plot_config_context(plot_config: PerFunctionPlotConfig) -> Generator[Config, None, None]:
    """Context manager to override parts of the resolved functions ``PlotConfig``.

    Parameters
    ----------
    plot_config : PerFunctionPlotConfig
        Function plot config override to update plot config for functions run inside of context.

    Yields
    ------
    Config
    """
    from pyglotaran_extras import CONFIG

    setattr(
        CONFIG.plotting,
        "__context_config",
        PerFunctionPlotConfig.model_validate(plot_config),
    )
    yield CONFIG
    delattr(CONFIG.plotting, "__context_config")
