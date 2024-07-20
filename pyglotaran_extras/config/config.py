"""Module containing configuration."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PrivateAttr
from pydantic import PydanticUserError
from pydantic import create_model
from pydantic.fields import FieldInfo
from ruamel.yaml import YAML

from pyglotaran_extras.config.plot_config import PlotConfig
from pyglotaran_extras.config.plot_config import PlotLabelOverrideMap
from pyglotaran_extras.config.plot_config import __PlotFunctionRegistry
from pyglotaran_extras.config.utils import add_yaml_repr
from pyglotaran_extras.io.setup_case_study import get_script_dir

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Iterable

# Only imported for builtin schema generation
from collections.abc import Sequence  # noqa: F401
from typing import Literal  # noqa: F401

CONFIG_FILE_STEM = "pygta_config"

EXPORT_TEMPLATE = """\
# yaml-language-server: $schema={schema_path}

{config_yaml}\
"""


class UsePlotConfigError(Exception):
    """Error thrown when ``use_plot_config`` has none json serializable kwargs."""

    def __init__(self, func_name: str, error: PydanticUserError) -> None:  # noqa: DOC
        """Use ``func_name`` and original ``error`` to create error message."""
        msg = (
            f"The function ``{func_name}`` decorated with ``use_plot_config`` has an keyword "
            "argument with a type annotation can not be represents in the config.\n"
            "Please use the name of this keyword argument in the ``exclude_from_config`` "
            "keyword argument to ``use_plot_config``.\n"
            f"Original error:\n{error}"
        )
        super().__init__(msg)


@add_yaml_repr
class Config(BaseModel):
    """Main configuration class."""

    model_config = ConfigDict(extra="forbid")

    plotting: PlotConfig = PlotConfig()
    _source_files: list[Path] = PrivateAttr(default_factory=list)
    _source_hash: int = PrivateAttr(default=hash(()))

    def merge(self, other: Config) -> Config:
        """Merge two ``Config``'s where ``other`` overrides values and return a new instance.

        Parameters
        ----------
        other : Config
            Other ``Config`` to merge in.

        Returns
        -------
        Config
        """
        merged = self.model_copy(deep=True)
        merged.plotting = merged.plotting.merge(other.plotting)
        for source_file in other._source_files:
            if source_file in merged._source_files:
                merged._source_files.remove(source_file)
            merged._source_files.append(source_file)
        merged._source_hash = merged._calculate_source_hash()
        return merged

    def _reset(self, other: Config | None = None) -> Config:
        """Reset self to ``other`` config or default initialization.

        Parameters
        ----------
        other : Config | None
            Other ``Config`` to to reset to.

        Returns
        -------
        Config
        """
        if other is None:
            other = Config()
        else:
            self._source_files = other._source_files
        self.plotting = other.plotting
        return self

    def _calculate_source_hash(self) -> int:  # noqa: DOC
        """Calculate hash of source files based on their modification time."""
        return hash(tuple(source_file.stat().st_mtime for source_file in self._source_files))

    def reload(self) -> Config:
        """Reset and reload config from files.

        Returns
        -------
        Config
        """
        if self._source_hash == self._calculate_source_hash():
            return self
        context_config = getattr(self.plotting, "__context_config", None)
        merged = self._reset()
        for config in load_config_files(self._source_files):
            merged = merged.merge(config)
        self.plotting = merged.plotting
        if context_config is not None:
            setattr(self.plotting, "__context_config", context_config)
        self._source_hash = merged._source_hash
        return self

    def load(self, config_file_path: Path | str) -> Config:
        """Disregard current config and config file paths, and reload from ``config_file_path``.

        Parameters
        ----------
        config_file_path : Path | str
            Path to the config file to load.

        Returns
        -------
        Config
        """
        self._source_files = [Path(config_file_path)]
        return self.reload()

    def export(self, export_folder: Path | str | None = None, *, update: bool = True) -> Path:
        """Export current config and schema to ``export_folder``.

        Parameters
        ----------
        export_folder : Path | str | None
            Folder to export config and scheme to. Defaults to None, which means that the script
            folder is used
        update : bool
            Whether to update or overwrite and existing config file. Defaults to True

        Returns
        -------
        Path
            Path to exported config file.
        """
        if export_folder is None:
            from pyglotaran_extras import SCRIPT_DIR

            export_folder = SCRIPT_DIR
        else:
            export_folder = Path(export_folder)
        export_folder.mkdir(parents=True, exist_ok=True)
        schema_path = create_config_schema(export_folder)
        export_path = export_folder / f"{CONFIG_FILE_STEM}.yml"
        if export_path.is_file() is True and update is True:
            merged = Config().load(export_path).merge(self)
            config = merged
        else:
            config = self
        export_path.write_text(
            EXPORT_TEMPLATE.format(schema_path=schema_path.name, config_yaml=config),
            encoding="utf8",
        )
        return export_path

    def rediscover(self, *, include_home_dir: bool = True, lookup_depth: int = 2) -> list[Path]:
        """Rediscover config paths based on the ``SCRIPT_DIR`` discovered on import.

        Parameters
        ----------
        include_home_dir : bool
            Where or not to include the users home folder in the config lookup. Defaults to True
        lookup_depth : int
            Depth at which to look for configs in parent folders of ``script_dir``.
            If set to ``1`` only ``script_dir`` will be considered as config dir.
            Defaults to ``2``.

        Returns
        -------
        list[Path]
            Paths of the discovered config files.
        """
        from pyglotaran_extras import SCRIPT_DIR

        self._source_files = list(
            discover_config_files(
                SCRIPT_DIR, include_home_dir=include_home_dir, lookup_depth=lookup_depth
            )
        )
        return self._source_files

    def init_project(self) -> Config:
        """Initialize configuration for the current project.

        This will use the configs discovered and resolved config during import to create a new
        config and schema for your current project inside of your working directory (script dir),
        if it didn't exist before.

        Returns
        -------
        Config
        """
        from pyglotaran_extras import SCRIPT_DIR

        if any(find_config_in_dir(SCRIPT_DIR)) is False:
            self.export()
        self.rediscover()
        self.reload()
        return self


def find_config_in_dir(dir_path: Path) -> Generator[Path, None, None]:
    """Find the config file inside of dir ``dir_path``.

    Parameters
    ----------
    dir_path : Path
        Directory path to look for a config file.

    Yields
    ------
    Path
    """
    for extension in (".yaml", ".yml"):
        config_file = (dir_path / CONFIG_FILE_STEM).with_suffix(extension)
        if config_file.is_file():
            yield config_file


def discover_config_files(
    script_dir: Path, *, include_home_dir: bool = True, lookup_depth: int = 2
) -> Generator[Path, None, None]:
    """Find config files in the users home folder and the current working dir and parents.

    Parameters
    ----------
    script_dir : Path
        Path to the current scripts/notebooks parent folder.
    include_home_dir : bool
        Where or not to include the users home folder in the config lookup. Defaults to True
    lookup_depth : int
        Depth at which to look for configs in parent folders of ``script_dir``.
        If set to ``1`` only ``script_dir`` will be considered as config dir.
        Defaults to ``2``.

    Yields
    ------
    Path
    """
    if include_home_dir is True:
        yield from find_config_in_dir(Path.home())
    parent_dirs = tuple(reversed((script_dir / "dummy").parents))
    if lookup_depth > 0 and lookup_depth <= len(parent_dirs):
        parent_dirs = parent_dirs[-lookup_depth:]
    for parent in parent_dirs:
        yield from find_config_in_dir(parent)


def load_config_files(config_paths: Iterable[Path]) -> Generator[Config, None, None]:
    """Load config files into new config instances.

    Parameters
    ----------
    config_paths : Iterable[Path]
        Path to the config file.

    Yields
    ------
    Config
    """
    yaml = YAML()
    for config_path in config_paths:
        try:
            config_dict = yaml.load(config_path)
            config = Config.model_validate(config_dict) if config_dict is not None else Config()
            config._source_files.append(config_path)
            yield config
        # We use a very broad range of exception to ensure the config loading at import never
        # breaks importing
        except Exception as error:  # noqa: BLE001
            print(  # noqa: T201
                "Error loading the config:\n",
                f"Source path: {config_path.as_posix()}\n",
                f"Error: {error}",
                file=sys.stderr,
                sep="",
            )


def merge_configs(configs: Iterable[Config]) -> Config:
    """Merge ``Config``'s from left to right, where the right ``Config`` overrides the left.

    Parameters
    ----------
    configs : Iterable[Config]
        Config instances to merge together.

    Returns
    -------
    Config
    """
    full_config = Config()
    for config in configs:
        full_config = full_config.merge(config)
    return full_config


def load_config(
    script_dir: Path, *, include_home_dir: bool = True, lookup_depth: int = 2
) -> Config:
    """Discover and load config files.

    Parameters
    ----------
    script_dir : Path
        Path to the current scripts/notebooks parent folder.
    include_home_dir : bool
        Where or not to include the users home folder in the config lookup. Defaults to True
    lookup_depth : int
        Depth at which to look for configs in parent folders of ``script_dir``.
        If set to ``1`` only ``script_dir`` will be considered as config dir.
        Defaults to ``2``.

    Returns
    -------
    Config

    See Also
    --------
    discover_config_files
    """
    config_paths = discover_config_files(
        script_dir, include_home_dir=include_home_dir, lookup_depth=lookup_depth
    )
    configs = load_config_files(config_paths)
    return merge_configs(configs)


def _find_script_dir_at_import(package_root_file: str) -> Path:
    """Find the script dir when importing ``pyglotaran_extras``.

    The assumption is that the first file not inside of ``pyglotaran_extras`` or importlib
    is the script in question.
    The max ``nesting_offset`` of 20 was chosen semi arbitrarily (typically ``nesting + offset``
    is around 9-13 depending on the import) to ensure that there won't be an infinite loop.

    Parameters
    ----------
    package_root_file : str
        The dunder file attribute (``__file__``) in the package root file.

    Returns
    -------
    Path
    """
    nesting_offset = 0
    importlib_path = Path(importlib.__file__).parent
    package_root = Path(package_root_file).parent
    script_dir = get_script_dir(nesting=2)
    while (
        importlib_path in (script_dir / "dummy").parents
        or package_root in (script_dir / "dummy").parents
    ) and nesting_offset < 20:
        nesting_offset += 1
        script_dir = get_script_dir(nesting=2 + nesting_offset)
    return script_dir


def create_config_schema(
    output_folder: Path | str | None = None,
    file_name: Path | str = f"{CONFIG_FILE_STEM}.schema.json",
) -> Path:
    """Create json schema file to be used for autocompletion and linting of the config.

    Parameters
    ----------
    output_folder : Path | str | None
        Folder to write schema file to. Defaults to None, which means that the script
            folder is used
    file_name : Path | str
        Name of the scheme file. Defaults to "pygta_config.schema.json"

    Returns
    -------
    Path
        Path to the file the schema got saved to.

    Raises
    ------
    UsePlotConfigError
        If any function decorated with ``use_plot_config`` has a keyword argument with a default
        value and a type annotation that can not be serialized into a json schema.
    """
    json_schema = Config.model_json_schema()
    general_kwargs: dict[str, Any] = {}

    for function_name, default_kwargs in __PlotFunctionRegistry.items():
        try:
            name_prefix = "".join([parts.capitalize() for parts in function_name.split("_")])
            fields: Any = {
                kwarg_name: (
                    kwarg_value["annotation"],
                    FieldInfo(
                        default=kwarg_value["default"], description=kwarg_value["docstring"]
                    ),
                )
                for kwarg_name, kwarg_value in default_kwargs.items()
            }
            kwargs_model_name = f"{name_prefix}Kwargs"
            func_kwargs = create_model(
                kwargs_model_name,
                __config__=ConfigDict(extra="forbid"),
                __doc__=(
                    f"Default arguments to use for ``{function_name}``, "
                    "if not specified in function call."
                ),
                **fields,
            )
            config_model_name = f"{name_prefix}Config"
            func_config = create_model(
                config_model_name,
                __config__=ConfigDict(extra="forbid"),
                __doc__=(
                    f"Plot function configuration specific to ``{function_name}`` "
                    "(overrides values in general)."
                ),
                default_args_override=(func_kwargs, {}),
                axis_label_override=(PlotLabelOverrideMap, PlotLabelOverrideMap()),
            )
            func_json_schema = func_config.model_json_schema()
            general_kwargs |= func_json_schema["$defs"][kwargs_model_name]["properties"]
            json_schema["$defs"] |= func_json_schema.pop("$defs")
            json_schema["$defs"][config_model_name] = func_json_schema
            json_schema["$defs"]["PlotConfig"]["properties"][function_name] = {
                "allOf": [{"$ref": f"#/$defs/{config_model_name}"}]
            }
        except PydanticUserError as error:
            raise UsePlotConfigError(function_name, error)  # noqa: B904
    json_schema["$defs"]["PerFunctionPlotConfig"]["properties"]["default_args_override"][
        "properties"
    ] = general_kwargs
    json_schema["$defs"]["PerFunctionPlotConfig"]["properties"]["default_args_override"][
        "additionalProperties"
    ] = False
    if output_folder is None:
        from pyglotaran_extras import SCRIPT_DIR

        output_folder = SCRIPT_DIR
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / file_name
    output_file.write_text(json.dumps(json_schema, ensure_ascii=False), encoding="utf8")
    return output_file
