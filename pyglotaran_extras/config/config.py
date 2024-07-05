"""Module containing configuration."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PrivateAttr
from pydantic import create_model
from pydantic.fields import FieldInfo
from ruamel.yaml import YAML

from pyglotaran_extras.config.plot_config import PlotConfig
from pyglotaran_extras.config.plot_config import PlotLabelOverRideMap
from pyglotaran_extras.config.plot_config import __PlotFunctionRegistry
from pyglotaran_extras.io.setup_case_study import get_script_dir

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Iterable

CONFIG_FILE_STEM = "pyglotaran_extras_config"

EXPORT_TEMPLATE = """\
# yaml-language-server: $schema={schema_path}

{config_yaml}\
"""


class Config(BaseModel):
    """Main configuration class."""

    model_config = ConfigDict(extra="forbid")

    plotting: PlotConfig = PlotConfig()
    _source_files: list[Path] = PrivateAttr(default_factory=list)

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
        return merged

    def _reset(self) -> Config:
        """Reset self to default initialization.

        Returns
        -------
        Config
        """
        self.plotting = PlotConfig()
        return self

    def reload(self) -> Config:
        """Reset and reload config from files.

        Returns
        -------
        Config
        """
        self._reset()
        merged = Config()
        for config in load_config_files(self._source_files):
            merged = merged.merge(config)
        self.plotting = merged.plotting
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

    def export(self, export_folder: Path | str = ".", *, update: bool = True) -> Path:
        """Export current config and schema to ``export_folder``.

        Parameters
        ----------
        export_folder : Path | str
            Folder to export config and scheme to. Defaults to "."
        update : bool
            Whether to update or overwrite and existing config file. Defaults to True

        Returns
        -------
        Path
            Path to exported config file.
        """
        export_folder = Path(export_folder)
        export_folder.mkdir(parents=True, exist_ok=True)
        schema_path = create_config_schema(export_folder)
        export_path = export_folder / f"{CONFIG_FILE_STEM}.yml"
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        buffer = StringIO()
        if export_path.is_file() is True and update is True:
            merged = Config().load(export_path).merge(self)
            yaml.dump(merged.model_dump(), buffer)
        else:
            yaml.dump(self.model_dump(), buffer)
        buffer.seek(0)
        export_path.write_text(
            EXPORT_TEMPLATE.format(schema_path=schema_path.name, config_yaml=buffer.read()),
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
    importlib_path = Path(sys.executable).parent / "lib/importlib"
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
    output_folder: Path | str = ".", file_name: Path | str = "pyglotaran_extras_config.schema.json"
) -> Path:
    """Create json schema file to be used for autocompletion and linting of the config.

    Parameters
    ----------
    output_folder : Path | str
        Folder to write schema file to. Defaults to "."
    file_name : Path | str
        Name of the scheme file. Defaults to "pyglotaran_extras_config_schema.json"

    Returns
    -------
    Path
        Path to the file the schema got saved to.
    """
    json_schema = Config.model_json_schema()
    general_kwargs: dict[str, Any] = {}

    for function_name, default_kwargs in __PlotFunctionRegistry.items():
        name_prefix = "".join([parts.capitalize() for parts in function_name.split("_")])
        fields = {
            kwarg_name: (
                kwarg_value["annotation"],
                FieldInfo(default=kwarg_value["default"], description=kwarg_value["docstring"]),
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
            axis_label_override=(PlotLabelOverRideMap, PlotLabelOverRideMap()),
        )
        func_json_schema = func_config.model_json_schema()
        general_kwargs |= func_json_schema["$defs"][kwargs_model_name]["properties"]
        json_schema["$defs"] |= func_json_schema.pop("$defs")
        json_schema["$defs"][config_model_name] = func_json_schema
        json_schema["$defs"]["PlotConfig"]["properties"][function_name] = {
            "allOf": [{"$ref": f"#/$defs/{config_model_name}"}]
        }
    json_schema["$defs"]["PerFunctionPlotConfig"]["properties"]["default_args_override"][
        "properties"
    ] = general_kwargs
    json_schema["$defs"]["PerFunctionPlotConfig"]["properties"]["default_args_override"][
        "additionalProperties"
    ] = False
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / file_name
    output_file.write_text(json.dumps(json_schema, ensure_ascii=False), encoding="utf8")
    return output_file
