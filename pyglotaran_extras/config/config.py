"""Module containing configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import PrivateAttr
from ruamel.yaml import YAML

from pyglotaran_extras.config.plot_config import PlotConfig
from pyglotaran_extras.io.setup_case_study import get_script_dir

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Iterable

CONFIG_FILE_STAM = "pyglotaran_extras_config"


class Config(BaseModel):
    """Main configuration class."""

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
        config_file = (dir_path / CONFIG_FILE_STAM).with_suffix(extension)
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
        config_dict = yaml.load(config_path)
        config = Config.model_validate(config_dict) if config_dict is not None else Config()
        config._source_files.append(config_path)
        yield config


def merge_configs(configs: Iterable[Config]) -> Config | None:
    """Merge ``Config``'s from left to right, where the right ``Config`` overrides the left.

    Parameters
    ----------
    configs : Iterable[Config]
        Config instances to merge together.

    Returns
    -------
    Config | None
    """
    full_config = None
    for config in configs:
        if full_config is None:  # noqa: SIM108
            full_config = config
        else:
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
    config = merge_configs(configs)
    return config if config is not None else Config()


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
