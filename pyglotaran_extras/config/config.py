"""Module containing configuration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import PrivateAttr
from ruamel.yaml import YAML

from pyglotaran_extras.config.plot_config import PlotConfig

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
