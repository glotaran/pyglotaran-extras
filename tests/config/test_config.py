"""Tests for ``pyglotaran_extras.config.config``."""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

from pyglotaran_extras.config.config import CONFIG_FILE_STAM
from pyglotaran_extras.config.config import Config
from pyglotaran_extras.config.config import discover_config_files
from pyglotaran_extras.config.config import find_config_in_dir
from pyglotaran_extras.config.config import load_config
from pyglotaran_extras.config.config import load_config_files
from pyglotaran_extras.config.config import merge_configs
from pyglotaran_extras.config.plot_config import PlotConfig
from tests import TEST_DATA
from tests.conftest import generator_is_exhausted


def test_config():
    """Empty and from values initialization works."""
    empty_config = Config()

    assert empty_config.plotting == PlotConfig()

    test_config_values = YAML().load(TEST_DATA / "config/pyglotaran_extras_config.yml")

    test_config = Config.model_validate(test_config_values)
    test_plot_config = PlotConfig.model_validate(test_config_values["plotting"])

    assert test_config.plotting == test_plot_config


def test_config_merge():
    """Merging creates the expected output.

    - Update fields that are present in original and update
    - Keep fields that are not present in the update
    - Add field that is only present in update
    """
    original_values, update_values, expected_values = tuple(
        YAML().load_all(TEST_DATA / "config/config_merge.yml")
    )
    original = Config.model_validate(original_values)
    update = Config.model_validate(update_values)
    expected = Config.model_validate(expected_values)

    assert original.merge(update) == expected

    config_with_paths = Config()
    config_with_paths._source_files = [Path("foo"), Path("bar"), Path("baz")]

    update = Config()
    update._source_files = [Path("foo")]

    assert config_with_paths.merge(update)._source_files == [Path("bar"), Path("baz"), Path("foo")]


def test_find_config_in_dir(tmp_path: Path):
    """Find one or two config files if present."""
    assert len(list(find_config_in_dir(tmp_path))) == 0

    yml_config_path = tmp_path / f"yml/{CONFIG_FILE_STAM}.yml"
    yml_config_path.parent.mkdir()
    yml_config_path.touch()

    assert next(find_config_in_dir(yml_config_path.parent)) == yml_config_path

    yaml_config_path = tmp_path / f"yaml/{CONFIG_FILE_STAM}.yaml"
    yaml_config_path.parent.mkdir()
    yaml_config_path.touch()

    assert next(find_config_in_dir(yaml_config_path.parent)) == yaml_config_path

    multi_config_dir = tmp_path / "multi"
    multi_config_dir.mkdir()
    (multi_config_dir / f"{CONFIG_FILE_STAM}.yml").touch()
    (multi_config_dir / f"{CONFIG_FILE_STAM}.yaml").touch()
    (multi_config_dir / f"{CONFIG_FILE_STAM}.json").touch()
    (multi_config_dir / f"{CONFIG_FILE_STAM}1.yml").touch()

    assert len(list(find_config_in_dir(multi_config_dir))) == 2


def test_discover_config_files(tmp_path: Path, mock_home: Path):
    """Discover all config files in the correct order."""
    file_name = f"{CONFIG_FILE_STAM}.yml"
    script_dir = tmp_path / "top_project/project/sub_project"
    script_dir.mkdir(parents=True)

    home_config = mock_home / file_name
    home_config.touch()

    top_project_config = script_dir.parent.parent / file_name
    top_project_config.touch()

    project_config = script_dir.parent / file_name
    project_config.touch()

    sub_project_config = script_dir / file_name
    sub_project_config.touch()

    default_discovery = discover_config_files(script_dir)

    assert next(default_discovery) == home_config
    assert next(default_discovery) == project_config
    assert next(default_discovery) == sub_project_config
    assert generator_is_exhausted(default_discovery) is True

    no_recursion_discovery = discover_config_files(script_dir, include_home_dir=False)

    assert next(no_recursion_discovery) == project_config
    assert next(no_recursion_discovery) == sub_project_config
    assert generator_is_exhausted(no_recursion_discovery) is True

    no_recursion_discovery = discover_config_files(script_dir, lookup_depth=1)

    assert next(no_recursion_discovery) == home_config
    assert next(no_recursion_discovery) == sub_project_config
    assert generator_is_exhausted(no_recursion_discovery) is True

    top_project_discovery = discover_config_files(script_dir, lookup_depth=3)

    assert next(top_project_discovery) == home_config
    assert next(top_project_discovery) == top_project_config
    assert next(top_project_discovery) == project_config
    assert next(top_project_discovery) == sub_project_config
    assert generator_is_exhausted(top_project_discovery) is True


def test_load_config_files(tmp_path: Path):
    """Read configs and add source path."""
    empty_config_file = tmp_path / f"{CONFIG_FILE_STAM}.yml"
    empty_config_file.touch()
    test_config_path = TEST_DATA / "config/pyglotaran_extras_config.yml"

    empty_file_loaded = load_config_files([empty_config_file])

    empty_config = next(empty_file_loaded)

    assert empty_config.model_dump() == Config().model_dump()
    assert empty_config._source_files == [empty_config_file]
    assert generator_is_exhausted(empty_file_loaded) is True

    two_configs = load_config_files([empty_config_file, test_config_path])

    empty_config = next(two_configs)

    assert empty_config.model_dump() == Config().model_dump()
    assert empty_config._source_files == [empty_config_file]

    test_config_values = YAML().load(test_config_path)

    expected_config = Config.model_validate(test_config_values)

    test_config = next(two_configs)

    assert test_config.model_dump() == expected_config.model_dump()
    assert test_config._source_files == [test_config_path]
    assert generator_is_exhausted(two_configs) is True


def test_merge_configs():
    """Check that the most right config overrides other values.

    Since we already tested all permutations for 2 configs in ``test_config_merge``
    this test can fucus on the case with more than 2 configs.
    """
    assert merge_configs([]) is None

    original_values, update_values, expected_values = tuple(
        YAML().load_all(TEST_DATA / "config/config_merge.yml")
    )
    additional_update_values = {
        "plotting": {
            "general": {"default_args_override": {"will_update_arg": "additional update"}}
        }
    }
    expected_values["plotting"]["general"]["default_args_override"]["will_update_arg"] = (
        "additional update"
    )

    original = Config.model_validate(original_values)
    update = Config.model_validate(update_values)
    additional_update = Config.model_validate(additional_update_values)
    expected = Config.model_validate(expected_values)

    assert merge_configs([original, update, additional_update]) == expected


def test_load_config(tmp_path: Path, mock_home: Path):
    """Load config and check that args are passed on to ``discover_config_files``."""
    assert load_config(tmp_path) == Config()

    yaml = YAML()
    original_values, update_values, expected_values = tuple(
        yaml.load_all(TEST_DATA / "config/config_merge.yml")
    )
    additional_update_values = {
        "plotting": {
            "general": {"default_args_override": {"will_update_arg": "additional update"}}
        }
    }
    expected_values["plotting"]["general"]["default_args_override"]["will_update_arg"] = (
        "additional update"
    )

    file_name = f"{CONFIG_FILE_STAM}.yml"
    script_dir = tmp_path / "top_project/project/sub_project"
    script_dir.mkdir(parents=True)

    home_config_path = mock_home / file_name
    yaml.dump(original_values, home_config_path)

    project_config_path = script_dir.parent / file_name
    yaml.dump(update_values, project_config_path)

    sub_project_config_path = script_dir / file_name
    yaml.dump(additional_update_values, sub_project_config_path)

    loaded_config = load_config(script_dir)

    assert loaded_config.model_dump() == Config.model_validate(expected_values).model_dump()
    assert loaded_config._source_files == [
        home_config_path,
        project_config_path,
        sub_project_config_path,
    ]

    minimal_lookup_config = load_config(script_dir, include_home_dir=False, lookup_depth=1)

    assert (
        minimal_lookup_config.model_dump()
        == Config.model_validate(additional_update_values).model_dump()
    )
    assert minimal_lookup_config._source_files == [sub_project_config_path]
