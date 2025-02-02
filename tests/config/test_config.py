"""Tests for ``pyglotaran_extras.config.config``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from shutil import copyfile
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from packaging.version import Version
from pydantic import __version__ as pydantic_version
from ruamel.yaml import YAML

from pyglotaran_extras import create_config_schema
from pyglotaran_extras.config.config import CONFIG_FILE_STEM
from pyglotaran_extras.config.config import Config
from pyglotaran_extras.config.config import discover_config_files
from pyglotaran_extras.config.config import find_config_in_dir
from pyglotaran_extras.config.config import load_config
from pyglotaran_extras.config.config import load_config_files
from pyglotaran_extras.config.config import merge_configs
from pyglotaran_extras.config.plot_config import PlotConfig
from pyglotaran_extras.config.plot_config import use_plot_config
from tests import TEST_DATA
from tests.conftest import generator_is_exhausted


def test_config(test_config_values: dict[str, Any]):
    """Empty and from values initialization works."""
    empty_config = Config()

    assert empty_config.plotting == PlotConfig()

    test_config = Config.model_validate(test_config_values)
    test_plot_config = PlotConfig.model_validate(test_config_values["plotting"])

    assert test_config.plotting == test_plot_config


def test_config_merge(tmp_path: Path):
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
    config_with_paths._source_files = [tmp_path / "foo", tmp_path / "bar", tmp_path / "baz"]
    [file.touch() for file in config_with_paths._source_files]  # type:ignore[func-returns-value]

    update = Config()
    update._source_files = [tmp_path / "foo"]

    assert config_with_paths.merge(update)._source_files == [
        tmp_path / "bar",
        tmp_path / "baz",
        tmp_path / "foo",
    ]


def test_config_reset(tmp_path: Path, test_config_values: dict[str, Any]):
    """All config values but the source files get reset when no other config is passed.

    If another config is passed all values (including source files) are reset.
    """

    test_config_path = tmp_path / f"{CONFIG_FILE_STEM}.yml"
    test_config = Config.model_validate(test_config_values)
    test_config._source_files = [test_config_path]

    test_config._reset()

    assert test_config.plotting == PlotConfig()
    assert test_config._source_files == [test_config_path]

    other = Config.model_validate(test_config_values)

    test_config._reset(other)

    assert test_config.plotting == PlotConfig.model_validate(test_config_values["plotting"])
    assert test_config._source_files == []
    assert test_config == other


def test_config_reload(tmp_path: Path, test_config_values: dict[str, Any]):
    """Config values get reloaded from file."""

    test_config_path = tmp_path / f"{CONFIG_FILE_STEM}.yml"
    test_config = Config.model_validate(test_config_values)
    test_config._source_files = [test_config_path]

    test_config_values["plotting"]["general"]["default_args_override"]["will_update_arg"] = (
        "file got updated"
    )

    YAML().dump(test_config_values, test_config_path)

    expected_plot_config = test_config.plotting.model_copy(deep=True)
    expected_plot_config.general.default_args_override["will_update_arg"] = "file got updated"

    test_config.reload()

    assert test_config.plotting.model_dump() == expected_plot_config.model_dump()
    assert test_config._source_files == [test_config_path]


def test_config_load(tmp_path: Path, test_config_values: dict[str, Any], test_config_file: Path):
    """Loading config replaces its values and and source files."""

    test_config = Config()
    test_config._source_files = [test_config_file]
    test_config.reload()

    test_config_values["plotting"]["general"]["default_args_override"]["will_update_arg"] = (
        "from new file"
    )

    test_config_path = tmp_path / f"{CONFIG_FILE_STEM}.yml"
    YAML().dump(test_config_values, test_config_path)

    expected_plot_config = test_config.plotting.model_copy(deep=True)
    expected_plot_config.general.default_args_override["will_update_arg"] = "from new file"

    test_config.load(test_config_path)

    assert test_config.plotting.model_dump() == expected_plot_config.model_dump()
    assert test_config._source_files == [test_config_path]


def test_config_export(tmp_path: Path, test_config_file: Path):
    """Exporting the config gives the expected result."""
    config = Config().load(test_config_file)

    exported_config_path = config.export(tmp_path)

    assert exported_config_path.is_file() is True
    assert exported_config_path.samefile(tmp_path / f"{CONFIG_FILE_STEM}.yml") is True
    assert (tmp_path / f"{CONFIG_FILE_STEM}.schema.json").is_file() is True

    assert exported_config_path.read_text(encoding="utf8") == test_config_file.read_text(
        encoding="utf8"
    )


def test_config_export_update(tmp_path: Path, test_config_file: Path):
    """Update existing config by default and overwrite if update kwarg is ``False``."""
    config = Config().load(test_config_file)
    existing_config_path = config.export(tmp_path)

    update_config = Config(
        plotting=PlotConfig.model_validate(
            {"test_func": {"axis_label_override": {"will_update_label": "new label"}}}
        )
    )
    update_config.export(tmp_path)
    # Ensure there is no comparison conflict due to different source file paths
    config._source_files = [existing_config_path]
    update_config._source_files = [existing_config_path]

    assert Config().load(existing_config_path) == config.merge(update_config)

    update_config.export(tmp_path, update=False)

    expected_config = Config().load(existing_config_path)
    assert expected_config._source_hash != update_config._source_hash

    expected_config._source_hash = update_config._source_hash
    assert expected_config == update_config


@pytest.mark.usefixtures("mock_config")
def test_config_rediscover(tmp_path: Path, mock_home: Path):
    """Check that new files are picked up and filters work."""
    config = Config()
    assert config._source_files == []

    home_config_path = mock_home / f"{CONFIG_FILE_STEM}.yml"
    home_config_path.touch()
    expected_config_path = tmp_path / f"{CONFIG_FILE_STEM}.yml"
    config_paths_default = config.rediscover()

    assert config_paths_default == [home_config_path, expected_config_path]
    assert config_paths_default == config._source_files

    config_paths_no_home = config.rediscover(include_home_dir=False)

    assert config_paths_no_home == [expected_config_path]
    assert config_paths_no_home == config._source_files

    project_config_path = tmp_path / f"project/{CONFIG_FILE_STEM}.yml"
    project_config_path.touch()

    config_paths_with_project = config.rediscover()

    assert config_paths_with_project == [
        home_config_path,
        expected_config_path,
        project_config_path,
    ]
    assert config_paths_with_project == config._source_files

    config_paths_only_project = config.rediscover(include_home_dir=False, lookup_depth=1)

    assert config_paths_only_project == [project_config_path]
    assert config_paths_only_project == config._source_files


@pytest.mark.usefixtures("mock_config")
def test_config_init_project(tmp_path: Path, mock_home: Path):
    """New config and schema are created."""
    from pyglotaran_extras import CONFIG
    from pyglotaran_extras import SCRIPT_DIR

    expected_config_file = SCRIPT_DIR / f"{CONFIG_FILE_STEM}.yml"
    expected_schema_file = SCRIPT_DIR / f"{CONFIG_FILE_STEM}.schema.json"

    assert tmp_path in SCRIPT_DIR.parents
    assert expected_config_file.is_file() is False
    assert expected_schema_file.is_file() is False

    assert expected_config_file not in CONFIG._source_files

    initial_config = CONFIG.model_copy(deep=True)
    CONFIG.init_project()

    assert expected_config_file.is_file() is True
    assert expected_schema_file.is_file() is True

    assert expected_config_file in CONFIG._source_files

    reloaded_config = Config()
    reloaded_config._source_files = [
        SCRIPT_DIR.parent / f"{CONFIG_FILE_STEM}.yml",
        expected_config_file,
    ]
    reloaded_config.reload()

    assert CONFIG == reloaded_config  # noqa: SIM300
    assert CONFIG != initial_config  # noqa: SIM300


def test_find_config_in_dir(tmp_path: Path):
    """Find one or two config files if present."""
    assert len(list(find_config_in_dir(tmp_path))) == 0

    yml_config_path = tmp_path / f"yml/{CONFIG_FILE_STEM}.yml"
    yml_config_path.parent.mkdir()
    yml_config_path.touch()

    assert next(find_config_in_dir(yml_config_path.parent)) == yml_config_path

    yaml_config_path = tmp_path / f"yaml/{CONFIG_FILE_STEM}.yaml"
    yaml_config_path.parent.mkdir()
    yaml_config_path.touch()

    assert next(find_config_in_dir(yaml_config_path.parent)) == yaml_config_path

    multi_config_dir = tmp_path / "multi"
    multi_config_dir.mkdir()
    (multi_config_dir / f"{CONFIG_FILE_STEM}.yml").touch()
    (multi_config_dir / f"{CONFIG_FILE_STEM}.yaml").touch()
    (multi_config_dir / f"{CONFIG_FILE_STEM}.json").touch()
    (multi_config_dir / f"{CONFIG_FILE_STEM}1.yml").touch()

    assert len(list(find_config_in_dir(multi_config_dir))) == 2


def test_discover_config_files(tmp_path: Path, mock_home: Path):
    """Discover all config files in the correct order."""
    file_name = f"{CONFIG_FILE_STEM}.yml"
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


def test_load_config_files(tmp_path: Path, test_config_file: Path):
    """Read configs and add source path."""
    empty_config_file = tmp_path / f"{CONFIG_FILE_STEM}.yml"
    empty_config_file.touch()

    empty_file_loaded = load_config_files([empty_config_file])

    empty_config = next(empty_file_loaded)

    assert empty_config.model_dump() == Config().model_dump()
    assert empty_config._source_files == [empty_config_file]
    assert generator_is_exhausted(empty_file_loaded) is True

    two_configs = load_config_files([empty_config_file, test_config_file])

    empty_config = next(two_configs)

    assert empty_config.model_dump() == Config().model_dump()
    assert empty_config._source_files == [empty_config_file]

    test_config_values = YAML().load(test_config_file)

    expected_config = Config.model_validate(test_config_values)

    test_config = next(two_configs)

    assert test_config.model_dump() == expected_config.model_dump()
    assert test_config._source_files == [test_config_file]
    assert generator_is_exhausted(two_configs) is True


def test_merge_configs():
    """Check that the most right config overrides other values.

    Since we already tested all permutations for 2 configs in ``test_config_merge``
    this test can fucus on the case with more than 2 configs.
    """
    assert merge_configs([]) == Config()

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

    file_name = f"{CONFIG_FILE_STEM}.yml"
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


@pytest.fixture
def import_load_script(tmp_path: Path):
    """Copy import load script into tmp_path."""
    src_path = TEST_DATA / "config/run_load_config_on_import.py"
    dest_path = tmp_path / "run_load_config_on_import.py"
    copyfile(src_path, dest_path)
    return dest_path


def test_load_config_on_import_no_local_config(tmp_path: Path, import_load_script: Path):
    """No local config found."""
    subprocess.run([sys.executable, import_load_script])

    assert (tmp_path / "source_files.json").is_file() is True
    assert (tmp_path / "plotting.json").is_file() is True

    source_files = json.loads((tmp_path / "source_files.json").read_text())
    assert any(tmp_path in Path(source_file).parents for source_file in source_files) is False, (
        f"{tmp_path=}, {source_files=}"
    )


def check_config(tmp_path: Path, plot_config_path: Path):
    """Only testing specific aspects makes this test resilient against config pollution.

    Since the config uses a locality hierarchy even if a user has a user config it will be
    overridden.
    """
    assert (tmp_path / "source_files.json").is_file() is True
    assert (tmp_path / "plotting.json").is_file() is True

    source_files = json.loads((tmp_path / "source_files.json").read_text())
    assert any(plot_config_path == Path(source_file) for source_file in source_files) is True, (
        f"{tmp_path=}, {source_files=}"
    )

    plotting_dict = json.loads((tmp_path / "plotting.json").read_text())

    assert (
        plotting_dict["general"]["default_args_override"]["will_update_arg"] == "will change arg"
    )
    assert plotting_dict["general"]["default_args_override"]["will_be_kept_arg"] == "general arg"
    assert (
        plotting_dict["general"]["axis_label_override"]["will_update_label"] == "will change label"
    )
    assert plotting_dict["general"]["axis_label_override"]["will_be_kept_label"] == "general label"

    assert (
        plotting_dict["test_func"]["default_args_override"]["will_update_arg"] == "test_func arg"
    )
    assert plotting_dict["test_func"]["default_args_override"]["will_be_added_arg"] == "new arg"
    assert (
        plotting_dict["test_func"]["axis_label_override"]["will_update_label"] == "test_func label"
    )
    assert plotting_dict["test_func"]["axis_label_override"]["will_be_added_label"] == "new label"


def test_load_config_on_import_local_config(
    tmp_path: Path, import_load_script: Path, test_config_file: Path
):
    """Check that config is properly loaded at import."""
    dest_path = tmp_path / f"{CONFIG_FILE_STEM}.yml"
    copyfile(test_config_file, dest_path)

    subprocess.run([sys.executable, import_load_script])

    check_config(tmp_path, dest_path)


def test_load_config_on_import_none_root_import(
    tmp_path: Path, import_load_script: Path, test_config_file: Path
):
    """Also works something from non root ``pyglotaran_extras`` was imported first."""
    dest_path = tmp_path / f"{CONFIG_FILE_STEM}.yml"
    copyfile(test_config_file, dest_path)

    import_load_script.write_text(
        "from pyglotaran_extras.inspect.a_matrix import show_a_matrixes\n"
        f"{import_load_script.read_text()}"
    )

    subprocess.run([sys.executable, import_load_script])

    check_config(tmp_path, dest_path)


def test_load_config_on_import_broken_config(tmp_path: Path, import_load_script: Path):
    """Broken config does not brake script."""
    src_path = TEST_DATA / "config/broken_config.yml"
    dest_path = tmp_path / f"{CONFIG_FILE_STEM}.yml"
    copyfile(src_path, dest_path)

    res = subprocess.run([sys.executable, import_load_script], text=True, capture_output=True)
    assert res.returncode == 0

    expected_errors = [
        "plotting.general.axis_label_override.target_name",
        "plotting.general.axis_label_override.not_allowed_general",
        "plotting.general.invalid_kw_general",
        "plotting.test_func.axis_label_override.target_name",
        "plotting.test_func.axis_label_override.not_allowed_test_func",
        "plotting.test_func.invalid_kw_test_func",
        "invalid_kw_root",
    ]

    assert f"{len(expected_errors)} validation errors for Config" in res.stderr

    assert f"Source path: {dest_path.as_posix()}" in res.stderr

    for error in expected_errors:
        assert error in res.stderr, f"Expected error '{error}' not found in stderr"


@pytest.mark.usefixtures("mock_config")
def test_create_config_schema(tmp_path: Path, test_config_values: dict[str, Any]):
    """A valid config doesn't cause any schema validation errors."""

    @use_plot_config()
    def other(
        will_be_kept_arg="default update",
    ):
        pass

    @use_plot_config()
    def test_func(
        will_update_arg="default update",
        will_be_kept_arg="default keep",
        will_be_added_arg="default add",
    ):
        pass

    json_schema = json.loads(create_config_schema(tmp_path).read_text())
    expected_schema = json.loads(
        (TEST_DATA / f"config/{CONFIG_FILE_STEM}.schema.json").read_text()
        if Version(pydantic_version) >= Version("2.9")
        else (TEST_DATA / f"config/{CONFIG_FILE_STEM}.schema.pydantic_lt_2_9.json").read_text()
    )

    assert json_schema == expected_schema

    validator = Draft202012Validator(json_schema)

    assert generator_is_exhausted(validator.iter_errors(test_config_values))


@pytest.mark.usefixtures("mock_config")
def test_create_config_schema_errors(tmp_path: Path):
    """A broken config does cause schema validation errors."""

    @use_plot_config()
    def test_func(
        will_update_arg="default update",
        # we don't define will_be_added_arg and will_be_kept_arg because we want them to cause
        # errors
    ):
        pass

    json_schema = json.loads(create_config_schema(tmp_path).read_text())
    expected_schema = (
        json.loads((TEST_DATA / "config/broken_config.schema.json").read_text())
        if Version(pydantic_version) >= Version("2.9")
        else json.loads(
            (TEST_DATA / "config/broken_config.schema.pydantic_lt_2_9.json").read_text()
        )
    )

    assert json_schema == expected_schema

    validator = Draft202012Validator(json_schema)

    broken_config_dict = YAML().load(TEST_DATA / "config/broken_config.yml")
    expected_errors = [
        "Additional properties are not allowed ('invalid_kw_root' was unexpected)",
        "Additional properties are not allowed ('invalid_kw_general' was unexpected)",
        "Additional properties are not allowed ('will_be_kept_arg' was unexpected)",
        "{'will_update_label': 'will change label', 'will_be_kept_label': {'not_allowed_general': "
        "False}} is not valid under any of the given schemas",
        "Additional properties are not allowed ('invalid_kw_test_func' was unexpected)",
        "Additional properties are not allowed ('will_be_added_arg' was unexpected)",
        "{'not_allowed_test_func': False} is not valid under any of the given schemas",
    ]
    for error, expected_error in zip(
        validator.iter_errors(broken_config_dict), expected_errors, strict=True
    ):
        assert error.message.splitlines()[0] == expected_error, [
            error.message.splitlines()[0] for error in validator.iter_errors(broken_config_dict)
        ]
