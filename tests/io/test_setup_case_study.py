from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

from tests.conftest import wrapped_get_script_dir

from pyglotaran_extras.io.setup_case_study import get_script_dir
from pyglotaran_extras.io.setup_case_study import setup_case_study

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_get_script_dir():
    """Called directly"""
    expected = Path(__file__).parent

    assert get_script_dir() == expected


def test_get_script_dir_in_closure():
    """Called inside other function imported from different file"""
    expected = Path(__file__).parent

    assert wrapped_get_script_dir() == expected


def test_get_script_dir_tmp_path(tmp_path: Path):
    """File in temp folder"""
    tmp_file = tmp_path / "foo.py"
    content = dedent(
        """
        from pyglotaran_extras.io.setup_case_study import get_script_dir
        print(get_script_dir())
        """
    )
    tmp_file.write_text(content)
    printed_result = subprocess.run(
        " ".join([sys.executable, tmp_file.resolve().as_posix()]), capture_output=True, shell=True
    )
    result = printed_result.stdout.decode().rstrip("\n\r")

    assert printed_result.returncode == 0
    assert Path(result) == tmp_path.resolve()


def test_setup_case_study(monkeypatch: MonkeyPatch, tmp_path: Path):
    """Default settings"""
    mock_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: mock_home)

    results_folder, script_folder = setup_case_study()

    assert mock_home.exists()
    assert results_folder.exists()
    assert results_folder == mock_home / "pyglotaran_results/io"
    assert script_folder == Path(__file__).parent


def test_setup_case_study_custom(tmp_path: Path):
    """Custom settings"""
    results_folder_root = tmp_path / "foo"

    results_folder, script_folder = setup_case_study(
        output_folder_name="foo", results_folder_root=tmp_path
    )

    assert results_folder_root.exists()
    assert results_folder.exists()
    assert results_folder == results_folder_root / "io"
    assert script_folder == Path(__file__).parent
