from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import pyglotaran_extras
from pyglotaran_extras.deprecation.deprecation_utils import OverDueDeprecation
from pyglotaran_extras.deprecation.deprecation_utils import PyglotaranExtrasApiDeprecationWarning
from pyglotaran_extras.deprecation.deprecation_utils import check_overdue
from pyglotaran_extras.deprecation.deprecation_utils import parse_version
from pyglotaran_extras.deprecation.deprecation_utils import pyglotaran_extras_version
from pyglotaran_extras.deprecation.deprecation_utils import warn_deprecated

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

OVERDUE_ERROR_MESSAGE = (
    "Support for 'pyglotaran_extras.deprecation.deprecation_utils.parse_version' "
    "was supposed to be dropped in version: '0.6.0'.\n"
    "Current version is: '1.0.0'"
)

DEP_UTILS_QUALNAME = "pyglotaran_extras.deprecation.deprecation_utils"

DEPRECATION_QUAL_NAME = f"{DEP_UTILS_QUALNAME}.parse_version(version_str)"
NEW_QUAL_NAME = f"{DEP_UTILS_QUALNAME}.check_overdue(qualnames)"

DEPRECATION_WARN_MESSAGE = (
    "Usage of 'pyglotaran_extras.deprecation.deprecation_utils.parse_version(version_str)' "
    "was deprecated, use "
    "'pyglotaran_extras.deprecation.deprecation_utils.check_overdue(qualnames)' "
    "instead.\nThis usage will be an error in version: '0.6.0'."
)


@pytest.fixture
def pyglotaran_extras_0_3_0(monkeypatch: MonkeyPatch):
    """Mock pyglotaran_extras version to always be 0.3.0 for the test."""
    monkeypatch.setattr(
        pyglotaran_extras.deprecation.deprecation_utils,
        "pyglotaran_extras_version",
        lambda: "0.3.0",
    )
    yield


@pytest.fixture
def pyglotaran_extras_1_0_0(monkeypatch: MonkeyPatch):
    """Mock pyglotaran_extras version to always be 1.0.0 for the test."""
    monkeypatch.setattr(
        pyglotaran_extras.deprecation.deprecation_utils,
        "pyglotaran_extras_version",
        lambda: "1.0.0",
    )
    yield


def test_pyglotaran_extras_version():
    """Versions are the same."""
    assert pyglotaran_extras_version() == pyglotaran_extras.__version__


@pytest.mark.parametrize(
    "version_str, expected",
    (
        ("0.0.1", (0, 0, 1)),
        ("0.0.1.post", (0, 0, 1)),
        ("0.0.1-dev", (0, 0, 1)),
        ("0.0.1-dev.post", (0, 0, 1)),
    ),
)
def test_parse_version(version_str: str, expected: tuple[int, int, int]):
    """Valid version strings."""
    assert parse_version(version_str) == expected


@pytest.mark.parametrize(
    "version_str",
    ("1", "0.1", "a.b.c"),
)
def test_parse_version_errors(version_str: str):
    """Invalid version strings"""
    with pytest.raises(ValueError, match=f"'{version_str}'"):
        parse_version(version_str)


@pytest.mark.usefixtures("pyglotaran_extras_0_3_0")
def test_check_overdue_no_raise(monkeypatch: MonkeyPatch):
    """Current version smaller then drop_version."""
    check_overdue(
        deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
        to_be_removed_in_version="0.6.0",
    )


@pytest.mark.usefixtures("pyglotaran_extras_1_0_0")
def test_check_overdue_raises(monkeypatch: MonkeyPatch):
    """Current version is equal or bigger than drop_version."""
    with pytest.raises(OverDueDeprecation) as excinfo:
        check_overdue(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )

    assert str(excinfo.value) == OVERDUE_ERROR_MESSAGE


@pytest.mark.usefixtures("pyglotaran_extras_0_3_0")
def test_warn_deprecated():
    """Warning gets shown when all is in order."""
    with pytest.warns(PyglotaranExtrasApiDeprecationWarning) as record:
        warn_deprecated(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )

        assert len(record) == 1
        assert record[0].message.args[0] == DEPRECATION_WARN_MESSAGE
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.usefixtures("pyglotaran_extras_1_0_0")
def test_warn_deprecated_overdue_deprecation(monkeypatch: MonkeyPatch):
    """Current version is equal or bigger than drop_version."""

    with pytest.raises(OverDueDeprecation) as excinfo:
        warn_deprecated(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )
    assert str(excinfo.value) == OVERDUE_ERROR_MESSAGE


@pytest.mark.filterwarnings("ignore:Usage")
@pytest.mark.xfail(strict=True, reason="Dev version aren't checked")
def test_warn_deprecated_no_overdue_deprecation_on_dev(monkeypatch: MonkeyPatch):
    """Current version is equal or bigger than drop_version but it's a dev version."""
    monkeypatch.setattr(
        pyglotaran_extras.deprecation.deprecation_utils,
        "glotaran_version",
        lambda: "0.6.0-dev",
    )

    with pytest.raises(OverDueDeprecation):
        warn_deprecated(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )
