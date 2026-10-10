"""Test suite for ``pyglotaran_extras``."""

from __future__ import annotations

from pathlib import Path

from glotaran import __version__ as glotaran_version
from packaging.version import Version

PYGLOTARAN_GE_0_8 = Version(glotaran_version) >= Version("0.8.0.dev0")
"""Whether the installed pyglotaran uses the v0.8 API (dataset-free scheme, new result)."""

TEST_DATA = Path(__file__).parent / "data"
